import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  IconButton,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Avatar,
  Button,
  Card,
  CardContent,
  Grid,
  useTheme,
  alpha
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import DeleteIcon from '@mui/icons-material/Delete';
import PersonIcon from '@mui/icons-material/Person';
import api from '../../services/apiService';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useAuth } from '../../hooks/useAuth';

// Direct API service calls to prevent circular dependencies
const aiChatServices = {
  createChat: (initialMessage) => 
    api.post('/admin/ai-chat/chats', { initialMessage }).then(response => response.data),
  
  getAllChats: () => 
    api.get('/admin/ai-chat/chats').then(response => response.data),
  
  getChat: (chatId) => 
    api.get(`/admin/ai-chat/chats/${chatId}`).then(response => response.data),
  
  sendMessage: (chatId, message) => 
    api.post(`/admin/ai-chat/chats/${chatId}/messages`, { message }).then(response => response.data),
  
  clearChatHistory: () => 
    api.post('/admin/ai-chat/clear-history').then(response => response.data)
};

const AiChatInterface = () => {
  const theme = useTheme();
  const { user } = useAuth();
  
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [error, setError] = useState(null);
  
  const messagesEndRef = useRef(null);
  
  // Fetch all chats for current user
  useEffect(() => {
    const fetchChats = async () => {
      try {
        setIsLoading(true);
        const chatData = await aiChatServices.getAllChats();
        setChats(chatData);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching chats:', error);
        setError('Failed to load chat history');
        setIsLoading(false);
      }
    };
    
    fetchChats();
  }, []);
  
  // Scroll to bottom of message container
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeChat]);
  
  // Create a new chat
  const handleCreateChat = async () => {
    if (!inputMessage.trim()) return;
    
    try {
      setIsCreatingChat(true);
      setError(null);
      
      const newChat = await aiChatServices.createChat(inputMessage);
      
      setChats(prevChats => [newChat, ...prevChats]);
      setActiveChat(newChat);
      setInputMessage('');
      setIsCreatingChat(false);
    } catch (error) {
      console.error('Error creating chat:', error);
      setError('Failed to create new chat');
      setIsCreatingChat(false);
    }
  };
  
  // Send a message in existing chat
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !activeChat) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      // Add optimistic update
      const optimisticId = Date.now().toString();
      const optimisticMessage = {
        _id: optimisticId,
        content: inputMessage,
        role: 'user',
        timestamp: new Date(),
        pending: true
      };
      
      setActiveChat(prevChat => ({
        ...prevChat,
        messages: [...prevChat.messages, optimisticMessage]
      }));
      
      // Store the message content for potential retries
      // (We'll use it in future enhancements)
      
      // Reset input field
      setInputMessage('');
      
      // Make API call with timeout handling
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timed out')), 30000)
      );
      
      const responsePromise = aiChatServices.sendMessage(activeChat._id, optimisticMessage.content);
      
      const response = await Promise.race([responsePromise, timeoutPromise]);
      
      // Update chat with real response
      setActiveChat(prevChat => {
        // Find the optimistic message and replace it with the real one
        const updatedMessages = prevChat.messages.map(msg => 
          msg._id === optimisticId ? { ...msg, pending: false } : msg
        );
        
        // Add AI response
        updatedMessages.push({
          _id: Date.now().toString() + '-response',
          content: response.aiResponse,
          role: 'ai',
          timestamp: new Date()
        });
        
        return {
          ...prevChat,
          messages: updatedMessages
        };
      });
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // More descriptive error message
      let errorMessage = 'Failed to send message';
      if (error.message === 'Request timed out') {
        errorMessage = 'Request timed out. The server may be busy, please try again.';
      } else if (error.response) {
        if (error.response.status === 429) {
          errorMessage = 'Too many requests. Please wait a moment before trying again.';
        } else if (error.response.status >= 500) {
          errorMessage = 'Server error. Our team has been notified.';
        }
      } else if (!navigator.onLine) {
        errorMessage = 'You appear to be offline. Please check your internet connection.';
      }
      
      setError(errorMessage);
      setIsLoading(false);
      
      // Remove optimistic update on error
      setActiveChat(prevChat => ({
        ...prevChat,
        messages: prevChat.messages.filter(msg => !msg.pending)
      }));
      
      // Offer ability to retry by putting the message back in the input field
      if (error.message !== 'Request timed out') {
        setInputMessage(prevChat => prevChat.messages.find(msg => msg.pending)?.content || '');
      }
    }
  };
  
  // Handle selecting a chat from history
  const handleSelectChat = async (chat) => {
    try {
      setIsLoading(true);
      const fullChat = await aiChatServices.getChat(chat._id);
      setActiveChat(fullChat);
      setIsLoading(false);
    } catch (error) {
      console.error('Error loading chat:', error);
      setError('Failed to load chat');
      setIsLoading(false);
    }
  };
  
  // Clear chat history
  const handleClearChat = async () => {
    if (window.confirm('Are you sure you want to clear this conversation? This will reset the AI\'s memory of your conversation.')) {
      try {
        setIsLoading(true);
        await aiChatServices.clearChatHistory();
        // First clear the active chat in the UI
        setActiveChat(null);
        setInputMessage('');
        // Then refetch the chat list to update the sidebar
        const chatData = await aiChatServices.getAllChats();
        setChats(chatData);
        setIsLoading(false);
      } catch (error) {
        console.error('Error clearing chat:', error);
        setError('Failed to clear chat');
        setIsLoading(false);
      }
    }
  };
  
  // Message component with enhanced markdown support
  const MessageContent = ({ content }) => {
    return (
      <ReactMarkdown
        children={content}
        components={{
          // Enhanced code block rendering
          code({node, inline, className, children, ...props}) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                children={String(children).replace(/\n$/, '')}
                style={atomDark}
                language={match[1]}
                PreTag="div"
                wrapLines={true}
                showLineNumbers={true}
                {...props}
              />
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          // Enhanced heading styling
          h1: ({node, ...props}) => (
            <Typography 
              variant="h5" 
              sx={{ 
                mt: 2, 
                mb: 1, 
                color: theme.palette.primary.main,
                fontWeight: 600
              }} 
              {...props} 
            />
          ),
          h2: ({node, ...props}) => (
            <Typography 
              variant="h6" 
              sx={{ 
                mt: 2, 
                mb: 1, 
                color: theme.palette.secondary.main,
                fontWeight: 600
              }} 
              {...props} 
            />
          ),
          // Enhanced paragraph styling for better readability
          p: ({node, ...props}) => (
            <Typography 
              variant="body1" 
              paragraph 
              sx={{ 
                lineHeight: 1.6, 
                my: 1 
              }} 
              {...props} 
            />
          ),
          // Enhanced list styling
          ul: ({node, ...props}) => (
            <Box component="ul" sx={{ pl: 2, my: 1 }} {...props} />
          ),
          li: ({node, ...props}) => (
            <Box 
              component="li" 
              sx={{ 
                my: 0.5,
                lineHeight: 1.5
              }} 
              {...props} 
            />
          ),
          // Enhanced table styling for data
          table: ({node, ...props}) => (
            <Box 
              component="table" 
              sx={{ 
                borderCollapse: 'collapse',
                width: '100%',
                my: 2,
                border: `1px solid ${theme.palette.divider}`
              }} 
              {...props} 
            />
          ),
          th: ({node, ...props}) => (
            <Box 
              component="th" 
              sx={{ 
                border: `1px solid ${theme.palette.divider}`,
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                padding: 1,
                textAlign: 'left'
              }} 
              {...props} 
            />
          ),
          td: ({node, ...props}) => (
            <Box 
              component="td" 
              sx={{ 
                border: `1px solid ${theme.palette.divider}`,
                padding: 1
              }} 
              {...props} 
            />
          )
        }}
      />
    );
  };
  
  return (
    <Box
      sx={{
        display: 'flex',
        height: 'calc(100vh - 120px)',
        bgcolor: 'background.default',
        borderRadius: 2,
        overflow: 'hidden',
        boxShadow: 3
      }}
    >
      {/* Chat list sidebar */}
      <Box
        sx={{
          width: 300,
          borderRight: `1px solid ${theme.palette.divider}`,
          bgcolor: alpha(theme.palette.background.paper, 0.8),
          display: { xs: activeChat ? 'none' : 'block', sm: 'block' }
        }}
      >
        <Box
          sx={{
            p: 2,
            borderBottom: `1px solid ${theme.palette.divider}`,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AutoAwesomeIcon color="primary" />
            AI Assistant
          </Typography>
          <Button
            variant="contained"
            size="small"
            onClick={() => {
              setActiveChat(null);
              setInputMessage('');
            }}
          >
            New Chat
          </Button>
        </Box>
        
        {isLoading && !activeChat && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress size={30} />
          </Box>
        )}
        
        <List sx={{ overflow: 'auto', maxHeight: 'calc(100vh - 180px)' }}>
          {chats.length > 0 ? (
            chats.map((chat) => (
              <ListItem
                key={chat._id}
                button
                selected={activeChat && activeChat._id === chat._id}
                onClick={() => handleSelectChat(chat)}
                sx={{
                  '&.Mui-selected': {
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                  }
                }}
              >
                <ListItemText
                  primary={chat.title || 'New Conversation'}
                  secondary={new Date(chat.updatedAt).toLocaleDateString()}
                  primaryTypographyProps={{
                    noWrap: true,
                    sx: { fontWeight: activeChat && activeChat._id === chat._id ? 600 : 400 }
                  }}
                />
              </ListItem>
            ))
          ) : (
            !isLoading && (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography color="text.secondary">No conversations yet</Typography>
              </Box>
            )
          )}
        </List>
      </Box>

      {/* Chat interface */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeChat ? (
          <>
            {/* Chat header */}
            <Box
              sx={{
                p: 2,
                borderBottom: `1px solid ${theme.palette.divider}`,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: theme.palette.background.paper
              }}
            >
              <Typography variant="h6">{activeChat.title || 'Conversation with AI Assistant'}</Typography>
              <IconButton color="error" onClick={handleClearChat} title="Clear conversation history">
                <DeleteIcon />
              </IconButton>
            </Box>
            
            {/* Messages */}
            <Box
              sx={{
                flex: 1,
                overflowY: 'auto',
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                gap: 2
              }}
            >
              {activeChat.messages.map((message) => (
                <Box
                  key={message._id}
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    mb: 2,
                    opacity: message.pending ? 0.7 : 1
                  }}
                >
                  <Avatar
                    sx={{
                      mr: 2,
                      bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main'
                    }}
                  >
                    {message.role === 'user' ? <PersonIcon /> : <AutoAwesomeIcon />}
                  </Avatar>
                  <Paper
                    elevation={message.role === 'ai' ? 2 : 1}
                    sx={{
                      p: 2,
                      maxWidth: '85%',
                      borderRadius: 2,
                      borderLeft: message.role === 'ai' ? `4px solid ${theme.palette.secondary.main}` : 'none',
                      bgcolor: message.role === 'user'
                        ? alpha(theme.palette.primary.main, 0.1)
                        : alpha(theme.palette.secondary.main, 0.1)
                    }}
                  >
                    <Typography
                      variant="subtitle2"
                      color={message.role === 'user' ? 'primary' : 'secondary'}
                      gutterBottom
                    >
                      {message.role === 'user' ? user?.name || 'You' : 'SafeStreets AI'}
                    </Typography>
                    
                    <Box sx={{ 
                      '& p': { mt: 0, mb: 1 }, 
                      '& code': { 
                        p: 0.5, 
                        borderRadius: 1, 
                        bgcolor: alpha('#000', 0.1) 
                      }
                    }}>
                      <MessageContent content={message.content} />
                    </Box>
                    
                    {message.pending && (
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <CircularProgress size={14} sx={{ mr: 1 }} />
                        <Typography variant="caption" color="text.secondary">Sending...</Typography>
                      </Box>
                    )}
                  </Paper>
                </Box>
              ))}
              <div ref={messagesEndRef} />
            </Box>
            
            {/* Input area */}
            <Box
              sx={{
                p: 2,
                backgroundColor: theme.palette.background.paper,
                borderTop: `1px solid ${theme.palette.divider}`
              }}
            >
              <Box
                component="form"
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSendMessage();
                }}
                sx={{ 
                  display: 'flex', 
                  gap: 1, 
                  position: 'relative',
                  alignItems: 'center'
                }}
              >
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="Type your message here..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  disabled={isLoading}
                  multiline
                  maxRows={4}
                  onKeyDown={(e) => {
                    // Send message on Enter key (but not with Shift for new line)
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      if (inputMessage.trim() && !isLoading) {
                        handleSendMessage();
                      }
                    }
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 2
                    }
                  }}
                />
                <IconButton
                  color="primary"
                  type="submit"
                  disabled={isLoading || !inputMessage.trim()}
                  sx={{
                    height: 56,
                    width: 56,
                    alignSelf: 'flex-end',
                    zIndex: 2,
                    position: 'relative'
                  }}
                >
                  {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
                </IconButton>
              </Box>
              {error && (
                <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
                  {error}
                </Typography>
              )}
            </Box>
          </>
        ) : (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              p: 3
            }}
          >
            <Card sx={{ maxWidth: 600, width: '100%', mb: 4, borderTop: `4px solid ${theme.palette.secondary.main}` }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Avatar sx={{ 
                    mr: 2, 
                    bgcolor: 'secondary.main',
                    width: 56,
                    height: 56
                  }}>
                    <AutoAwesomeIcon fontSize="large" />
                  </Avatar>
                  <Box>
                    <Typography variant="h4" sx={{ fontWeight: 600 }}>
                      SafeStreets AI Assistant
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary">
                      Professional Consultation for Infrastructure Management
                    </Typography>
                  </Box>
                </Box>
                <Typography variant="body1" paragraph sx={{ mb: 3 }}>
                  Welcome to your professional road infrastructure management assistant, powered by Google's Gemini AI. How can I assist you today?
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ 
                      backgroundColor: alpha(theme.palette.secondary.main, 0.1), 
                      p: 2, 
                      borderRadius: 2,
                      height: '100%'
                    }}>
                      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, color: theme.palette.secondary.main }}>
                        Technical Analysis
                      </Typography>
                      <Typography variant="body2" component="div">
                        <Box component="ul" sx={{ pl: 2, mb: 0 }}>
                          <Box component="li" sx={{ mb: 1 }}>Detailed damage classification</Box>
                          <Box component="li" sx={{ mb: 1 }}>Severity assessment frameworks</Box>
                          <Box component="li" sx={{ mb: 1 }}>Priority-based maintenance planning</Box>
                          <Box component="li" sx={{ mb: 0 }}>Cost-benefit analysis</Box>
                        </Box>
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ 
                      backgroundColor: alpha(theme.palette.primary.main, 0.1), 
                      p: 2, 
                      borderRadius: 2,
                      height: '100%' 
                    }}>
                      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, color: theme.palette.primary.main }}>
                        Administrative Support
                      </Typography>
                      <Typography variant="body2" component="div">
                        <Box component="ul" sx={{ pl: 2, mb: 0 }}>
                          <Box component="li" sx={{ mb: 1 }}>Resource allocation optimization</Box>
                          <Box component="li" sx={{ mb: 1 }}>Workflow management strategies</Box>
                          <Box component="li" sx={{ mb: 1 }}>Regulatory compliance assistance</Box>
                          <Box component="li" sx={{ mb: 0 }}>Data-driven decision support</Box>
                        </Box>
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
            
            <Box sx={{ width: '100%', maxWidth: 600 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Ask anything about road maintenance, damage assessment, or SafeStreets system..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                multiline
                rows={3}
                sx={{ mb: 2 }}
              />
              <Button
                fullWidth
                variant="contained"
                color="primary"
                size="large"
                startIcon={isCreatingChat ? <CircularProgress size={20} color="inherit" /> : <AutoAwesomeIcon />}
                onClick={handleCreateChat}
                disabled={isCreatingChat || !inputMessage.trim()}
              >
                Start New Conversation
              </Button>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default AiChatInterface;
