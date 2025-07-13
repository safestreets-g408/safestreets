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
    <Paper
      elevation={0}
      sx={{
        display: 'flex',
        height: 'calc(100vh - 160px)',
        bgcolor: 'background.default',
        borderRadius: 1,
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider'
      }}
    >
      {/* Chat list sidebar */}
      <Box
        sx={{
          width: 280,
          borderRight: '1px solid',
          borderColor: 'divider',
          bgcolor: 'background.default',
          display: { xs: activeChat ? 'none' : 'block', sm: 'block' }
        }}
      >
        <Box
          sx={{
            p: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <Typography 
            variant="subtitle2" 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 0.75,
              fontWeight: 500,
              color: 'text.primary'
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: '1rem' }} color="primary" />
            Conversations
          </Typography>
          <Button
            variant="text"
            color="primary"
            size="small"
            sx={{ 
              minWidth: 0,
              textTransform: 'none',
              fontWeight: 500,
              px: 1
            }}
            onClick={() => {
              setActiveChat(null);
              setInputMessage('');
            }}
          >
            New
          </Button>
        </Box>
        
        {isLoading && !activeChat && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress size={30} />
          </Box>
        )}
        
        <List 
          sx={{ 
            overflow: 'auto', 
            maxHeight: 'calc(100vh - 220px)',
            py: 0.5,
            px: 1
          }}
        >
          {chats.length > 0 ? (
            chats.map((chat) => (
              <ListItem
                key={chat._id}
                button
                disableGutters
                selected={activeChat && activeChat._id === chat._id}
                onClick={() => handleSelectChat(chat)}
                sx={{
                  mb: 0.5,
                  borderRadius: 1,
                  py: 1,
                  px: 1.5,
                  '&.Mui-selected': {
                    bgcolor: theme => alpha(theme.palette.primary.main, 0.08),
                    '&:hover': {
                      bgcolor: theme => alpha(theme.palette.primary.main, 0.12)
                    }
                  },
                  '&:hover': {
                    bgcolor: theme => alpha(theme.palette.action.hover, 0.04)
                  }
                }}
              >
                <ListItemText
                  primary={chat.title || 'New Conversation'}
                  secondary={new Date(chat.updatedAt).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                  })}
                  primaryTypographyProps={{
                    noWrap: true,
                    sx: { 
                      fontWeight: activeChat && activeChat._id === chat._id ? 500 : 400,
                      color: 'text.primary',
                      fontSize: '0.85rem'
                    }
                  }}
                  secondaryTypographyProps={{
                    sx: { fontSize: '0.75rem', color: 'text.secondary' }
                  }}
                />
              </ListItem>
            ))
          ) : (
            !isLoading && (
              <Box sx={{ 
                p: 3, 
                textAlign: 'center'
              }}>
                <Typography 
                  color="text.secondary"
                  sx={{ 
                    fontSize: '0.85rem'
                  }}
                >
                  No conversations yet
                </Typography>
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
                borderBottom: '1px solid',
                borderColor: 'divider',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <AutoAwesomeIcon 
                  color="primary"
                  fontSize="small"
                  sx={{ mr: 1 }}
                />
                <Typography 
                  variant="subtitle2"
                  sx={{
                    fontWeight: 500,
                    color: 'text.primary'
                  }}
                >
                  {activeChat.title || 'Conversation with AI Assistant'}
                </Typography>
              </Box>
              <IconButton 
                color="default" 
                size="small"
                sx={{ padding: 0.5 }}
                onClick={handleClearChat} 
                title="Clear conversation history"
              >
                <DeleteIcon fontSize="small" />
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
                gap: 2,
                bgcolor: 'background.default'
              }}
            >
              {activeChat.messages.map((message) => (
                <Box
                  key={message._id}
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    mb: 1.5,
                    opacity: message.pending ? 0.7 : 1
                  }}
                >
                  <Avatar
                    sx={{
                      mr: 1.5,
                      bgcolor: message.role === 'user' 
                        ? theme => alpha(theme.palette.primary.main, 0.1)
                        : theme => alpha(theme.palette.success.main, 0.1),
                      color: message.role === 'user' ? 'primary.main' : 'success.main',
                      width: 32,
                      height: 32
                    }}
                  >
                    {message.role === 'user' ? <PersonIcon fontSize="small" /> : <AutoAwesomeIcon fontSize="small" />}
                  </Avatar>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      maxWidth: '85%',
                      borderRadius: 1,
                      bgcolor: message.role === 'user'
                        ? theme => alpha(theme.palette.primary.main, 0.04)
                        : theme => alpha(theme.palette.background.paper, 1),
                      border: '1px solid',
                      borderColor: 'divider'
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        fontWeight: 500,
                        display: 'block',
                        mb: 0.75,
                        color: message.role === 'user' ? 'text.secondary' : 'text.secondary'
                      }}
                    >
                      {message.role === 'user' ? user?.name || 'You' : 'SafeStreets AI'}
                    </Typography>
                    
                    <Box sx={{ 
                      '& p': { 
                        mt: 0, 
                        mb: 1.5,
                        lineHeight: 1.6,
                        fontSize: '0.85rem',
                        color: 'text.primary'
                      }, 
                      '& code': { 
                        p: 0.5, 
                        borderRadius: 1, 
                        bgcolor: 'background.default',
                        color: 'primary.main',
                        fontFamily: '"Roboto Mono", monospace',
                        fontSize: '0.85rem'
                      },
                      '& ul, & ol': {
                        pl: 2,
                        mb: 1
                      },
                      '& li': {
                        mb: 0.5
                      }
                    }}>
                      <MessageContent content={message.content} />
                    </Box>
                    
                    {message.pending && (
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                        <CircularProgress size={12} sx={{ mr: 0.75 }} />
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
                backgroundColor: 'background.paper',
                borderTop: '1px solid',
                borderColor: 'divider'
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
                  alignItems: 'flex-end'
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
                  minRows={1}
                  maxRows={3}
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
                      borderRadius: 1,
                      backgroundColor: 'background.paper',
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'primary.main'
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'primary.main',
                        borderWidth: '1px'
                      }
                    },
                    '& .MuiOutlinedInput-input': {
                      padding: '12px 14px'
                    }
                  }}
                />
                <IconButton
                  color="primary"
                  type="submit"
                  disabled={isLoading || !inputMessage.trim()}
                  size="small"
                  sx={{
                    height: 40,
                    width: 40,
                    borderRadius: 1,
                    bgcolor: isLoading || !inputMessage.trim()
                      ? 'transparent'
                      : alpha(theme => theme.palette.primary.main, 0.05)
                  }}
                >
                  {isLoading ? <CircularProgress size={20} color="inherit" /> : <SendIcon fontSize="small" />}
                </IconButton>
              </Box>
              {error && (
                <Typography 
                  variant="caption" 
                  color="error" 
                  sx={{ 
                    mt: 1, 
                    display: 'block'
                  }}
                >
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
            <Paper 
              elevation={0}
              sx={{ 
                maxWidth: 700, 
                width: '100%', 
                mb: 4,
                borderRadius: 2,
                overflow: 'hidden',
                bgcolor: 'background.paper',
                border: '1px solid',
                borderColor: theme => theme.palette.divider,
                position: 'relative'
              }}
            >
              <Box sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Avatar sx={{ 
                    mr: 2, 
                    bgcolor: theme => alpha(theme.palette.success.main, 0.1),
                    width: 48,
                    height: 48,
                    color: 'success.main'
                  }}>
                    <AutoAwesomeIcon />
                  </Avatar>
                  <Box>
                    <Typography 
                      variant="h5" 
                      sx={{ 
                        fontWeight: 600,
                        color: 'text.primary',
                        mb: 0.5
                      }}
                    >
                      SafeStreets AI Assistant
                    </Typography>
                    <Typography 
                      variant="body2" 
                      sx={{
                        color: 'text.secondary',
                        fontWeight: 400
                      }}
                    >
                      Professional Consultation for Infrastructure Management
                    </Typography>
                  </Box>
                </Box>

                <Typography 
                  variant="body2" 
                  sx={{ 
                    mb: 3,
                    color: 'text.secondary',
                    fontSize: '0.9rem',
                    lineHeight: 1.6
                  }}
                >
                  Welcome to your professional road infrastructure management assistant, powered by Google's Gemini AI. How can I assist you today?
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Paper 
                      elevation={0}
                      sx={{ 
                        p: 2, 
                        borderRadius: 2,
                        height: '100%',
                        border: '1px solid',
                        borderColor: theme => theme.palette.divider
                      }}
                    >
                      <Typography 
                        variant="subtitle2"
                        sx={{ 
                          fontWeight: 600, 
                          mb: 1.5,
                          display: 'flex',
                          alignItems: 'center',
                          color: 'text.primary'
                        }}
                      >
                        Technical Analysis
                      </Typography>
                      <Typography variant="body2" component="div" sx={{ color: 'text.secondary' }}>
                        <Box component="ul" sx={{ pl: 2, mb: 0, mt: 0 }}>
                          <Box component="li" sx={{ mb: 1, fontSize: '0.85rem' }}>Detailed damage classification</Box>
                          <Box component="li" sx={{ mb: 1, fontSize: '0.85rem' }}>Severity assessment frameworks</Box>
                          <Box component="li" sx={{ mb: 1, fontSize: '0.85rem' }}>Priority-based maintenance planning</Box>
                          <Box component="li" sx={{ mb: 0, fontSize: '0.85rem' }}>Cost-benefit analysis</Box>
                        </Box>
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Paper 
                      elevation={0}
                      sx={{ 
                        p: 2, 
                        borderRadius: 2,
                        height: '100%',
                        border: '1px solid',
                        borderColor: theme => theme.palette.divider
                      }}
                    >
                      <Typography 
                        variant="subtitle2"
                        sx={{ 
                          fontWeight: 600, 
                          mb: 1.5,
                          display: 'flex',
                          alignItems: 'center',
                          color: 'text.primary'
                        }}
                      >
                        Administrative Support
                      </Typography>
                      <Typography variant="body2" component="div" sx={{ color: 'text.secondary' }}>
                        <Box component="ul" sx={{ pl: 2, mb: 0, mt: 0 }}>
                          <Box component="li" sx={{ mb: 1, fontSize: '0.85rem' }}>Resource allocation optimization</Box>
                          <Box component="li" sx={{ mb: 1, fontSize: '0.85rem' }}>Workflow management strategies</Box>
                          <Box component="li" sx={{ mb: 1, fontSize: '0.85rem' }}>Regulatory compliance assistance</Box>
                          <Box component="li" sx={{ mb: 0, fontSize: '0.85rem' }}>Data-driven decision support</Box>
                        </Box>
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            </Paper>
            
            <Box sx={{ width: '100%', maxWidth: 700 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Ask anything about road maintenance, damage assessment, or SafeStreets system..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                multiline
                rows={2}
                sx={{ 
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1,
                    backgroundColor: 'background.paper',
                    border: '1px solid',
                    borderColor: 'divider',
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'primary.main'
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'success.main',
                      borderWidth: '1px'
                    }
                  }
                }}
              />
              <Button
                fullWidth
                variant="contained"
                color="primary"
                size="medium"
                startIcon={isCreatingChat ? <CircularProgress size={18} color="inherit" /> : <AutoAwesomeIcon />}
                onClick={handleCreateChat}
                disabled={isCreatingChat || !inputMessage.trim()}
                sx={{
                  py: 1,
                  borderRadius: 1,
                  textTransform: 'none',
                  fontWeight: 500
                }}
              >
                Start New Conversation
              </Button>
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default AiChatInterface;
