import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useTheme, alpha } from '@mui/material/styles';
import {
  Box,
  Paper,
  Typography,
  TextField,
  IconButton,
  List,
  ListItem,
  Avatar,
  Chip,
  InputAdornment,
  CircularProgress,
  Alert,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Send as SendIcon,
  Person as PersonIcon,
  AdminPanelSettings as AdminIcon,
  Close as CloseIcon,
  Assignment as AssignmentIcon,
  MoreVert as MoreVertIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { format, isToday, isYesterday } from 'date-fns';
import { useSocket } from '../../context/SocketContext';
import { chatService } from '../../services/chatService';
import { useAuth } from '../../hooks/useAuth';
import PropTypes from 'prop-types';
import ReportSelectorDialog from './ReportSelectorDialog';
import ReportMessage from './ReportMessage';

const ChatWindow = ({ tenantId, tenantName, contactName, roomType, roomId, onClose }) => {
  const theme = useTheme();
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [typingUsers, setTypingUsers] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [page, setPage] = useState(1);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreMenuAnchorEl, setMoreMenuAnchorEl] = useState(null);
  const [reportSelectorOpen, setReportSelectorOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const typingTimeoutRef = useRef(null);

  const { socket, joinChat, markAsRead, startTyping, stopTyping } = useSocket();
  const { user } = useAuth();

  // Format message timestamp
  const formatMessageTime = (timestamp) => {
    const date = new Date(timestamp);
    if (isToday(date)) {
      return format(date, 'HH:mm');
    } else if (isYesterday(date)) {
      return `Yesterday ${format(date, 'HH:mm')}`;
    } else {
      return format(date, 'MMM dd, HH:mm');
    }
  };
  
  // Handle more menu open
  const handleMoreMenuOpen = (event) => {
    setMoreMenuAnchorEl(event.currentTarget);
  };
  
  // Handle more menu close
  const handleMoreMenuClose = () => {
    setMoreMenuAnchorEl(null);
  };
  
  // Handle opening report selector
  const handleOpenReportSelector = () => {
    setReportSelectorOpen(true);
    handleMoreMenuClose();
  };
  
  // Handle report selection
  const handleSelectReport = async (report) => {
    setReportSelectorOpen(false);
    
    // If no report was selected, do nothing
    if (!report) return;
    
    setSending(true);
    
    try {
      // Format the report as a special text message
      // We'll use a special prefix to identify it as a report in the renderer
      const formattedMessage = `__REPORT_JSON__:${JSON.stringify(report)}`;
      
      // Send as text message type since 'report' is not in the backend's enum
      await chatService.sendMessage(tenantId, {
        message: formattedMessage,
        messageType: 'text'
      });
    } catch (err) {
      console.error('Error sending report:', err);
      setError('Failed to send report');
    } finally {
      setSending(false);
    }
  };

  // Scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Load messages
  const loadMessages = useCallback(async (pageNum = 1, append = false) => {
    try {
      if (pageNum === 1) setLoading(true);
      else setLoadingMore(true);
      
      setError(null); // Clear previous errors

      let response;
      if (roomType === 'field_worker_chat' && roomId) {
        response = await chatService.getFieldWorkerChatMessages(roomId, pageNum);
      } else {
        response = await chatService.getChatMessages(tenantId, pageNum);
      }
      
      if (append) {
        setMessages(prev => [...response.messages, ...prev]);
      } else {
        setMessages(response.messages);
        setTimeout(scrollToBottom, 100);
      }
      
      setHasMore(response.hasMore);
      setPage(pageNum);
    } catch (err) {
      console.error('Error loading messages:', err);
      setError(err.message || 'Failed to load messages. Please try again.');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [tenantId, roomType, roomId, scrollToBottom]);

  // Load more messages
  const loadMoreMessages = () => {
    if (hasMore && !loadingMore) {
      loadMessages(page + 1, true);
    }
  };

  // Handle sending message
  const handleSendMessage = async () => {
    if (!newMessage.trim() || sending) return;

    setSending(true);
    const messageText = newMessage.trim();
    setNewMessage('');

    try {
      // Send via API only (socket will receive the message via backend broadcast)
      await chatService.sendMessage(tenantId, {
        message: messageText,
        messageType: 'text'
      });
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message');
      setNewMessage(messageText); // Restore message on error
    } finally {
      setSending(false);
    }
  };

  // Handle typing
  const handleTyping = () => {
    startTyping(tenantId);
    
    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }
    
    // Stop typing after 3 seconds
    typingTimeoutRef.current = setTimeout(() => {
      stopTyping(tenantId);
    }, 3000);
  };

  // Initialize chat
  useEffect(() => {
    if (tenantId) {
      joinChat(tenantId);
      loadMessages();
      markAsRead(tenantId);
    }
  }, [tenantId, joinChat, loadMessages, markAsRead]);

  // Socket event listeners
  useEffect(() => {
    if (!socket) return;

    const handleNewMessage = (message) => {
      // Enhanced message filtering for different chat types
      const currentTenantIdStr = tenantId ? tenantId.toString() : '';
      const currentRoomIdStr = roomId ? roomId.toString() : '';
      
      let isForThisChat = false;
      
      if (roomType === 'field_worker_chat' && roomId) {
        // For field worker chats, match by roomId
        isForThisChat = 
          (message.chatId && message.chatId.toString() === currentRoomIdStr) ||
          (message.roomId && message.roomId.toString() === currentRoomIdStr);
      } else {
        // For regular tenant chats, match by tenantId
        isForThisChat = 
          (message.tenantId && message.tenantId.toString() === currentTenantIdStr) || // Direct tenantId match 
          (message.chatId === `tenant_${tenantId}`) || // tenant_ID format
          (message.chatId && message.chatId.toString() === currentTenantIdStr) || // Direct chatId match
          (message.adminId && message.fromFieldWorker && user && 
            (message.adminId === user._id || message.adminId === user.id)) || // Direct message to this admin
          (message.chatId && typeof message.chatId === 'string' && message.chatId.includes(tenantId));
      }
      
      if (isForThisChat) {
        // Check for duplicate messages
        setMessages(prev => {
          const existingMessage = prev.find(msg => msg._id === message._id);
          if (existingMessage) {
            return prev;
          }
          return [...prev, message];
        });
        
        setTimeout(scrollToBottom, 100);
        
        // Mark as read if not from current user
        if (message.senderId !== user._id && message.senderId !== user.id) {
          markAsRead(tenantId);
        }
      }
    };

    const handleUserTyping = (data) => {
      if (data.userId !== user._id && data.userId !== user.id) {
        setTypingUsers(prev => {
          if (data.isTyping) {
            return prev.includes(data.userName) ? prev : [...prev, data.userName];
          } else {
            return prev.filter(name => name !== data.userName);
          }
        });
      }
    };

    socket.on('new_message', handleNewMessage);
    socket.on('user_typing', handleUserTyping);

    return () => {
      socket.off('new_message', handleNewMessage);
      socket.off('user_typing', handleUserTyping);
    };
  }, [socket, tenantId, roomId, roomType, user, markAsRead, scrollToBottom]);

  // Cleanup typing timeout
  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  // Safe message renderer
  const renderMessageContent = useCallback((message, messageType) => {
    if (!message) return 'No message content';
    
    // Handle text messages that contain report data (with our special prefix)
    if (typeof message === 'string' && message.startsWith('__REPORT_JSON__:')) {
      try {
        // Extract and parse the report data
        const jsonString = message.replace('__REPORT_JSON__:', '');
        const reportData = JSON.parse(jsonString);
        return <ReportMessage report={reportData} />;
      } catch (err) {
        console.error('Error parsing report message:', err);
        return 'Error displaying report. Please contact support.';
      }
    }
    
    // Handle regular text messages
    if (typeof message === 'string') return message;
    if (typeof message === 'object' && message.text) return message.text;
    if (typeof message === 'object' && message.message) return message.message;
    
    return 'Invalid message format';
  }, []);

  // Safe sender name renderer
  const renderSenderName = useCallback((senderName) => {
    if (typeof senderName === 'string') return senderName;
    if (typeof senderName === 'object' && senderName.name) return senderName.name;
    return 'Unknown User';
  }, []);

  // Error state component
  const renderErrorState = () => (
    <Box 
      sx={{ 
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        p: 3,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 3,
          px: 4,
          py: 5,
          maxWidth: 400,
        }}
      >
        <RefreshIcon 
          sx={{ 
            color: theme.palette.error.main, 
            fontSize: 36,
          }} 
        />
        
        <Box sx={{ textAlign: 'center' }}>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 600, 
              mb: 1.5,
              color: theme.palette.error.main,
            }}
          >
            Failed to Load Messages
          </Typography>
          <Typography 
            variant="body2" 
            color="text.secondary"
            sx={{
              mb: 3,
              lineHeight: 1.6,
              maxWidth: 320,
            }}
          >
            {error || 'Unable to load chat messages. This could be due to a network issue or server problem.'}
          </Typography>
          
          <IconButton
            size="large"
            onClick={() => loadMessages()}
            sx={{ 
              color: theme.palette.primary.main,
            }}
          >
            <RefreshIcon sx={{ fontSize: 28 }} />
          </IconButton>
        </Box>
      </Box>
    </Box>
  );

  if (loading) {
    return (
      <Paper 
        elevation={0} 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <Box sx={{ 
          p: 2, 
          borderBottom: `1px solid ${theme.palette.divider}`,
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between' 
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={24} />
            <Typography variant="body1">Loading conversation...</Typography>
          </Box>
        </Box>
      </Paper>
    );
  }

  // Show error state if there is an error and not currently loading
  if (error && !loading) {
    return (
      <Paper 
        elevation={0} 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {renderErrorState()}
      </Paper>
    );
  }

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box sx={{ 
        p: 2, 
        borderBottom: `1px solid ${theme.palette.divider}`,
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between' 
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar 
            sx={{ 
              bgcolor: theme.palette.primary.main,
            }}
          >
            <PersonIcon />
          </Avatar>
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              {user.role === 'super-admin' ? tenantName : (contactName || 'Super Administrator')}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {user.role === 'super-admin' ? 'Tenant Admin Chat' : 'Support Chat'} • Online
            </Typography>
          </Box>
        </Box>
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </Box>

      {/* Messages */}
      <Box 
        ref={messagesContainerRef}
        sx={{ 
          flex: 1, 
          overflow: 'auto', 
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          bgcolor: theme.palette.background.default,
          '&::-webkit-scrollbar': {
            width: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(0, 0, 0, 0.03)',
          },
          '&::-webkit-scrollbar-thumb': {
            background: alpha(theme.palette.primary.main, 0.2),
            borderRadius: '3px',
            '&:hover': {
              background: alpha(theme.palette.primary.main, 0.3),
            },
          },
        }}
        onScroll={(e) => {
          const { scrollTop } = e.target;
          if (scrollTop === 0 && hasMore) {
            loadMoreMessages();
          }
        }}
      >
        {loadingMore && (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress size={24} />
          </Box>
        )}

        <List sx={{ flex: 1, py: 0 }}>
          {messages.map((message, index) => {
            const isOwn = message.senderId === user._id || message.senderId === user.id;
            const showTime = index === 0 || 
              new Date(messages[index - 1].createdAt).getTime() < new Date(message.createdAt).getTime() - 300000; // 5 minutes

            return (
              <ListItem
                key={message._id}
                sx={{
                  flexDirection: 'column',
                  alignItems: isOwn ? 'flex-end' : 'flex-start',
                  py: 1,
                  px: 0,
                  position: 'relative',
                }}
              >
                {showTime && (
                  <Box sx={{ 
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mb: 1, 
                    width: '100%',
                  }}>
                    <Typography 
                      variant="caption" 
                      color="text.secondary"
                      sx={{ fontSize: '0.75rem' }}
                    >
                      {formatMessageTime(message.createdAt)}
                    </Typography>
                  </Box>
                )}
                
                <Box
                  sx={{
                    maxWidth: '75%',
                    bgcolor: isOwn 
                      ? theme.palette.primary.main
                      : theme.palette.background.paper,
                    color: isOwn ? 'white' : 'text.primary',
                    p: 2,
                    borderRadius: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1,
                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                    border: isOwn 
                      ? 'none' 
                      : `1px solid ${theme.palette.divider}`,
                  }}
                >
                  {!isOwn && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Typography variant="caption" sx={{ 
                        fontWeight: 600, 
                        color: message.senderRole === 'super_admin' 
                          ? theme.palette.primary.main
                          : theme.palette.primary.main,
                      }}>
                        {renderSenderName(message.senderName)}
                      </Typography>
                      <Chip 
                        label={message.senderRole === 'super_admin' ? 'Super Admin' : 'Tenant Admin'}
                        size="small"
                        sx={{ 
                          height: 20, 
                          fontSize: '0.65rem',
                          bgcolor: alpha(theme.palette.primary.main, 0.1),
                          color: theme.palette.primary.main,
                          fontWeight: 500,
                        }}
                      />
                    </Box>
                  )}
                  
                  <Box sx={{ 
                    fontSize: '0.95rem', 
                    lineHeight: 1.5,
                  }}>
                    {renderMessageContent(message.message, message.messageType)}
                  </Box>

                  {/* Message actions */}
                  <Box sx={{ 
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1,
                  }}>
                    <IconButton 
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        setMoreMenuAnchorEl(e.currentTarget);
                      }}
                      sx={{ color: 'text.secondary' }}
                    >
                      <MoreVertIcon fontSize="small" />
                    </IconButton>
                    <Menu
                      anchorEl={moreMenuAnchorEl}
                      open={Boolean(moreMenuAnchorEl)}
                      onClose={() => setMoreMenuAnchorEl(null)}
                    >
                      <MenuItem 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleOpenReportSelector();
                          setMoreMenuAnchorEl(null);
                        }}
                      >
                        <AssignmentIcon fontSize="small" sx={{ mr: 1 }} />
                        Report Message
                      </MenuItem>
                    </Menu>
                  </Box>
                </Box>
              </ListItem>
            );
          })}
        </List>

        {/* Typing indicator */}
        {typingUsers.length > 0 && (
          <Box sx={{ 
            px: 2, 
            py: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            maxWidth: '250px',
          }}>
            <Typography 
              variant="caption" 
              color="text.secondary"
              sx={{ fontStyle: 'italic' }}
            >
              {typingUsers.join(', ')} {typingUsers.length === 1 ? 'is' : 'are'} typing...
            </Typography>
          </Box>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Error display */}
      {error && (
        <Alert 
          severity="error" 
          onClose={() => setError(null)} 
          sx={{ m: 2 }}
        >
          {error}
        </Alert>
      )}

      {/* Message input */}
      <Box sx={{ 
        p: 2, 
        borderTop: `1px solid ${theme.palette.divider}`,
        bgcolor: theme.palette.background.paper,
      }}>
        <TextField
          fullWidth
          placeholder="Type your message..."
          value={newMessage}
          onChange={(e) => {
            setNewMessage(e.target.value);
            handleTyping();
          }}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
          disabled={sending}
          multiline
          maxRows={4}
          variant="outlined"
          size="small"
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={handleSendMessage}
                  disabled={!newMessage.trim() || sending}
                  color="primary"
                >
                  {sending ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
        
        {/* Options Menu */}
        <Menu
          anchorEl={moreMenuAnchorEl}
          open={Boolean(moreMenuAnchorEl)}
          onClose={handleMoreMenuClose}
        >
          <MenuItem onClick={handleOpenReportSelector}>
            <AssignmentIcon sx={{ mr: 1 }} />
            <Typography variant="body2">Share Damage Report</Typography>
          </MenuItem>
        </Menu>
      </Box>
      
      {/* Report Selector Dialog */}
      <ReportSelectorDialog
        open={reportSelectorOpen}
        onClose={() => setReportSelectorOpen(false)}
        onSelectReport={handleSelectReport}
        tenantId={tenantId}
      />
    </Paper>
  );
};

// PropTypes for better development experience
ChatWindow.propTypes = {
  tenantId: PropTypes.string.isRequired,
  tenantName: PropTypes.string,
  contactName: PropTypes.string,
  roomType: PropTypes.string,
  roomId: PropTypes.string,
  onClose: PropTypes.func.isRequired
};

export default ChatWindow;
