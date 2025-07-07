import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useTheme } from '@mui/material/styles';
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
  Skeleton,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Send as SendIcon,
  Person as PersonIcon,
  AdminPanelSettings as AdminIcon,
  Close as CloseIcon,
  Assignment as AssignmentIcon,
  MoreVert as MoreVertIcon
} from '@mui/icons-material';
import { format, isToday, isYesterday } from 'date-fns';
import { useSocket } from '../../context/SocketContext';
import { chatService } from '../../services/chatService';
import { useAuth } from '../../hooks/useAuth';
import PropTypes from 'prop-types';
import ReportSelectorDialog from './ReportSelectorDialog';
import ReportMessage from './ReportMessage';

const ChatWindow = ({ tenantId, tenantName, contactName, onClose }) => {
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

      const response = await chatService.getChatMessages(tenantId, pageNum);
      
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
      setError('Failed to load messages');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [tenantId, scrollToBottom]);

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
      console.log('Received new_message event:', message);
      
      // We need to check multiple possible chatId formats
      const isTenantChat = 
        message.chatId === `tenant_${tenantId}` || 
        message.chatId.toString() === tenantId ||
        message.chatId.includes(tenantId);
      
      // For debugging
      if (!isTenantChat) {
        console.log(`Message chatId ${message.chatId} doesn't match expected tenantId ${tenantId}`);
        // Also log if this is possibly from a field worker we're interested in
        if (message.senderModel === 'FieldWorker') {
          console.log('This is a field worker message - checking if it belongs in this chat');
          // Additional logging to help debug
          console.log('Message full details:', message);
        }
      }
      
      if (isTenantChat || (user?.tenant?._id === tenantId || user?.tenant === tenantId)) {
        console.log('Adding message to chat:', message);
        setMessages(prev => [...prev, message]);
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
  }, [socket, tenantId, user, markAsRead, scrollToBottom]);

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
    console.log('Rendering message:', message, 'Type:', messageType); // Debug log
    
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



  if (loading) {
    return (
      <Paper 
        elevation={0} 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          borderRadius: 3,
          overflow: 'hidden',
          border: '1px solid rgba(255, 255, 255, 0.8)',
          bgcolor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
          position: 'relative',
        }}
      >
        {/* Header Skeleton */}
        <Box sx={{ 
          p: 3, 
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
          borderBottom: '1px solid rgba(139, 92, 246, 0.15)',
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between' 
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Skeleton variant="circular" width={48} height={48} animation="wave" />
            <Box>
              <Skeleton variant="rectangular" width={180} height={24} sx={{ borderRadius: 1, mb: 1 }} animation="wave" />
              <Skeleton variant="rectangular" width={120} height={16} sx={{ borderRadius: 1 }} animation="wave" />
            </Box>
          </Box>
          <Skeleton variant="circular" width={36} height={36} animation="wave" />
        </Box>
        
        {/* Messages Area Skeleton */}
        <Box sx={{ 
          flex: 1, 
          p: 3,
          display: 'flex',
          flexDirection: 'column',
          background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(249, 250, 251, 0.95) 100%)',
        }}>
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center',
              height: '100%',
              flexDirection: 'column',
              gap: 3,
            }}
          >
            <CircularProgress 
              size={60} 
              thickness={4}
              sx={{ 
                color: '#8b5cf6',
                opacity: 0.5,
              }} 
            />
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#6b7280', 
                fontWeight: 500,
                opacity: 0.7,
              }}
            >
              Loading conversation...
            </Typography>
          </Box>
        </Box>
        
        {/* Input Skeleton */}
        <Box sx={{ 
          p: 3, 
          borderTop: '1px solid rgba(226, 232, 240, 0.8)',
          background: 'rgba(249, 250, 251, 0.8)',
        }}>
          <Skeleton 
            variant="rectangular" 
            height={54} 
            sx={{ 
              borderRadius: '24px',
              animation: "pulse 1.5s infinite ease-in-out",
              '@keyframes pulse': {
                '0%': { opacity: 0.6 },
                '50%': { opacity: 0.8 },
                '100%': { opacity: 0.6 },
              }
            }} 
          />
        </Box>
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
        borderRadius: 3,
        overflow: 'hidden',
        border: '1px solid rgba(255, 255, 255, 0.8)',
        bgcolor: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
      }}
    >
      {/* Enhanced Header */}
      <Box sx={{ 
        p: 3, 
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
        borderBottom: '1px solid rgba(139, 92, 246, 0.15)',
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between' 
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box sx={{
            position: 'relative',
          }}>
            <Avatar 
              sx={{ 
                bgcolor: 'primary.main',
                background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                width: 48,
                height: 48,
                boxShadow: '0 5px 15px rgba(139, 92, 246, 0.25)',
              }}
            >
              <PersonIcon />
            </Avatar>
            <Box sx={{
              position: 'absolute',
              bottom: -2,
              right: -2,
              width: 14,
              height: 14,
              borderRadius: '50%',
              bgcolor: theme.palette.success.main,
              border: '2px solid white',
              boxShadow: '0 2px 5px rgba(16, 185, 129, 0.3)',
            }} />
          </Box>
          <Box>
            <Typography variant="h6" sx={{ 
              fontWeight: 700,                color: theme.palette.text.primary,
              lineHeight: 1.3, 
            }}>
              {user.role === 'super-admin' ? tenantName : (contactName || 'Super Administrator')}
            </Typography>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 1 
            }}>
              <Box sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: theme.palette.success.main,
              }} />
              <Typography variant="caption" sx={{ 
                color: '#6b7280', 
                fontSize: '0.8rem',
                fontWeight: 500,
              }}>
                {user.role === 'super-admin' ? 'Tenant Admin Chat' : 'Support Chat'} • Online
              </Typography>
            </Box>
          </Box>
        </Box>
        <IconButton 
          onClick={onClose}
          sx={{ 
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
            '&:hover': { 
              bgcolor: 'rgba(255, 255, 255, 0.95)',
              transform: 'scale(1.05)'
            },
            transition: 'all 0.2s ease-in-out'
          }}
        >
          <CloseIcon sx={{ color: theme.palette.text.secondary }} />
        </IconButton>
      </Box>

      {/* Messages */}
      <Box 
        ref={messagesContainerRef}
        sx={{ 
          flex: 1, 
          overflow: 'auto', 
          p: 3,
          display: 'flex',
          flexDirection: 'column',
          background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(249, 250, 251, 0.95) 100%)',
          '&::-webkit-scrollbar': {
            width: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(0, 0, 0, 0.03)',
            borderRadius: '3px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%)',
            borderRadius: '3px',
            '&:hover': {
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.5) 0%, rgba(139, 92, 246, 0.5) 100%)',
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
            <CircularProgress size={28} sx={{ 
              color: '#8b5cf6',
              opacity: 0.7,
            }} />
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
                    mb: 2, 
                    mt: index > 0 ? 2 : 0,
                    width: '100%',
                    position: 'relative',
                  }}>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        color: '#6b7280', 
                        mb: 0,
                        fontSize: '0.75rem',
                        fontWeight: 500,
                        px: 2,
                        py: 0.5,
                        borderRadius: '12px',
                        bgcolor: 'rgba(139, 92, 246, 0.08)',
                      }}
                    >
                      {formatMessageTime(message.createdAt)}
                    </Typography>
                  </Box>
                )}
                
                <Box
                  sx={{
                    maxWidth: '75%',
                    bgcolor: isOwn 
                      ? 'primary.main'
                      : 'rgba(255, 255, 255, 0.9)',
                    color: isOwn ? 'white' : 'text.primary',
                    p: 2,
                    borderRadius: isOwn ? '20px 4px 20px 20px' : '4px 20px 20px 20px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1,
                    boxShadow: isOwn 
                      ? '0 5px 15px rgba(139, 92, 246, 0.25)'
                      : '0 3px 10px rgba(0, 0, 0, 0.05)',
                    border: isOwn ? 'none' : '1px solid rgba(226, 232, 240, 0.8)',
                    background: isOwn 
                      ? 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)' 
                      : 'rgba(255, 255, 255, 0.9)',
                    backdropFilter: 'blur(10px)',
                    position: 'relative',
                  }}
                >
                  {!isOwn && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.75 }}>
                      {message.senderRole === 'super_admin' ? (
                        <AdminIcon sx={{ fontSize: 16, color: '#8b5cf6' }} />
                      ) : (
                        <PersonIcon sx={{ fontSize: 16, color: '#3b82f6' }} />
                      )}
                      <Typography variant="caption" sx={{ 
                        fontWeight: 700, 
                        color: message.senderRole === 'super_admin' ? '#8b5cf6' : '#3b82f6',
                        fontSize: '0.8rem',
                      }}>
                        {renderSenderName(message.senderName)}
                      </Typography>
                      <Chip 
                        label={message.senderRole === 'super_admin' ? 'Super Admin' : 'Tenant Admin'}
                        size="small"
                        sx={{ 
                          height: 20, 
                          fontSize: '0.65rem',
                          bgcolor: message.senderRole === 'super_admin' ? 'rgba(139, 92, 246, 0.1)' : 'rgba(59, 130, 246, 0.1)',
                          color: message.senderRole === 'super_admin' ? '#8b5cf6' : '#3b82f6',
                          border: `1px solid ${message.senderRole === 'super_admin' ? 'rgba(139, 92, 246, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
                          fontWeight: 600,
                          '& .MuiChip-label': {
                            px: 1,
                          }
                        }}
                      />
                    </Box>
                  )}
                  
                  <Box sx={{ 
                    fontSize: '0.95rem', 
                    lineHeight: 1.5,
                    fontWeight: isOwn ? 400 : 400, 
                  }}>
                    {renderMessageContent(message.message, message.messageType)}
                  </Box>

                  {/* Message actions for reporting */}
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
                      sx={{ 
                        color: 'rgba(0, 0, 0, 0.54)',
                        '&:hover': {
                          color: '#8b5cf6',
                        },
                      }}
                    >
                      <MoreVertIcon fontSize="small" />
                    </IconButton>
                    <Menu
                      anchorEl={moreMenuAnchorEl}
                      open={Boolean(moreMenuAnchorEl)}
                      onClose={() => setMoreMenuAnchorEl(null)}
                      PaperProps={{
                        elevation: 0,
                        sx: {
                          bgcolor: 'white',
                          borderRadius: 2,
                          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                          mt: 1,
                        },
                      }}
                    >
                      <MenuItem 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleOpenReportSelector();
                          setMoreMenuAnchorEl(null);
                        }}
                        sx={{ 
                          color: 'text.primary',
                          '&:hover': {
                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                          },
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

        {/* Typing indicator with modern styling */}
        {typingUsers.length > 0 && (
          <Box sx={{ 
            px: 3, 
            py: 2,
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            margin: '8px 16px',
            border: '1px solid rgba(226, 232, 240, 0.8)',
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            maxWidth: '250px',
          }}>
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}>
              <Box sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: '#8b5cf6',
                animation: 'pulse1 1s infinite',
                '@keyframes pulse1': {
                  '0%, 100%': { opacity: 0.3 },
                  '50%': { opacity: 1 }
                }
              }} />
              <Box sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: '#8b5cf6',
                animation: 'pulse2 1s infinite',
                animationDelay: '0.2s',
                '@keyframes pulse2': {
                  '0%, 100%': { opacity: 0.3 },
                  '50%': { opacity: 1 }
                }
              }} />
              <Box sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: '#8b5cf6',
                animation: 'pulse3 1s infinite',
                animationDelay: '0.4s',
                '@keyframes pulse3': {
                  '0%, 100%': { opacity: 0.3 },
                  '50%': { opacity: 1 }
                }
              }} />
            </Box>
            <Typography 
              variant="caption" 
              sx={{ 
                color: '#6d28d9', 
                fontStyle: 'italic',
                fontSize: '0.85rem',
                fontWeight: 500,
              }}
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
          sx={{ 
            m: 2,
            borderRadius: 3,
            border: '1px solid rgba(239, 68, 68, 0.2)',
            boxShadow: '0 4px 12px rgba(239, 68, 68, 0.1)',
            '& .MuiAlert-message': {
              fontSize: '0.9rem',
              fontWeight: 500,
            },
            '& .MuiAlert-icon': {
              color: '#ef4444'
            }
          }}
        >
          {error}
        </Alert>
      )}

      {/* Enhanced Message input */}
      <Box sx={{ 
        p: 3, 
        borderTop: '1px solid rgba(226, 232, 240, 0.8)',
        background: 'rgba(249, 250, 251, 0.8)',
        backdropFilter: 'blur(10px)',
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
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: '24px',
              bgcolor: 'white',
              boxShadow: '0 2px 12px rgba(0, 0, 0, 0.04)',
              border: '1px solid rgba(226, 232, 240, 0.8)',
              '& fieldset': {
                borderColor: 'transparent',
              },
              '&:hover fieldset': {
                borderColor: '#8b5cf6',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#8b5cf6',
                borderWidth: '1px',
              },
              '&:hover': {
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.06)',
              },
            },
            '& .MuiOutlinedInput-input': {
              fontSize: '0.95rem',
              padding: '14px 20px',
              '&::placeholder': {
                opacity: 0.6,
                fontStyle: 'italic',
              },
            },
          }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <IconButton
                  onClick={(e) => handleMoreMenuOpen(e)}
                  size="small"
                  sx={{
                    color: '#6b7280',
                    '&:hover': { color: '#8b5cf6' },
                  }}
                >
                  <MoreVertIcon />
                </IconButton>
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={handleSendMessage}
                  disabled={!newMessage.trim() || sending}
                  size="large"
                  sx={{
                    background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                    color: 'white',
                    width: 44,
                    height: 44,
                    mr: 0.5,
                    boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)',
                    transition: 'all 0.2s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 15px rgba(139, 92, 246, 0.4)',
                      background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                    },
                    '&.Mui-disabled': {
                      bgcolor: 'rgba(203, 213, 225, 0.8)',
                      color: 'rgba(148, 163, 184, 0.8)',
                    },
                  }}
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
          PaperProps={{
            elevation: 3,
            sx: {
              borderRadius: 2,
              minWidth: 180,
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
            }
          }}
        >
          <MenuItem 
            onClick={handleOpenReportSelector}
            sx={{ 
              py: 1.5, 
              px: 2,
              '&:hover': { bgcolor: 'rgba(139, 92, 246, 0.08)' },
            }}
          >
            <AssignmentIcon 
              sx={{ 
                color: '#3b82f6',
                mr: 1,
                fontSize: '1.2rem'
              }} 
            />
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
  onClose: PropTypes.func.isRequired
};

export default ChatWindow;
