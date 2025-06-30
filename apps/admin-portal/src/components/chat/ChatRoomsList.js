import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Badge,
  Divider,
  TextField,
  InputAdornment,
  Skeleton,
  IconButton
} from '@mui/material';
import {
  Search as SearchIcon,
  Chat as ChatIcon,
  AdminPanelSettings as AdminIcon,
  AccessTime as TimeIcon,
  Refresh as RefreshIcon,
  Circle as OnlineIcon
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { useSocket } from '../../context/SocketContext';
import { chatService } from '../../services/chatService';
import { useAuth } from '../../hooks/useAuth';
import PropTypes from 'prop-types';

const ChatRoomsList = ({ onSelectRoom, selectedRoomId }) => {
  const [chatRooms, setChatRooms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  const { chatNotifications, unreadCounts, clearNotification } = useSocket();
  const { user } = useAuth();

  // Load chat rooms with enhanced error handling
  const loadChatRooms = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);
      
      console.log('User data for chat:', user);
      
      // Test auth first
      try {
        await chatService.testAuth();
      } catch (authError) {
        console.error('Auth test failed:', authError);
        if (authError.response?.status === 401) {
          setError('Authentication failed. Please log in again.');
          return;
        } else {
          setError('Connection failed. Please check your internet connection.');
          return;
        }
      }
      
      let rooms = [];
      
      if (user.role === 'super-admin') {
        rooms = await chatService.getAllChatRooms();
      } else {
        rooms = await chatService.getTenantChatRooms();
      }
      
      setChatRooms(rooms);
    } catch (err) {
      console.error('Error loading chat rooms:', err);
      setError('Failed to load chat rooms. Please try again.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [user]);

  useEffect(() => {
    if (user) {
      loadChatRooms();
    }
  }, [loadChatRooms, user]);

  // Enhanced filter with better search logic
  const filteredRooms = useMemo(() => {
    if (!searchTerm.trim()) return chatRooms;
    
    return chatRooms.filter(room => {
      const searchLower = searchTerm.toLowerCase();
      const lastMessageText = typeof room.lastMessage === 'object' 
        ? (room.lastMessage?.message || '') 
        : (room.lastMessage || '');
        
      if (user.role === 'super-admin') {
        return (
          room.tenantName?.toLowerCase().includes(searchLower) ||
          lastMessageText.toLowerCase().includes(searchLower)
        );
      } else {
        return (
          (room.contactName || 'Support Team').toLowerCase().includes(searchLower) ||
          lastMessageText.toLowerCase().includes(searchLower)
        );
      }
    });
  }, [chatRooms, searchTerm, user.role]);

  // Handle room selection with better UX
  const handleRoomSelect = useCallback((room) => {
    onSelectRoom(room);
    clearNotification(room.tenantId);
  }, [onSelectRoom, clearNotification]);

  // Get unread count for a room
  const getUnreadCount = useCallback((tenantId) => {
    return unreadCounts[tenantId] || 0;
  }, [unreadCounts]);

  // Format last message time with better handling
  const formatLastMessageTime = useCallback((timestamp) => {
    if (!timestamp) return '';
    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    } catch {
      return '';
    }
  }, []);

  // Enhanced loading state with modern skeletons
  const renderLoadingState = () => (
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
      <Box sx={{ 
        p: 3, 
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
        borderBottom: '1px solid',
        borderColor: 'rgba(139, 92, 246, 0.2)'
      }}>
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
        }}>
          <Skeleton variant="rectangular" height={32} width={160} sx={{ borderRadius: 1.5 }} />
          <Skeleton variant="circular" width={32} height={32} />
        </Box>
        <Skeleton variant="rectangular" height={40} sx={{ borderRadius: 2.5, mb: 1 }} />
      </Box>

      <Box sx={{ flex: 1, p: 1, overflow: 'hidden' }}>
        {[...Array(5)].map((_, index) => (
          <React.Fragment key={index}>
            <Box sx={{ 
              display: 'flex', 
              py: 2, 
              px: 3,
              borderRadius: 2,
              mx: 1,
              my: 0.5,
              backgroundColor: index === 0 ? 'rgba(139, 92, 246, 0.08)' : 'transparent',
            }}>
              <Skeleton variant="circular" width={48} height={48} sx={{ mr: 2 }} />
              <Box sx={{ flex: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Skeleton variant="rectangular" width={120} height={22} sx={{ borderRadius: 1 }} />
                  <Skeleton variant="rectangular" width={60} height={22} sx={{ borderRadius: 3 }} />
                </Box>
                <Skeleton variant="rectangular" width="90%" height={18} sx={{ borderRadius: 1 }} />
              </Box>
            </Box>
            {index < 4 && (
              <Divider sx={{ mx: 3, opacity: 0.3 }} />
            )}
          </React.Fragment>
        ))}
      </Box>
    </Paper>
  );

  // Enhanced error state with modern design
  const renderErrorState = () => (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        borderRadius: 3,
        border: '1px solid rgba(255, 255, 255, 0.8)',
        bgcolor: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 3,
          px: 3,
          py: 4,
          borderRadius: 3,
          background: 'rgba(239, 68, 68, 0.05)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          boxShadow: '0 4px 15px rgba(239, 68, 68, 0.1)',
          maxWidth: 320,
        }}
      >
        <Box
          sx={{
            width: 64,
            height: 64,
            borderRadius: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            boxShadow: '0 2px 10px rgba(239, 68, 68, 0.1)',
          }}
        >
          <RefreshIcon 
            sx={{ 
              color: '#ef4444', 
              fontSize: 32,
            }} 
          />
        </Box>
        
        <Box sx={{ textAlign: 'center' }}>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 600, 
              mb: 1,
              color: '#dc2626',
            }}
          >
            Connection Error
          </Typography>
          <Typography 
            variant="body2" 
            color="text.secondary"
            sx={{
              mb: 3,
              lineHeight: 1.6,
            }}
          >
            {error}
          </Typography>
          <IconButton
            size="large"
            onClick={() => loadChatRooms()}
            sx={{ 
              bgcolor: 'rgba(239, 68, 68, 0.1)',
              color: '#dc2626',
              width: 54,
              height: 54,
              border: '1px solid rgba(239, 68, 68, 0.2)',
              '&:hover': { 
                bgcolor: 'rgba(239, 68, 68, 0.15)',
                transform: 'scale(1.05)',
                boxShadow: '0 4px 12px rgba(239, 68, 68, 0.15)',
              },
              transition: 'all 0.2s ease-in-out'
            }}
          >
            <RefreshIcon sx={{ fontSize: 24 }} />
          </IconButton>
        </Box>
      </Box>
    </Paper>
  );

  if (loading) return renderLoadingState();
  if (error) return renderErrorState();

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
        borderBottom: '1px solid',
        borderColor: 'rgba(139, 92, 246, 0.2)'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            <ChatIcon sx={{ color: '#6d28d9' }} />
            {user?.role === 'super-admin' ? 'All Conversations' : 'Support Chat'}
          </Typography>
          
          <IconButton 
            size="small" 
            onClick={() => loadChatRooms(true)}
            disabled={refreshing}
            sx={{ 
              bgcolor: 'rgba(255, 255, 255, 0.8)',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
              '&:hover': { 
                bgcolor: 'rgba(255, 255, 255, 0.95)',
                transform: 'scale(1.05)'
              },
              transition: 'all 0.2s ease-in-out'
            }}
            aria-label="Refresh conversations"
          >
            <RefreshIcon sx={{ 
              fontSize: 18,
              color: '#6d28d9',
              animation: refreshing ? 'spin 1s linear infinite' : 'none',
              '@keyframes spin': {
                '0%': { transform: 'rotate(0deg)' },
                '100%': { transform: 'rotate(360deg)' }
              }
            }} />
          </IconButton>
        </Box>
        
        {/* Enhanced Search Field */}
        <TextField
          fullWidth
          size="small"
          placeholder="Search conversations..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon sx={{ color: '#8b5cf6', fontSize: 20 }} />
              </InputAdornment>
            ),
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 2.5,
              bgcolor: 'rgba(255, 255, 255, 0.8)',
              boxShadow: '0 2px 10px rgba(0, 0, 0, 0.03)',
              border: '1px solid rgba(139, 92, 246, 0.2)',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
              },
              '&:hover fieldset': {
                borderColor: '#8b5cf6',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#8b5cf6',
                borderWidth: 2
              },
              '& fieldset': {
                borderColor: 'transparent',
              }
            },
            '& input::placeholder': {
              opacity: 0.7,
              fontStyle: 'italic',
              fontSize: '0.9rem',
            },
          }}
        />
        
        {searchTerm && (
          <Typography variant="caption" sx={{ 
            mt: 1, 
            color: '#8b5cf6', 
            display: 'block',
            fontWeight: 500 
          }}>
            {filteredRooms.length} result{filteredRooms.length !== 1 ? 's' : ''} found
          </Typography>
        )}
      </Box>

      {/* Enhanced Chat Rooms List */}
      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        {filteredRooms.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{ height: '100%' }}
          >
            <Box sx={{ 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center', 
              justifyContent: 'center', 
              height: '100%',
              p: 4,
              color: 'text.secondary'
            }}>
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Box sx={{
                  width: 80,
                  height: 80,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '20px',
                  background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
                  boxShadow: '0 5px 15px rgba(139, 92, 246, 0.1)',
                  mb: 3,
                  border: '1px solid rgba(139, 92, 246, 0.2)',
                }}>
                  <ChatIcon sx={{ fontSize: 36, color: '#8b5cf6', opacity: 0.8 }} />
                </Box>
              </motion.div>
              <Typography variant="h6" textAlign="center" sx={{ 
                fontWeight: 600, 
                mb: 1,
                color: '#6d28d9', 
              }}>
                {searchTerm ? 'No conversations found' : 'No conversations yet'}
              </Typography>
              <Typography variant="body2" textAlign="center" sx={{ 
                opacity: 0.8, 
                maxWidth: 250,
                lineHeight: 1.6,
                color: '#6b7280',
              }}>
                {searchTerm 
                  ? 'Try adjusting your search terms'
                  : 'Conversations will appear here when they start'
                }
              </Typography>
            </Box>
          </motion.div>
        ) : (
          <List sx={{ p: 0, height: '100%', overflow: 'auto' }}>
            <AnimatePresence>
              {filteredRooms.map((room, index) => {
                const unreadCount = getUnreadCount(room.tenantId);
                const isSelected = selectedRoomId === room.tenantId;
                
                return (
                  <motion.div
                    key={room.tenantId}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <ListItem disablePadding>
                      <ListItemButton
                        onClick={() => handleRoomSelect(room)}
                        selected={isSelected}
                        sx={{
                          px: 3,
                          py: 2,
                          transition: 'all 0.2s ease-in-out',
                          position: 'relative',
                          borderRadius: 2,
                          mx: 1,
                          mb: 0.5,
                          '&:hover': {
                            bgcolor: 'rgba(139, 92, 246, 0.08)',
                            transform: 'translateX(4px)',
                          },
                          '&.Mui-selected': {
                            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)',
                            color: '#6d28d9',
                            '&:hover': {
                              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)',
                            },
                            '& .MuiListItemText-primary': {
                              color: '#6d28d9',
                              fontWeight: 600
                            },
                            '& .MuiListItemText-secondary': {
                              color: 'rgba(109, 40, 217, 0.7)'
                            }
                          },
                          '&::before': {
                            content: '""',
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: 4,
                            borderRadius: '0 4px 4px 0',
                            bgcolor: isSelected ? '#8b5cf6' : 'transparent',
                            transition: 'all 0.2s ease-in-out'
                          }
                        }}
                      >
                        <ListItemAvatar>
                          <Badge
                            overlap="circular"
                            badgeContent={unreadCount}
                            color="error"
                            max={99}
                            sx={{
                              '& .MuiBadge-badge': {
                                fontSize: '0.75rem',
                                minWidth: 18,
                                height: 18,
                                fontWeight: 600,
                                background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)',
                                boxShadow: '0 3px 8px rgba(239, 68, 68, 0.3)',
                                animation: unreadCount > 0 ? 'pulse 2s infinite' : 'none',
                                '@keyframes pulse': {
                                  '0%': { transform: 'scale(1)' },
                                  '50%': { transform: 'scale(1.1)' },
                                  '100%': { transform: 'scale(1)' }
                                }
                              }
                            }}
                          >
                            <Avatar
                              sx={{
                                background: isSelected 
                                  ? 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)' 
                                  : 'linear-gradient(135deg, #4b5563 0%, #6b7280 100%)',
                                color: 'white',
                                width: 48,
                                height: 48,
                                boxShadow: isSelected 
                                  ? '0 5px 15px rgba(139, 92, 246, 0.3)' 
                                  : '0 4px 12px rgba(0, 0, 0, 0.1)',
                                transition: 'all 0.2s ease-in-out'
                              }}
                            >
                              {user?.role === 'super-admin' ? (
                                room.tenantName?.charAt(0)?.toUpperCase() || 'T'
                              ) : (
                                <AdminIcon />
                              )}
                            </Avatar>
                          </Badge>
                        </ListItemAvatar>
                        
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography
                                variant="subtitle1"
                                noWrap
                                sx={{
                                  fontWeight: unreadCount > 0 ? 700 : 500,
                                  flex: 1,
                                  fontSize: '0.95rem',
                                  letterSpacing: '0.02em',
                                }}
                              >
                                {user?.role === 'super-admin' 
                                  ? room.tenantName 
                                  : (room.contactName || 'Support Team')
                                }
                              </Typography>
                              {room.lastMessageTime && (
                                <Box sx={{ 
                                  display: 'flex', 
                                  alignItems: 'center', 
                                  gap: 0.5,
                                  bgcolor: 'rgba(139, 92, 246, 0.1)',
                                  borderRadius: '12px',
                                  py: 0.5,
                                  px: 1,
                                }}>
                                  <TimeIcon sx={{ fontSize: 12, color: '#8b5cf6' }} />
                                  <Typography
                                    variant="caption"
                                    sx={{
                                      color: '#6d28d9',
                                      fontWeight: 600,
                                      fontSize: '0.7rem'
                                    }}
                                  >
                                    {formatLastMessageTime(room.lastMessageTime)}
                                  </Typography>
                                </Box>
                              )}
                            </Box>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.75 }}>
                              <Typography
                                variant="body2"
                                noWrap
                                sx={{
                                  opacity: 0.8,
                                  fontWeight: unreadCount > 0 ? 500 : 400,
                                  flex: 1,
                                  fontSize: '0.85rem',
                                }}
                              >
                                {typeof room.lastMessage === 'object' 
                                  ? (room.lastMessage?.message || 'No messages yet')
                                  : (room.lastMessage || 'No messages yet')
                                }
                              </Typography>
                              {unreadCount > 0 && (
                                <OnlineIcon sx={{ fontSize: 10, color: '#10b981' }} />
                              )}
                            </Box>
                          }
                        />
                      </ListItemButton>
                    </ListItem>
                    {index < filteredRooms.length - 1 && (
                      <Divider 
                        sx={{ 
                          mx: 3,
                          opacity: 0.3
                        }} 
                      />
                    )}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </List>
        )}
      </Box>

      {/* Enhanced Notifications indicator */}
      {chatNotifications.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Box sx={{ 
            p: 2, 
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
            borderTop: '1px solid rgba(139, 92, 246, 0.2)',
            display: 'flex',
            alignItems: 'center',
            gap: 1
          }}>
            <Badge
              badgeContent={chatNotifications.length}
              color="primary"
              sx={{
                '& .MuiBadge-badge': {
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  minWidth: 18,
                  height: 18,
                  background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                  animation: 'pulse 2s infinite',
                  boxShadow: '0 2px 6px rgba(139, 92, 246, 0.3)',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.2)' },
                    '100%': { transform: 'scale(1)' }
                  }
                }
              }}
            >
              <Box sx={{
                width: 32,
                height: 32,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '10px',
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
                boxShadow: '0 2px 8px rgba(139, 92, 246, 0.15)',
              }}>
                <ChatIcon sx={{ color: '#8b5cf6', fontSize: 18 }} />
              </Box>
            </Badge>
            <Typography variant="body2" sx={{ 
              color: '#6d28d9',
              fontSize: '0.85rem',
              fontWeight: 600,
              flex: 1
            }}>
              {chatNotifications.length} new notification{chatNotifications.length > 1 ? 's' : ''}
            </Typography>
          </Box>
        </motion.div>
      )}
    </Paper>
  );
};

// PropTypes for better development experience
ChatRoomsList.propTypes = {
  onSelectRoom: PropTypes.func.isRequired,
  selectedRoomId: PropTypes.string
};

export default ChatRoomsList;
