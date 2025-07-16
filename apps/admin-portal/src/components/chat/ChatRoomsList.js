import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useTheme } from '@mui/material/styles';
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
import { useSocket } from '../../context/SocketContext';
import { chatService } from '../../services/chatService';
import { useAuth } from '../../hooks/useAuth';
import PropTypes from 'prop-types';

const ChatRoomsList = ({ onSelectRoom, selectedRoomId }) => {
  const theme = useTheme();
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
          room.fieldWorker?.name?.toLowerCase().includes(searchLower) ||
          lastMessageText.toLowerCase().includes(searchLower)
        );
      } else {
        return (
          (room.contactName || 'Support Team').toLowerCase().includes(searchLower) ||
          room.fieldWorker?.name?.toLowerCase().includes(searchLower) ||
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

  if (loading) {
    return (
      <Box sx={{ p: 2, height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">Loading conversations...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <RefreshIcon color="error" sx={{ fontSize: 40, mb: 2 }} />
        <Typography variant="body1" color="error" gutterBottom fontWeight={500}>
          {error}
        </Typography>
        <IconButton 
          color="primary"
          onClick={() => loadChatRooms()}
          disabled={refreshing}
        >
          <RefreshIcon />
        </IconButton>
      </Box>
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
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
        }}>
          <Typography variant="subtitle1" fontWeight={500}>
            Conversations
          </Typography>
          <IconButton 
            size="small" 
            onClick={() => loadChatRooms(true)}
            disabled={refreshing}
          >
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Box>
        <TextField
          placeholder="Search conversations..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          fullWidth
          size="small"
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
          }}
        />
      </Box>

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {filteredRooms.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <ChatIcon sx={{ fontSize: 40, color: 'text.disabled', mb: 1 }} />
            <Typography variant="body2" color="text.secondary">
              {searchTerm ? 'No matching conversations found' : 'No conversations yet'}
              </Typography>
            </Box>
        ) : (
          <List disablePadding>
              {filteredRooms.map((room, index) => {
              const isSelected = selectedRoomId === room.tenantId;
                const unreadCount = getUnreadCount(room.tenantId);
                
                return (
                <React.Fragment key={room.tenantId || index}>
                    <ListItem disablePadding>
                      <ListItemButton
                        onClick={() => handleRoomSelect(room)}
                        selected={isSelected}
                        sx={{
                        py: 1.5,
                        px: 2,
                        }}
                      >
                        <ListItemAvatar>
                          <Badge
                            overlap="circular"
                          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                          variant="dot"
                          color="success"
                          invisible={!room.isOnline}
                        >
                          <Avatar>
                            {user.role === 'super-admin' ? 
                              <AdminIcon /> : 
                              <ChatIcon />
                            }
                          </Avatar>
                        </Badge>
                      </ListItemAvatar>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography 
                              variant="body1" 
                              sx={{
                                fontWeight: unreadCount > 0 ? 600 : 400,
                                color: 'text.primary',
                              }}
                              noWrap
                            >
                              {user.role === 'super-admin' 
                                ? room.tenantName || 'Unnamed Tenant' 
                                : room.contactName || 'Support Team'
                              }
                            </Typography>
                            {unreadCount > 0 && (
                              <Badge 
                                badgeContent={unreadCount} 
                                color="primary"
                                max={99}
                                sx={{ ml: 1 }}
                              />
                            )}
                          </Box>
                        }
                        secondary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography
                              variant="body2" 
                              color="text.secondary" 
                                noWrap
                                sx={{
                                maxWidth: '70%',
                                fontWeight: unreadCount > 0 ? 500 : 400,
                                }}
                              >
                              {room.lastMessage ? 
                                (typeof room.lastMessage === 'object' ? 
                                  room.lastMessage.message : 
                                  room.lastMessage
                                ) : 
                                'No messages yet'
                                }
                              </Typography>
                              {room.lastMessageTime && (
                              <Typography 
                                variant="caption" 
                                color="text.secondary"
                                sx={{ 
                                  display: 'flex', 
                                  alignItems: 'center', 
                                  gap: 0.5,
                                  fontSize: '0.7rem',
                                    }}
                                  >
                                <TimeIcon sx={{ fontSize: '0.9rem' }} />
                                {formatLastMessageTime(room.lastMessageTime)}
                              </Typography>
                              )}
                            </Box>
                          }
                        />
                      </ListItemButton>
                    </ListItem>
                  {index < filteredRooms.length - 1 && <Divider component="li" />}
                </React.Fragment>
                );
              })}
          </List>
        )}
      </Box>
    </Paper>
  );
};

ChatRoomsList.propTypes = {
  onSelectRoom: PropTypes.func.isRequired,
  selectedRoomId: PropTypes.string
};

export default ChatRoomsList;
