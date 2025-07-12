import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import io from 'socket.io-client';
import { useAuth } from '../hooks/useAuth';

const SocketContext = createContext();

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [chatNotifications, setChatNotifications] = useState([]);
  const [unreadCounts, setUnreadCounts] = useState({});
  const { user } = useAuth();
  const token = localStorage.getItem('admin_auth_token');
  const socketRef = useRef(null);

  const connect = useCallback(() => {
    if (!token) return;
    
    // Disconnect existing socket first
    if (socketRef.current) {
      console.log('Disconnecting existing socket before creating new one');
      socketRef.current.disconnect();
      socketRef.current = null;
      setSocket(null);
      setIsConnected(false);
    }
    
    console.log('Connecting to socket server...');
    const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5030';
    console.log('Backend URL:', backendUrl);

    const newSocket = io(backendUrl, {
      withCredentials: true,
      transports: ['websocket', 'polling'],
      auth: {
        token, // Send token with initial connection
        userType: 'admin'
      }
    });

    newSocket.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
      // Authenticate with the server
      newSocket.emit('authenticate', token);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setIsConnected(false);
    });

    newSocket.on('authenticated', (userInfo) => {
      console.log('Authenticated:', userInfo);
      
      // Join appropriate rooms based on user role
      if (userInfo.role === 'super-admin') {
        newSocket.emit('join_room', { room: 'super_admin' });
        console.log('Joined super_admin room');
      }
      
      // Join admin-specific room
      newSocket.emit('join_room', { room: `admin_${userInfo.id}` });
      console.log(`Joined admin_${userInfo.id} room`);
      
      // Join tenant room if applicable
      if (userInfo.tenantId) {
        newSocket.emit('join_room', { room: `tenant_${userInfo.tenantId}` });
        newSocket.emit('join_room', { room: `chat_${userInfo.tenantId}` });
        console.log(`Joined tenant rooms for ${userInfo.tenantId}`);
      }
    });

    newSocket.on('auth_error', (error) => {
      console.error('Authentication error:', error);
    });

    newSocket.on('new_message', (message) => {
      // Log all incoming messages for debugging
      console.log('SocketContext: Received new_message event:', message);
      
      // Enhanced logging of message properties
      console.log('Message properties:', {
        id: message._id,
        sender: message.senderId,
        senderModel: message.senderModel,
        senderName: message.senderName,
        chatId: message.chatId,
        tenantId: message.tenantId,
        fromFieldWorker: message.fromFieldWorker,
        adminId: message.adminId
      });
      
      // Handle ALL messages, not just from field workers
      let tenantId = message.tenantId;
      
      // Try multiple ways to extract tenantId
      if (!tenantId && message.chatId) {
        if (typeof message.chatId === 'string') {
          if (message.chatId.includes('_')) {
            tenantId = message.chatId.split('_')[1];
          } else if (message.chatId.match(/^[a-f\d]{24}$/i)) {
            // This is a MongoDB ObjectId (field worker chat room)
            tenantId = message.chatId;
          }
        }
      }
      
      // Only create notifications for messages from non-admins and avoid duplicates
      if (tenantId && message.senderModel !== 'Admin' && message._id) {
        console.log('SocketContext: Creating notification for message');
        
        // Check if we already have a notification for this message
        setChatNotifications(prev => {
          const existingNotification = prev.find(notif => 
            notif.messageId === message._id || 
            (notif.chatId === message.chatId && notif.message === message.message && 
             Math.abs(new Date(notif.timestamp) - new Date()) < 5000) // 5 second window
          );
          
          if (existingNotification) {
            console.log('SocketContext: Duplicate notification detected, skipping');
            return prev;
          }
          
          return [{
            id: Date.now(),
            messageId: message._id,
            type: 'new_message',
            tenantId,
            senderName: message.senderName,
            message: message.message,
            timestamp: new Date(),
            read: false,
            fromFieldWorker: message.fromFieldWorker || message.senderModel === 'FieldWorker',
            chatId: message.chatId
          }, ...prev].slice(0, 10);
        });
        
        setUnreadCounts(prev => ({
          ...prev,
          [tenantId]: (prev[tenantId] || 0) + 1
        }));
      }
    });

    // Enhanced chat notification handling - DISABLED to prevent duplicates
    // The new_message handler above should be sufficient for all notifications
    /*
    newSocket.on('chat_notification', (notification) => {
      console.log('SocketContext: Received chat_notification:', notification);
      
      // Check for duplicate notifications
      setChatNotifications(prev => {
        const existingNotification = prev.find(notif => 
          (notif.messageId && notification.messageId && notif.messageId === notification.messageId) ||
          (notif.chatId === notification.chatId && notif.message === notification.message && 
           Math.abs(new Date(notif.timestamp) - new Date()) < 5000) // 5 second window
        );
        
        if (existingNotification) {
          console.log('SocketContext: Duplicate chat notification detected, skipping');
          return prev;
        }
        
        return [{
          id: Date.now(),
          ...notification,
          read: false
        }, ...prev].slice(0, 10);
      });
      
      // Update unread count for the specific tenant
      if (notification.tenantId) {
        setUnreadCounts(prev => ({
          ...prev,
          [notification.tenantId]: (prev[notification.tenantId] || 0) + 1
        }));
      }
      
      // Show browser notification if permission is granted
      if (Notification.permission === 'granted') {
        new Notification(`New message from ${notification.senderName}`, {
          body: notification.message,
          icon: '/favicon.ico',
          tag: `chat-${notification.tenantId}`
        });
      }
    });
    */
    
    // Global message handler for broadcast messages - DISABLED to prevent duplicates
    // The new_message handler above should be sufficient for all message handling
    /*
    newSocket.on('global_message', (data) => {
      console.log('SocketContext: Received global_message:', data);
      
      // Check if this is targeted at current admin
      if (data.type === 'admin_message' && user && 
          (data.targetAdmin === user._id || data.targetAdmin === user.id)) {
        
        console.log('Global message is targeted at current admin, forwarding to handlers');
        
        // Re-emit as chat notification
        if (data.notification) {
          setChatNotifications(prev => [data.notification, ...prev].slice(0, 10));
          
          // Update unread count if tenantId is available
          if (data.notification.tenantId) {
            setUnreadCounts(prev => ({
              ...prev,
              [data.notification.tenantId]: (prev[data.notification.tenantId] || 0) + 1
            }));
          }
        }
        
        // Re-emit message for chat components
        if (data.message) {
          newSocket.emit('new_message', data.message);
        }
      }
    });
    */

    newSocket.on('user_typing', (data) => {
      // This will be handled by individual chat components
    });

    newSocket.on('messages_read', (data) => {
      // Reset unread count when messages are read
      if (data.tenantId) {
        setUnreadCounts(prev => ({
          ...prev,
          [data.tenantId]: 0
        }));
      }
    });

    newSocket.on('error', (error) => {
      console.error('Socket error:', error);
    });

    setSocket(newSocket);
    socketRef.current = newSocket;
  }, [token]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setSocket(null);
      setIsConnected(false);
    }
  }, []);

  const joinChat = useCallback((tenantId) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('join_chat', tenantId);
    }
  }, [isConnected]);

  const sendMessage = useCallback((tenantId, message, messageType = 'text', attachmentUrl = null) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('send_message', {
        tenantId,
        message,
        messageType,
        attachmentUrl
      });
    }
  }, [isConnected]);

  const markAsRead = useCallback((tenantId) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('mark_read', tenantId);
      setUnreadCounts(prev => ({
        ...prev,
        [tenantId]: 0
      }));
    }
  }, [isConnected]);

  const startTyping = useCallback((tenantId) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('typing', { tenantId, isTyping: true });
    }
  }, [isConnected]);

  const stopTyping = useCallback((tenantId) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('typing', { tenantId, isTyping: false });
    }
  }, [isConnected]);

  const clearNotification = useCallback((tenantId) => {
    setChatNotifications(prev => prev.filter(notif => notif.tenantId !== tenantId));
  }, []);

  const clearAllNotifications = useCallback(() => {
    setChatNotifications([]);
  }, []);

  // Connect when user is authenticated
  useEffect(() => {
    if (user && token) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      // Cleanup on unmount
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [user, token, connect, disconnect]);

  const value = {
    socket,
    isConnected,
    chatNotifications,
    unreadCounts,
    joinChat,
    sendMessage,
    markAsRead,
    startTyping,
    stopTyping,
    clearNotification,
    clearAllNotifications
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};
