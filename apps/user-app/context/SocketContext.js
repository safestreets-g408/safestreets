import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { io } from 'socket.io-client';
import { useAuth } from './AuthContext';
import { getAuthToken } from '../utils/auth';
import { API_BASE_URL } from '../config';

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
  const [connected, setConnected] = useState(false);
  const { isAuthenticated, fieldWorker } = useAuth();
  
  useEffect(() => {
    let socketInstance = null;
    
    const initializeSocket = async () => {
      if (!isAuthenticated || !fieldWorker) return;
      
      try {
        const token = await getAuthToken();
        if (!token) return;
        
        socketInstance = io(API_BASE_URL.replace('/api', ''), {
          auth: {
            token,
            userType: 'fieldworker'
          },
          transports: ['websocket', 'polling'],
          reconnection: true,
          reconnectionAttempts: 5,
          reconnectionDelay: 1000
        });
        
        socketInstance.on('connect', () => {
          console.log('Socket connected');
          setConnected(true);
        });
        
        socketInstance.on('disconnect', () => {
          console.log('Socket disconnected');
          setConnected(false);
        });
        
        socketInstance.on('error', (error) => {
          console.error('Socket error:', error);
          setConnected(false);
        });
        
        setSocket(socketInstance);
      } catch (error) {
        console.error('Error initializing socket:', error);
      }
    };
    
    initializeSocket();
    
    return () => {
      if (socketInstance) {
        socketInstance.disconnect();
        setSocket(null);
        setConnected(false);
      }
    };
  }, [isAuthenticated, fieldWorker]);
  
  const joinChat = useCallback((adminId, chatRoomId) => {
    if (socket && connected && fieldWorker?.tenant) {
      // Join tenant-wide chat room
      const tenantRoom = `tenant_${fieldWorker.tenant}`;
      socket.emit('join_room', { room: tenantRoom });
      console.log(`Joined tenant chat room: ${tenantRoom}`);
      
      // Join specific admin chat room if provided
      if (adminId) {
        const adminRoom = `admin_${adminId}`;
        socket.emit('join_room', { room: adminRoom });
        console.log(`Joined admin chat room: ${adminRoom}`);
        
        // Also join the direct chat room (if it exists)
        if (fieldWorker?._id) {
          const directRoom = `direct_${fieldWorker._id}_${adminId}`;
          socket.emit('join_room', { room: directRoom });
          console.log(`Joined direct chat room: ${directRoom}`);
        }
      }
      
      // Join the specific chat room if we have its ID
      if (chatRoomId) {
        const specificChatRoom = `chat_${chatRoomId}`;
        socket.emit('join_room', { room: specificChatRoom });
        console.log(`Joined specific chat room: ${specificChatRoom}`);
      }
    }
  }, [socket, connected, fieldWorker]);
  
  const markAsRead = useCallback(() => {
    if (socket && connected && fieldWorker?.tenant) {
      socket.emit('mark_as_read', { tenantId: fieldWorker.tenant });
      console.log(`Marked messages as read for tenant ${fieldWorker.tenant}`);
    }
  }, [socket, connected, fieldWorker]);
  
  const startTyping = useCallback(() => {
    if (socket && connected && fieldWorker?.tenant) {
      socket.emit('typing', { 
        tenantId: fieldWorker.tenant,
        isTyping: true,
        userName: fieldWorker?.name || 'Field Worker'
      });
    }
  }, [socket, connected, fieldWorker]);
  
  const stopTyping = useCallback(() => {
    if (socket && connected && fieldWorker?.tenant) {
      socket.emit('typing', { 
        tenantId: fieldWorker.tenant,
        isTyping: false,
        userName: fieldWorker?.name || 'Field Worker'
      });
    }
  }, [socket, connected, fieldWorker]);
  
  const value = {
    socket,
    connected,
    joinChat,
    markAsRead,
    startTyping,
    stopTyping
  };
  
  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};