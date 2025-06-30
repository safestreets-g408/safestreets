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
  
  const joinChat = useCallback(() => {
    if (socket && connected && fieldWorker?.tenant) {
      const chatRoom = `tenant_${fieldWorker.tenant}`;
      socket.emit('join_room', { room: chatRoom });
      console.log(`Joined chat room: ${chatRoom}`);
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