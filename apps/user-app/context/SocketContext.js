import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { io } from 'socket.io-client';
import { useAuth } from './AuthContext';
import { getAuthToken } from '../utils/auth';
import { API_BASE_URL } from '../config';

const SocketContext = createContext();
// Singleton socket reference for external access
let socketInstance = null;

// Function to get socket instance from outside React context
export const getSocket = () => socketInstance;

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
    // Reset reference on unmount
    return () => {
      if (socketInstance) {
        socketInstance.disconnect();
        socketInstance = null;
      }
    };
  }, []);
  
  useEffect(() => {
    // Clear any existing socket
    if (socketInstance) {
      socketInstance.disconnect();
      socketInstance = null;
    }
    
    const initializeSocket = async () => {
      if (!isAuthenticated || !fieldWorker) return;
      
      try {
        const token = await getAuthToken();
        if (!token) return;
        
        // Determine the socket server URL
        let socketUrl = API_BASE_URL;
        // If the URL ends with /api, remove it to get the socket server URL
        if (socketUrl.endsWith('/api')) {
          socketUrl = socketUrl.replace('/api', '');
        }
        
        console.log('Initializing socket connection to:', socketUrl);
        
        socketInstance = io(socketUrl, {
          auth: {
            token,
            userType: 'fieldworker',
            userId: fieldWorker._id
          },
          transports: ['websocket', 'polling'],
          reconnection: true,
          reconnectionAttempts: 10,  // Increase retry attempts
          reconnectionDelay: 1000,
          reconnectionDelayMax: 5000,
          timeout: 20000  // Increase timeout for slow connections
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
        
        // Add ping-pong for connection testing
        socketInstance.on('ping_server', () => {
          console.log('Ping received from server');
          socketInstance.emit('pong_client');
        });
        
        socketInstance.on('pong_server', () => {
          console.log('Server responded to ping');
          setConnected(true);
        });
        
        setSocket(socketInstance);
      } catch (error) {
        console.error('Error initializing socket:', error);
        socketInstance = null;
      }
    };
    
    initializeSocket();
    
    return () => {
      // Don't disconnect on component unmount,
      // we'll keep the socket reference for the app lifecycle
    };
  }, [isAuthenticated, fieldWorker]);
  
  const joinChat = useCallback((adminId, chatRoomId) => {
    if (socket && connected && fieldWorker?.tenant) {
      console.log('Joining chat rooms with adminId:', adminId, 'and chatRoomId:', chatRoomId);
      
      // Join tenant-wide chat room
      const tenantRoom = `tenant_${fieldWorker.tenant}`;
      socket.emit('join_room', { room: tenantRoom });
      console.log(`Joined tenant chat room: ${tenantRoom}`);
      
      // Also join the tenant chat room using the chat_ prefix for consistency
      const tenantChatRoom = `chat_${fieldWorker.tenant}`;
      socket.emit('join_room', { room: tenantChatRoom });
      console.log(`Joined tenant chat room with chat_ prefix: ${tenantChatRoom}`);
      
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
      
      // Force reconnect to ensure connection is fresh
      socket.emit('ping_server');
    } else {
      console.warn('Cannot join chat: socket not connected or missing fieldworker data');
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
  
  // Explicitly reconnect socket if needed
  const reconnectSocket = useCallback(() => {
    console.log('Explicitly reconnecting socket...');
    if (socketInstance) {
      // If socket exists but is disconnected, try to reconnect
      if (!socketInstance.connected) {
        console.log('Socket disconnected. Attempting to reconnect...');
        socketInstance.connect();
        return true;
      } else {
        console.log('Socket already connected.');
        return false;
      }
    } else {
      console.log('No socket instance. Initializing new socket...');
      // Reinitialize the socket
      initializeSocket();
      return true;
    }
  }, [socketInstance]);

  const value = {
    socket,
    connected,
    joinChat,
    markAsRead,
    startTyping,
    stopTyping,
    reconnectSocket
  };
  
  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};