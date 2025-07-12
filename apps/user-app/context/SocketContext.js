import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { Platform, AppState } from 'react-native';
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
  const [connectionError, setConnectionError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const { isAuthenticated, fieldWorker } = useAuth();
  
  // Initialize socket connection
  const initializeSocketConnection = useCallback(async () => {
    if (!isAuthenticated || !fieldWorker) {
      console.log('Not authenticated or no fieldWorker data, skipping socket connection');
      return;
    }
    
    if (isInitializing) {
      console.log('Socket initialization already in progress, skipping...');
      return;
    }
    
    setIsInitializing(true);
    
    try {
      const token = await getAuthToken();
      if (!token) {
        console.error('No auth token available for socket connection');
        setConnectionError('Authentication required');
        return;
      }
      
      // Clean up existing socket if it exists
      if (socketInstance) {
        console.log('Cleaning up existing socket before creating new one');
        socketInstance.removeAllListeners(); // Remove all event listeners first
        socketInstance.disconnect();
        socketInstance = null;
      }
      
      // Determine the socket server URL
      let socketUrl = API_BASE_URL;
      // If the URL ends with /api, remove it to get the socket server URL
      if (socketUrl.endsWith('/api')) {
        socketUrl = socketUrl.replace('/api', '');
      }
      
      console.log('Initializing new socket connection to:', socketUrl);
      
      // Format auth properly - the server may be expecting a specific format
      // First, modify the URL to include the token as a query parameter as some servers prefer this
      const urlWithToken = `${socketUrl}?token=${encodeURIComponent(token)}&userType=fieldworker&userId=${encodeURIComponent(fieldWorker._id)}`;
      
      console.log('Connecting to socket with auth token included in URL');
      
      socketInstance = io(urlWithToken, {
        // Keep auth in options too for redundancy (server might check in multiple places)
        auth: {
          token: token,  // Make sure the token is properly formatted
          userType: 'fieldworker',
          userId: fieldWorker._id,
          // Add extra fields that might help with auth
          tenantId: fieldWorker.tenant,
          authorization: `Bearer ${token}` // Some servers expect this format
        },
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 30,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 10000,
        timeout: 30000,
        forceNew: true,               // Change to true to avoid reusing cached connections with old auth
        multiplex: false,             // Disable multiplexing to ensure fresh connection
        autoConnect: true,
        rejectUnauthorized: false,
        extraHeaders: {              // Add token in headers too for maximum compatibility
          'Authorization': `Bearer ${token}`,
          'X-User-Type': 'fieldworker',
          'X-User-Id': fieldWorker._id
        }
      });
      
      // Handle authentication confirmation from server
      socketInstance.on('authenticated', () => {
        console.log('Server confirmed authentication success');
        // We now know the server accepted our credentials
      });
      
      // Handle authentication errors from server
      socketInstance.on('auth_error', (message) => {
        console.error('Authentication error from server:', message);
        setConnectionError(`Authentication failed: ${message}`);
        // Don't try to reconnect immediately, user might need to re-login
      });
      
      socketInstance.on('connect', () => {
        console.log('Socket connected successfully');
        setConnected(true);
        setConnectionError(null);
        
        // After connection, explicitly authenticate again to ensure server recognizes us
        getAuthToken().then(currentToken => {
          if (currentToken && typeof currentToken === 'string' && currentToken.trim() !== '') {
            console.log('Sending explicit authentication after connect, token type:', typeof currentToken, 'length:', currentToken.length);
            socketInstance.emit('authenticate', currentToken);
          } else {
            console.error('Invalid token for authentication:', typeof currentToken, currentToken);
          }
        });
        
        // Join tenant room automatically on connect
        if (fieldWorker?.tenant) {
          const tenantRoom = `tenant_${fieldWorker.tenant}`;
          socketInstance.emit('join_room', { 
            room: tenantRoom,
            userId: fieldWorker._id,
            userType: 'fieldworker'
          });
          console.log(`Auto-joined tenant room: ${tenantRoom}`);
        }
      });
      
      socketInstance.on('disconnect', (reason) => {
        console.log(`Socket disconnected. Reason: ${reason}`);
        setConnected(false);
        
        // If the server closed the connection due to auth issues, we need to reinitialize
        if (reason === 'io server disconnect' || reason === 'io client disconnect') {
          console.log('Disconnect initiated by server or client, will need manual reconnect');
          // We don't auto-reconnect here as it might be intentional
        }
      });
      
      socketInstance.on('error', (error) => {
        console.error('Socket error:', error);
        setConnected(false);
        setConnectionError(error);
        
        // Handle authentication errors specifically
        if (typeof error === 'string' && error.includes('authenticated')) {
          console.log('Authentication error detected, will try to refresh token and reconnect');
          
          // Wait a moment then try to refresh the connection with a new token
          setTimeout(async () => {
            try {
              // Get a fresh token
              const freshToken = await getAuthToken();
              if (freshToken) {
                console.log('Got fresh token, reinitializing socket connection');
                // Clean up the existing socket
                if (socketInstance) {
                  socketInstance.disconnect();
                  socketInstance = null;
                }
                // Reinitialize with fresh token
                initializeSocketConnection();
              }
            } catch (err) {
              console.error('Failed to get fresh token:', err);
            }
          }, 2000);
        }
      });
      
      socketInstance.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        setConnectionError(`Connection error: ${error.message}`);
        
        // Check if this is an auth error
        if (error.message.includes('auth') || error.message.includes('unauthorized')) {
          console.log('Authentication error in connection, will try to get new token');
          
          // Wait and try with fresh token
          setTimeout(async () => {
            try {
              const freshToken = await getAuthToken();
              if (freshToken) {
                console.log('Got fresh token after connect error, reinitializing');
                initializeSocketConnection();
              }
            } catch (err) {
              console.error('Failed to get fresh token after connect error:', err);
            }
          }, 3000);
        }
      });
      
      socketInstance.on('reconnect_failed', () => {
        console.error('Socket reconnection failed after all attempts');
        setConnectionError('Failed to reconnect after multiple attempts');
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
      
      // Listen for new messages and notifications
      socketInstance.on('new_message', (message) => {
        console.log('New message received:', message);
        // You can add logic here to update UI or show notification
      });
      
      socketInstance.on('notification', (notification) => {
        console.log('Notification received:', notification);
        // You can add logic here to show notification
      });
      
      setSocket(socketInstance);
      setIsInitializing(false);
    } catch (error) {
      console.error('Error initializing socket:', error);
      setConnectionError(`Initialization error: ${error.message}`);
      socketInstance = null;
      setIsInitializing(false);
    }
  }, [isAuthenticated, fieldWorker, isInitializing]);
  
  // Function to reconnect the socket - exposed for external use
  const reconnectSocket = useCallback(() => {
    console.log('Manually reconnecting socket...');
    
    // Prevent multiple concurrent reconnection attempts
    if (reconnectSocket.isReconnecting) {
      console.log('Reconnection already in progress, skipping...');
      return;
    }
    reconnectSocket.isReconnecting = true;
    
    // Reset connection attempts counter
    setConnectionAttempts(0);
    
    // Clean up and reinitialize the socket completely
    try {
      console.log('Performing complete socket reinitialization...');
      
      // Clean up existing socket completely
      if (socketInstance) {
        console.log('Cleaning up existing socket for reconnection');
        socketInstance.removeAllListeners();
        socketInstance.disconnect();
        socketInstance = null;
        setSocket(null);
        setConnected(false);
      }
      
      // Wait a moment before reinitializing
      setTimeout(() => {
        initializeSocketConnection().finally(() => {
          reconnectSocket.isReconnecting = false;
        });
      }, 1000);
      
    } catch (error) {
      console.error('Error during reconnect process:', error);
      reconnectSocket.isReconnecting = false;
      // Final fallback - just try to reinitialize
      initializeSocketConnection();
    }
  }, [initializeSocketConnection]);
  
  // Keep track of connection attempts
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  
  // Add heartbeat to check connection periodically
  useEffect(() => {
    if (!connected || !socketInstance) return;
    
    // Set up heartbeat to check connection every 30 seconds (increased from 20s)
    const heartbeatInterval = setInterval(() => {
      if (socketInstance && socketInstance.connected) {
        // Socket is connected, send a ping to keep it alive
        socketInstance.emit('ping_server');
        console.log('Heartbeat: sent ping to server');
      } else if (socketInstance && !socketInstance.connected && connectionAttempts < 3) {
        // Socket exists but is disconnected, try to reconnect (reduced max attempts)
        console.log('Heartbeat: Socket disconnected, attempting reconnect');
        setConnectionAttempts(prev => prev + 1);
        // Use reconnectSocket instead of direct connect to avoid duplicates
        reconnectSocket();
      } else if (connectionAttempts >= 3) {
        // Too many failed attempts, stop trying for now
        console.log('Heartbeat: Too many failed attempts, stopping automatic reconnection');
        setConnectionError('Connection lost - please try manual reconnect');
        setConnectionAttempts(0);
      }
    }, 30000); // Increased interval to reduce connection pressure
    
    return () => {
      clearInterval(heartbeatInterval);
    };
  }, [connected, socketInstance, connectionAttempts, reconnectSocket]);
  
  // Initialize socket when authenticated
  useEffect(() => {
    if (isAuthenticated && fieldWorker) {
      initializeSocketConnection();
      setConnectionAttempts(0);
    }
    
    return () => {
      // Don't disconnect on component unmount,
      // we'll keep the socket reference for the app lifecycle
    };
  }, [isAuthenticated, fieldWorker, initializeSocketConnection]);
  
  // Monitor app state changes
  useEffect(() => {
    const handleAppStateChange = (nextAppState) => {
      console.log('App state changed to:', nextAppState);
      if (nextAppState === 'active') {
        // App has come to the foreground
        console.log('App is active, checking socket connection');
        if (socketInstance && !socketInstance.connected) {
          console.log('Socket disconnected while app was inactive, reconnecting');
          reconnectSocket();
        } else if (!socketInstance) {
          console.log('No socket instance found after app became active, initializing');
          initializeSocketConnection();
        } else {
          console.log('Socket appears to be connected, sending ping to verify');
          socketInstance.emit('ping_server');
        }
      } else if (nextAppState === 'background') {
        // App has gone to the background
        console.log('App went to background, socket will try to stay connected');
        // We don't disconnect here - let the socket try to maintain connection
      }
    };
    
    // Subscribe to app state changes
    const subscription = AppState.addEventListener('change', handleAppStateChange);
    
    return () => {
      // Clean up subscription
      subscription.remove();
    };
  }, [socketInstance, reconnectSocket, initializeSocketConnection]);
  
  // Join specific chat rooms
  const joinChat = useCallback(async (adminId, chatRoomId) => {
    if (!socket || !connected || !fieldWorker?.tenant) {
      console.warn('Cannot join chat: socket not connected or missing fieldworker data');
      return;
    }
    
    try {
      console.log('Joining chat rooms with adminId:', adminId, 'and chatRoomId:', chatRoomId);
      
      // First re-authenticate only if socket seems disconnected or unauthenticated
      const currentToken = await getAuthToken();
      if (!currentToken || typeof currentToken !== 'string' || currentToken.trim() === '') {
        console.error('No valid token available when attempting to join chat rooms:', typeof currentToken, currentToken);
        return;
      }
      
      // Only re-authenticate if the socket appears to need it
      if (!socket.connected || !socket.userId) {
        console.log('Socket not connected or not authenticated, re-authenticating for chat rooms');
        socket.emit('authenticate', currentToken);
        
        // Wait a moment for authentication to complete
        await new Promise(resolve => setTimeout(resolve, 500));
      } else {
        console.log('Socket already authenticated, proceeding to join rooms');
      }
      
      // Create a standard payload with auth info for all room joins
      const basePayload = {
        userId: fieldWorker._id,
        userType: 'fieldworker',
        token: currentToken,
        tenantId: fieldWorker.tenant
      };
      
      // Join tenant-wide chat room with auth info
      const tenantRoom = `tenant_${fieldWorker.tenant}`;
      socket.emit('join_room', { 
        ...basePayload,
        room: tenantRoom 
      });
      console.log(`Joined tenant chat room: ${tenantRoom}`);
      
      // Also join the tenant chat room using the chat_ prefix for consistency
      const tenantChatRoom = `chat_${fieldWorker.tenant}`;
      socket.emit('join_room', { 
        ...basePayload,
        room: tenantChatRoom 
      });
      console.log(`Joined tenant chat room with chat_ prefix: ${tenantChatRoom}`);
      
      // Join specific admin chat room if provided
      if (adminId) {
        const adminRoom = `admin_${adminId}`;
        socket.emit('join_room', { 
          ...basePayload,
          room: adminRoom,
          recipientId: adminId
        });
        console.log(`Joined admin chat room: ${adminRoom}`);
        
        // Also join the direct chat room (if it exists)
        if (fieldWorker?._id) {
          const directRoom = `direct_${fieldWorker._id}_${adminId}`;
          socket.emit('join_room', { 
            ...basePayload,
            room: directRoom,
            recipientId: adminId
          });
          console.log(`Joined direct chat room: ${directRoom}`);
        }
      }
      
      // Join the specific chat room if we have its ID
      if (chatRoomId) {
        const specificChatRoom = `chat_${chatRoomId}`;
        socket.emit('join_room', { 
          ...basePayload,
          room: specificChatRoom,
          chatRoomId: chatRoomId
        });
        console.log(`Joined specific chat room: ${specificChatRoom}`);
      }
      
      // Force a ping to verify connection after joining rooms
      socket.emit('ping_server');
      
    } catch (error) {
      console.error('Error during chat room joins:', error);
    }
  }, [socket, connected, fieldWorker]);
  
  // Mark messages as read
  const markAsRead = useCallback(async (roomId) => {
    if (!socket || !connected || !fieldWorker?._id) return;
    
    try {
      const currentToken = await getAuthToken();
      if (!currentToken) return;
      
      const basePayload = {
        userId: fieldWorker._id,
        userType: 'fieldworker',
        token: currentToken
      };
      
      if (roomId) {
        socket.emit('mark_as_read', { 
          ...basePayload,
          roomId,
        });
        console.log(`Marked messages as read for room ${roomId}`);
      } else if (fieldWorker.tenant) {
        socket.emit('mark_as_read', { 
          ...basePayload,
          tenantId: fieldWorker.tenant,
        });
        console.log(`Marked all messages as read for tenant ${fieldWorker.tenant}`);
      }
    } catch (error) {
      console.error('Error marking messages as read:', error);
    }
  }, [socket, connected, fieldWorker]);
  
  // Typing indicators
  const startTyping = useCallback(async (roomId) => {
    if (!socket || !connected || !fieldWorker?._id) return;
    
    try {
      const currentToken = await getAuthToken();
      if (!currentToken) return;
      
      socket.emit('typing', { 
        roomId,
        userId: fieldWorker._id,
        userName: fieldWorker?.name || 'Field Worker',
        isTyping: true,
        token: currentToken
      });
    } catch (error) {
      console.error('Error sending typing indicator:', error);
    }
  }, [socket, connected, fieldWorker]);
  
  const stopTyping = useCallback(async (roomId) => {
    if (!socket || !connected || !fieldWorker?._id) return;
    
    try {
      const currentToken = await getAuthToken();
      if (!currentToken) return;
      
      socket.emit('typing', { 
        roomId,
        userId: fieldWorker._id,
        userName: fieldWorker?.name || 'Field Worker',
        isTyping: false,
        token: currentToken
      });
    } catch (error) {
      console.error('Error sending stop typing indicator:', error);
    }
  }, [socket, connected, fieldWorker]);
  
  return (
    <SocketContext.Provider value={{
      socket,
      connected,
      connectionError,
      joinChat,
      markAsRead,
      startTyping,
      stopTyping,
      reconnectSocket
    }}>
      {children}
    </SocketContext.Provider>
  );
};
