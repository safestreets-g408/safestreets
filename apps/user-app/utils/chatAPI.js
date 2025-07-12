import axios from 'axios';
import { getAuthToken } from './auth';
import { API_BASE_URL } from '../config';

// Create a separate axios instance for chat API
const chatAPI = axios.create();

// Variable to store base URL - will be initialized on first API call
let baseUrlInitialized = false;

// Function to get the base URL - it may need to be determined at runtime
const getBaseUrl = async () => {
  // First try the configured base URL
  if (API_BASE_URL && API_BASE_URL !== 'undefined') {
    console.log('Using API_BASE_URL from config:', API_BASE_URL);
    return API_BASE_URL;
  }

  // If not set, try multiple fallbacks
  console.log('API_BASE_URL not set, trying fallbacks');
  
  // Try common local development URLs
  const fallbackUrls = [
    'http://192.168.23.177:5030/api', // Manual override
    'http://localhost:5030/api',
    'http://127.0.0.1:5030/api',
    'http://10.0.2.2:5030/api' // For Android emulator
  ];
  
  for (const url of fallbackUrls) {
    try {
      const response = await fetch(`${url}/health`, { 
        method: 'GET',
        timeout: 2000
      });
      if (response.ok) {
        console.log(`Found working API at ${url}`);
        return url;
      }
    } catch (err) {
      console.log(`Failed to connect to ${url}:`, err.message);
    }
  }
  
  // Default fallback
  console.warn('No working API found, using default');
  return 'http://192.168.23.177:5030/api';
};

// Ensure the base URL is set
const ensureBaseUrl = async () => {
  if (!baseUrlInitialized) {
    const baseUrl = await getBaseUrl();
    chatAPI.defaults.baseURL = baseUrl;
    baseUrlInitialized = true;
    console.log('Chat API base URL initialized to:', baseUrl);
  }
};

// Add authorization token to requests
chatAPI.interceptors.request.use(
  async (config) => {
    try {
      const token = await getAuthToken();
      
      if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
      } else {
        console.warn('No token available for API request');
      }
      return config;
    } catch (error) {
      console.error('Error setting up authorization header:', error);
      return config;
    }
  },
  (error) => {
    console.error('Error in request interceptor:', error);
    return Promise.reject(error);
  }
);

// Handle responses and errors
chatAPI.interceptors.response.use(
  (response) => {
    // For successful responses, return the data property or the entire response
    return response.data || response;
  },
  async (error) => {
    if (error.response) {
      // The server responded with a status code outside the 2xx range
      console.error('Server error:', error.response.status, error.response.data);
      
      // If we get a 401 Unauthorized, we could handle token refresh here
      if (error.response.status === 401) {
        console.error('Unauthorized access - token may be expired');
      }
      
      // Add more specific error handling as needed
      if (error.response.status === 404) {
        console.error('Resource not found');
      }
    } else if (error.request) {
      console.error('Network error - no response received:', error.message);
    } else {
      console.error('Error setting up the request:', error.message);
    }
    return Promise.reject(error);
  }
);

// Export chatAPI for direct use if needed
export { chatAPI };

export const chatService = {
  // Health check for the chat service
  healthCheck: async () => {
    try {
      await ensureBaseUrl();
      console.log('Running chat service health check...');
      
      // Check backend API health with timeout
      let apiStatus = { api: false };
      try {
        const healthResponse = await chatAPI.get('/health', { timeout: 5000 });
        apiStatus = {
          api: true,
          message: 'API connection successful',
          data: healthResponse
        };
      } catch (apiError) {
        apiStatus = {
          api: false,
          error: apiError.message,
          code: apiError.code || apiError.response?.status,
          details: apiError.response?.data || 'No response data'
        };
        console.error('API health check failed:', apiStatus);
      }
      
      // Check socket connection if available
      let socketStatus = { socket: false };
      try {
        const { getSocket } = require('../context/SocketContext');
        const socket = getSocket();
        
        if (socket) {
          socketStatus = {
            socket: socket.connected,
            message: socket.connected ? 'Socket is connected' : 'Socket exists but disconnected',
            id: socket.id || 'no-id',
            nsp: socket.nsp || 'default',
            reconnecting: socket.reconnecting || false,
            disconnected: socket.disconnected || !socket.connected,
          };
          
          // If socket exists but is disconnected, try emitting a test event
          if (socket && !socket.connected) {
            socket.emit('ping_server');
            console.log('Sent test ping to server during health check');
          }
        } else {
          socketStatus = {
            socket: false,
            message: 'No socket instance found'
          };
        }
      } catch (socketError) {
        socketStatus = {
          socket: false,
          error: socketError.message,
          message: 'Error checking socket status'
        };
        console.error('Socket check error:', socketError);
      }
      
      // Determine overall health
      const healthy = apiStatus.api && (socketStatus.socket || apiStatus.api);
      
      return {
        healthy,
        api: apiStatus,
        socket: socketStatus,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        healthy: false,
        api: { api: false, error: error.message },
        socket: { socket: false, error: 'Not checked due to overall failure' },
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  },

  // Get all admins the field worker can chat with
  getAdminChatList: async () => {
    try {
      await ensureBaseUrl();
      const response = await chatAPI.get('/fieldworker/chat/admins');
      return response;
    } catch (error) {
      console.error('Failed to get admin chat list:', error.message);
      throw error;
    }
  },
  
  // Refresh chat list - handles various fallback scenarios
  refreshChats: async () => {
    try {
      await ensureBaseUrl();
      console.log('Refreshing chats list...');
      const response = await chatAPI.get('/fieldworker/chat/admins');
      console.log(`Refreshed ${response?.length || 0} chats successfully`);
      return response;
    } catch (error) {
      console.error('Failed to refresh chats:', error);
      
      // If the main endpoint fails, try a fallback approach
      try {
        console.log('Trying fallback approach to refresh chats...');
        // This is an alternate way to fetch chats if the main endpoint is failing
        const fallbackResponse = await chatAPI.get('/fieldworker/chat/rooms');
        
        // Transform the response to match the expected format if needed
        if (Array.isArray(fallbackResponse)) {
          return fallbackResponse;
        } else {
          throw new Error('Invalid response format from fallback endpoint');
        }
      } catch (fallbackError) {
        console.error('Fallback approach also failed:', fallbackError);
        throw error; // Throw the original error
      }
    }
  },
  
  // Get or create the tenant chat room for the field worker
  getChatRoom: async (adminId) => {
    try {
      if (!adminId) {
        console.error('getChatRoom called without adminId');
        throw new Error('Admin ID is required to get chat room');
      }
      
      console.log(`Making request to get chat room with admin ID: ${adminId}`);
      await ensureBaseUrl();
      
      let response;
      let lastError = null;
      let attempts = 0;
      const maxAttempts = 3;
      
      // Implement retry logic for the primary endpoint with exponential backoff
      while (attempts < maxAttempts && !response) {
        attempts++;
        try {
          // Primary endpoint - should create a room if it doesn't exist
          console.log(`Attempt ${attempts}/${maxAttempts}: Creating/getting chat room with primary endpoint`);
          response = await chatAPI.post('/fieldworker/chat/room', {
            recipientId: adminId,
            recipientType: 'admin'
          });
          console.log('Successfully created/got chat room with primary endpoint');
          break;
        } catch (err) {
          lastError = err;
          console.error(`Primary endpoint failed (attempt ${attempts}/${maxAttempts}):`, err.message);
          
          // Log detailed error information for debugging
          if (err.isAxiosError && err.response) {
            console.error('Error details:', {
              status: err.response.status,
              data: err.response.data
            });
            
            // If we get a 400 bad request, the data might be invalid, so don't retry
            if (err.response.status === 400) {
              console.error('Bad request (400) - invalid data. Not retrying.');
              throw err;
            }
          }
          
          // If this is the last attempt, don't wait
          if (attempts >= maxAttempts) {
            console.error(`All ${maxAttempts} attempts failed. Giving up on primary endpoint.`);
          } else {
            // Wait with exponential backoff before next attempt
            const waitTime = Math.min(1000 * Math.pow(2, attempts - 1), 5000);
            console.log(`Waiting ${waitTime}ms before next attempt...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
          }
        }
      }
      
      // If we still don't have a response after all attempts with the primary endpoint,
      // try a direct fallback with modified parameters as a last resort
      if (!response) {
        try {
          console.log('Trying direct fallback endpoint with different parameters');
          response = await chatAPI.post(`/fieldworker/chat/room`, {
            recipientId: adminId,
            recipientType: 'admin',
            createIfNotExists: true,
            forceCreation: true  // Add extra parameters that might help
          });
          console.log('Successfully created chat room with direct fallback');
        } catch (err3) {
          console.error('All approaches failed to create chat room:', err3.message);
          
          // As a last resort, create a mock chat room (temporary)
          console.log('All endpoints failed. Creating mock chat room as last resort');
          return {
            roomId: `chat_${adminId}_temporary_${Date.now()}`,
            temporary: true,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            participants: [
              { userId: adminId, userModel: 'Admin', name: 'Admin Support' }
            ]
          };
        }
      }
      
      if (response && response.data) {
        return response.data;
      } else {
        return response;
      }
    } catch (error) {
      console.error('Failed to get chat room:', error.message);
      
      // Return a fallback chat room as last resort
      return {
        roomId: `chat_${adminId}_error_${Date.now()}`,
        isErrorFallback: true,
        error: error.message,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        participants: [
          { id: adminId, type: 'Admin', name: 'Admin Support' }
        ]
      };
    }
  },
  
  // Get messages for a chat with an admin
  getChatMessages: async (adminId, page = 1, limit = 50) => {
    try {
      if (!adminId) {
        console.error('getChatMessages called without adminId');
        throw new Error('Admin ID is required to get chat messages');
      }
      
      // Ensure base URL is initialized
      await ensureBaseUrl();
      
      try {
        console.log(`Fetching messages for admin ${adminId} (page ${page}, limit ${limit})`);
        const response = await chatAPI.get(`/fieldworker/chat/admin/${adminId}/messages`, {
          params: { page, limit }
        });
        
        // Log success details
        const messageCount = response?.messages?.length || 0;
        console.log(`Successfully fetched ${messageCount} messages`);
        return response;
      } catch (err) {
        console.error('Error fetching messages:', err);
        
        if (err.isAxiosError && err.response) {
          console.error('Error response details:', {
            status: err.response.status,
            data: err.response.data
          });
          
          // If it's a 404, the chat room might not exist yet
          if (err.response.status === 404) {
            console.log('Chat room not found, attempting to create it first...');
            try {
              // Try to create the room first
              await chatService.getChatRoom(adminId);
              
              // Then retry getting messages
              console.log('Retrying message fetch after creating chat room...');
              const retryResponse = await chatAPI.get(`/fieldworker/chat/admin/${adminId}/messages`, {
                params: { page, limit }
              });
              return retryResponse;
            } catch (roomErr) {
              console.error('Failed to create chat room before fetching messages:', roomErr);
            }
          }
        }
        
        // Return empty data as fallback
        console.log('Returning empty messages array as fallback');
        return {
          messages: [],
          page: page,
          limit: limit,
          hasMore: false
        };
      }
    } catch (error) {
      console.error('Failed to get chat messages:', error.message);
      
      // Return empty data instead of throwing the error
      return {
        messages: [],
        page: page,
        limit: limit,
        hasMore: false,
        error: error.message
      };
    }
  },
  
  // Send a message to the tenant chat room
  sendMessage: async (adminId, content) => {
    try {
      if (!adminId) {
        console.error('sendMessage called without adminId');
        throw new Error('Admin ID is required to send a message');
      }
      
      // Ensure base URL is initialized
      await ensureBaseUrl();
      
      const messageData = typeof content === 'string' 
        ? { message: content, messageType: 'text' } 
        : content;
      
      if (messageData.messageType && !['text', 'image', 'file'].includes(messageData.messageType)) {
        console.warn(`Invalid messageType: ${messageData.messageType}. Defaulting to 'text'`);
        messageData.messageType = 'text';
      }
      
      console.log(`Sending message to admin ${adminId}:`, messageData);
      
      // First, make sure the chat room exists
      try {
        // This will create the room if it doesn't exist
        await chatService.getChatRoom(adminId);
      } catch (roomErr) {
        console.warn('Could not verify chat room before sending message:', roomErr.message);
        // Continue anyway, as the message endpoint might create the room
      }
      
      // Use the standard endpoint with retry logic
      let response = null;
      let attempts = 0;
      const maxAttempts = 3;
      
      while (attempts < maxAttempts && !response) {
        attempts++;
        
        try {
          // The backend will determine the room from the authenticated user
          console.log(`Attempt ${attempts}: Sending message to admin ${adminId}`);
          response = await chatAPI.post(`/fieldworker/chat/admin/${adminId}/message`, messageData);
          console.log('Message sent successfully, server response:', response);
          break;
        } catch (err) {
          console.error(`Attempt ${attempts} failed:`, err.message);
          
          if (err.isAxiosError && err.response) {
            console.error('Error details:', {
              status: err.response.status,
              data: err.response.data
            });
          }
          
          if (attempts >= maxAttempts) {
            console.error('All attempts failed. Rethrowing error.');
            throw err; // Re-throw after max attempts
          }
          
          // Try fallback endpoint on error
          try {
            console.log(`Attempt ${attempts}: Using fallback endpoint`);
            response = await chatAPI.post(`/fieldworker/chat/message`, {
              recipientId: adminId,
              recipientType: 'admin',
              ...messageData
            });
            console.log('Message sent using fallback endpoint');
            break;
          } catch (fallbackErr) {
            console.error('Fallback endpoint failed:', fallbackErr.message);
            // Wait a moment before next attempt
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
      
      // Message was sent successfully via API
      if (response && response.data) {
        // Log detailed information about the successful message
        console.log('Message sent successfully. Details:', {
          id: response.data._id || 'No ID returned',
          chatId: response.data.chatId || 'No chatId returned', 
          senderId: response.data.senderId || 'No senderId returned',
          timestamp: response.data.createdAt || 'No timestamp returned'
        });
        
        // Import socket context from global app context
        try {
          // Use the existing socket from the app's context if available
          const { getSocket } = require('../context/SocketContext');
          const existingSocket = getSocket && getSocket();
          
          if (existingSocket) {
            // Check connection status
            if (existingSocket.connected) {
              console.log('Socket is connected. Emitting message event...');
              
              // Ensure connection is working by sending a ping first
              existingSocket.emit('ping_server');
              
              // Emit using the existing socket connection
              existingSocket.emit('message_sent', {
                ...response.data,
                adminId,
                fromFieldWorker: true,
                timestamp: new Date().toISOString()
              });
              
              console.log('Message event emitted through existing socket');
            } else {
              console.log('Socket exists but is not connected. Attempting reconnection...');
              
              // Try to reconnect the socket
              existingSocket.connect();
              
              // Set a timeout to check if connection is established and then emit
              setTimeout(() => {
                if (existingSocket.connected) {
                  console.log('Socket reconnected successfully. Emitting message event...');
                  existingSocket.emit('message_sent', {
                    ...response.data,
                    adminId,
                    fromFieldWorker: true,
                    timestamp: new Date().toISOString()
                  });
                } else {
                  console.warn('Failed to reconnect socket. Message sent via API but not via socket.');
                }
              }, 1000);
            }
          } else {
            console.log('No socket instance available. Message sent via API only.');
          }
        } catch (contextErr) {
          console.warn('Could not access socket from context:', contextErr.message);
          console.error('Socket context error details:', contextErr);
        }
      }
      
      // Handle both cases for safety
      return response.data || response;
    } catch (error) {
      console.error(`Failed to send message to admin ${adminId}:`, error.message);
      
      if (error.isAxiosError && error.response) {
        console.error('API error details:', {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers
        });
      }
      
      throw error;
    }
  },
  
  // Mark messages in the tenant chat room as read
  markMessagesAsRead: async (adminId) => {
    try {
      if (!adminId) {
        console.error('markMessagesAsRead called without adminId');
        throw new Error('Admin ID is required to mark messages as read');
      }
      
      // Ensure base URL is initialized
      await ensureBaseUrl();
      
      const response = await chatAPI.put(`/fieldworker/chat/admin/${adminId}/read`);
      return response;
    } catch (error) {
      console.error('Failed to mark messages as read:', error.message);
      throw error;
    }
  },
  
  // Send a damage report in a chat
  sendReportInChat: async (adminId, reportId) => {
    try {
      // Ensure base URL is initialized
      await ensureBaseUrl();
      
      const response = await chatAPI.post(`/fieldworker/chat/admin/${adminId}/share-report`, {
        reportId
      });
      return response;
    } catch (error) {
      console.error('Failed to share report:', error.message);
      throw error;
    }
  },

  // Check chat service health and connectivity
  healthCheck: async () => {
    try {
      // Ensure base URL is initialized
      await ensureBaseUrl();
      
      console.log('Performing chat service health check...');
      
      // First, check API connectivity by getting admin list
      const apiCheck = await chatAPI.get('/fieldworker/chat/admins')
        .then(response => {
          console.log('API health check successful');
          return { api: true, message: 'API connection successful' };
        })
        .catch(error => {
          console.error('API health check failed:', error.message);
          return { api: false, message: `API connection failed: ${error.message}` };
        });
      
      // Then check socket connectivity if possible
      let socketCheck = { socket: false, message: 'Socket not initialized' };
      try {
        const { getSocket } = require('../context/SocketContext');
        const socket = getSocket && getSocket();
        
        if (socket) {
          socketCheck = socket.connected 
            ? { socket: true, message: 'Socket is connected' }
            : { socket: false, message: 'Socket exists but is disconnected' };
        }
      } catch (socketErr) {
        console.error('Socket check failed:', socketErr.message);
        socketCheck = { socket: false, message: `Socket check error: ${socketErr.message}` };
      }
      
      // Return combined health status
      return {
        timestamp: new Date().toISOString(),
        healthy: apiCheck.api && socketCheck.socket,
        api: apiCheck,
        socket: socketCheck
      };
    } catch (error) {
      console.error('Chat health check failed:', error.message);
      return {
        timestamp: new Date().toISOString(),
        healthy: false,
        error: error.message
      };
    }
  }
};
