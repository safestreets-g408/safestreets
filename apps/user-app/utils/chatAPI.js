import axios from 'axios';
import { API_BASE_URL } from '../config';
import { getAuthToken } from './auth';

// Create axios instance with default config
const chatAPI = axios.create({
  baseURL: `${API_BASE_URL.replace('/api', '')}/api/fieldworker/chat`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
chatAPI.interceptors.request.use(async (config) => {
  try {
    const token = await getAuthToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  } catch (err) {
    console.error('Error getting auth token for chat API:', err);
  }
  return config;
});

// Add response interceptor to handle auth errors
chatAPI.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('Chat API error:', error.response?.status, error.response?.data?.message);
    return Promise.reject(error);
  }
);

export const chatService = {
  // Get all admins the field worker can chat with
  getAdminChatList: async () => {
    try {
      const response = await chatAPI.get('/admins');
      return response.data;
    } catch (error) {
      console.error('Failed to get admin chat list:', error.message);
      throw error;
    }
  },
  
  // Get or create the tenant chat room for the field worker
  getChatRoom: async () => {
    try {
      // The backend will determine the tenant from the authenticated field worker
      const response = await chatAPI.post(`/room`);
      return response.data;
    } catch (error) {
      console.error(`Failed to get chat room:`, error.message);
      throw error;
    }
  },
  
  // Get messages for the tenant chat room
  getChatMessages: async (page = 1, limit = 50) => {
    try {
      // The backend will get the tenant from the user and find the right room
      const response = await chatAPI.get(`/messages`, {
        params: { page, limit }
      });
      return response.data;
    } catch (error) {
      console.error(`Failed to get messages:`, error.message);
      throw error;
    }
  },
  
  // Send a message to the tenant chat room
  sendMessage: async (content) => {
    try {
      const messageData = typeof content === 'string' 
        ? { message: content, messageType: 'text' } 
        : content;
      
      if (messageData.messageType && !['text', 'image', 'file'].includes(messageData.messageType)) {
        console.warn(`Invalid messageType: ${messageData.messageType}. Defaulting to 'text'`);
        messageData.messageType = 'text';
      }
      
      // The backend will determine the room from the authenticated user
      const response = await chatAPI.post(`/message`, messageData);
      return response.data;
    } catch (error) {
      console.error(`Failed to send message:`, error.message);
      throw error;
    }
  },
  
  // Mark messages in the tenant chat room as read
  markMessagesAsRead: async () => {
    try {
      // Backend determines room from user
      const response = await chatAPI.put(`/read`);
      return response.data;
    } catch (error) {
      console.error(`Failed to mark messages as read:`, error.message);
      throw error;
    }
  },
  
  // Send a damage report to the tenant chat room
  sendReportInChat: async (reportId) => {
    try {
      // Backend determines room from user
      const response = await chatAPI.post(`/share-report`, { reportId });
      return response.data;
    } catch (error) {
      console.error(`Failed to send report ${reportId}:`, error.message);
      throw error;
    }
  }
};