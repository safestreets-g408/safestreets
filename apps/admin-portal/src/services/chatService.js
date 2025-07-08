import api from './apiService';

const CHAT_ENDPOINT = '/chat';

export const chatService = {
  // Test auth
  testAuth: async () => {
    const response = await api.get(`${CHAT_ENDPOINT}/test`);
    return response.data;
  },

  // Get all chat rooms (for super admin)
  getAllChatRooms: async () => {
    const response = await api.get(`${CHAT_ENDPOINT}/rooms`);
    return response.data;
  },

  // Get tenant chat rooms (for tenant admin)
  getTenantChatRooms: async () => {
    const response = await api.get(`${CHAT_ENDPOINT}/tenant-rooms`);
    return response.data;
  },

  // Get specific chat room
  getChatRoom: async (tenantId) => {
    const response = await api.get(`${CHAT_ENDPOINT}/room/${tenantId}`);
    return response.data;
  },

  // Get messages for a specific chat room
  getChatMessages: async (tenantId, page = 1, limit = 20) => {
    try {
      // Validate tenantId
      if (!tenantId) {
        throw new Error('Invalid tenant ID');
      }
      
      const response = await api.get(`${CHAT_ENDPOINT}/room/${tenantId}/messages`, {
        params: { page, limit }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to load messages:', error.response?.data || error.message);
      const errorMessage = 
        error.response?.data?.message || 
        error.message || 
        'Failed to load messages. Please check your connection and try again.';
      throw new Error(errorMessage);
    }
  },

  // Send a message with better error handling
  sendMessage: async (tenantId, content) => {
    try {
      console.log('Sending message to room:', tenantId, 'Content:', content);
      
      // Validate tenantId
      if (!tenantId) {
        throw new Error('Invalid tenant ID');
      }
      
      const messageData = typeof content === 'string' ? { message: content } : content;
      
      // Make sure the messageType is valid for the backend
      if (messageData.messageType && !['text', 'image', 'file'].includes(messageData.messageType)) {
        console.warn(`Invalid messageType: ${messageData.messageType}. Defaulting to 'text'`);
        messageData.messageType = 'text';
      }
      
      // Ensure message is not empty
      if (!messageData.message || messageData.message.trim() === '') {
        throw new Error('Message cannot be empty');
      }
      
      const response = await api.post(`${CHAT_ENDPOINT}/room/${tenantId}/message`, messageData);
      return response.data;
    } catch (error) {
      console.error('Failed to send message:', error.response?.data || error.message);
      const errorMessage = 
        error.response?.data?.message || 
        error.message || 
        'Failed to send message. Please check your connection and try again.';
      throw new Error(errorMessage);
    }
  },

  // Mark messages as read
  markMessagesAsRead: async (tenantId) => {
    const response = await api.put(`${CHAT_ENDPOINT}/room/${tenantId}/read`);
    return response.data;
  },

  // Get chat rooms based on user role (enhanced version)
  getChatRoomsByRole: async () => {
    const response = await api.get(`${CHAT_ENDPOINT}/rooms-by-role`);
    return response.data;
  },
};

export default chatService;
