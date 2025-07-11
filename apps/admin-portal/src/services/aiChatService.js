import api from './apiService';

/**
 * Service for interacting with the AI Chat API
 */
class AiChatService {
  /**
   * Create a new AI chat conversation
   * @param {string} initialMessage - First message from the user
   * @returns {Promise<Object>} - The created chat
   */
  createChat(initialMessage) {
    return api.post('/admin/ai-chat/chats', { initialMessage })
      .then(response => response.data);
  }
  
  /**
   * Get all AI chat conversations
   * @returns {Promise<Array>} - List of chat conversations
   */
  getAllChats() {
    return api.get('/admin/ai-chat/chats')
      .then(response => response.data);
  }
  
  /**
   * Get a specific AI chat conversation
   * @param {string} chatId - ID of the chat to retrieve
   * @returns {Promise<Object>} - Chat conversation details
   */
  getChat(chatId) {
    return api.get(`/admin/ai-chat/chats/${chatId}`)
      .then(response => response.data);
  }
  
  /**
   * Send a message in an existing chat
   * @param {string} chatId - ID of the chat
   * @param {string} message - Message to send
   * @returns {Promise<Object>} - The result with the AI response
   */
  sendMessage(chatId, message) {
    return api.post(`/admin/ai-chat/chats/${chatId}/messages`, { message })
      .then(response => response.data);
  }
  
  /**
   * Clear chat history
   * @returns {Promise<Object>} - Result of the clear operation
   */
  clearChatHistory() {
    return api.post('/admin/ai-chat/clear-history')
      .then(response => response.data);
  }
  
  /**
   * Analyze a road damage image with AI
   * @param {string} imageBase64 - Base64 encoded image
   * @param {Object} damageInfo - Additional information about the damage
   * @returns {Promise<Object>} - Analysis results
   */
  analyzeImage(imageBase64, damageInfo = {}) {
    return api.post('/admin/ai-chat/analyze-image', { imageBase64, damageInfo })
      .then(response => response.data);
  }
}

const aiChatService = new AiChatService();
export default aiChatService;
