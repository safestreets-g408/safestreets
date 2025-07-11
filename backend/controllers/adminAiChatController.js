/**
 * Admin AI Chat Controller
 * Handles admin interactions with the Gemini AI assistant
 */

const AiChat = require('../models/AiChat');
const { generateChatResponse, analyzeRoadDamageImage, clearChatHistory } = require('../utils/geminiClient');
const config = require('../config');

// Create a new AI chat conversation
const createAiChat = async (req, res) => {
  try {
    const { admin } = req;
    const { initialMessage } = req.body;
    
    if (!initialMessage) {
      return res.status(400).json({ message: 'Initial message is required' });
    }
    
    // Generate AI response for the initial message
    const aiResponse = await generateChatResponse(
      admin._id.toString(),
      initialMessage,
      admin.role === 'super-admin' ? 'super_admin' : 'admin'
    );
    
    // Generate a more descriptive title for the chat
    let chatTitle = initialMessage.slice(0, 50) + (initialMessage.length > 50 ? '...' : '');
    
    // Try to extract a more meaningful title from the AI response
    if (aiResponse) {
      // Look for the first sentence or meaningful chunk in the AI response
      const firstLine = aiResponse.split('\n')[0].trim();
      if (firstLine && firstLine.length > 10 && firstLine.length < 80) {
        chatTitle = firstLine.replace(/^(Hello|Hi|Greetings|Sure|I'd be happy to help|I can help with that).*?,\s*/i, '');
      }
    }
    
    // Create a new AI chat
    const chat = new AiChat({
      userId: admin._id,
      userModel: 'Admin',
      tenantId: admin.tenant,
      title: chatTitle,
      messages: [
        {
          content: initialMessage,
          role: 'user'
        },
        {
          content: aiResponse,
          role: 'ai',
          metadata: {
            modelName: config.gemini.defaultModel
          }
        }
      ]
    });
    
    await chat.save();
    
    res.status(201).json(chat);
  } catch (error) {
    console.error('Error creating AI chat:', error);
    res.status(500).json({ message: 'Error creating AI chat', error: error.message });
  }
};

// Get all AI chat conversations for the admin
const getAdminAiChats = async (req, res) => {
  try {
    const { admin } = req;
    
    // Apply tenant isolation for non-super-admin
    const query = { 
      userId: admin._id,
      userModel: 'Admin'
    };
    
    const chats = await AiChat.find(query)
      .sort({ updatedAt: -1 })
      .limit(50);
    
    res.json(chats);
  } catch (error) {
    console.error('Error fetching admin AI chats:', error);
    res.status(500).json({ message: 'Error fetching AI chats', error: error.message });
  }
};

// Get a specific AI chat conversation
const getAiChat = async (req, res) => {
  try {
    const { chatId } = req.params;
    const { admin } = req;
    
    const chat = await AiChat.findById(chatId);
    
    if (!chat) {
      return res.status(404).json({ message: 'AI chat not found' });
    }
    
    // Check ownership
    if (chat.userId.toString() !== admin._id.toString() || chat.userModel !== 'Admin') {
      return res.status(403).json({ message: 'Access denied' });
    }
    
    res.json(chat);
  } catch (error) {
    console.error('Error fetching AI chat:', error);
    res.status(500).json({ message: 'Error fetching AI chat', error: error.message });
  }
};

// Send a message to the AI assistant
const sendMessage = async (req, res) => {
  try {
    const { chatId } = req.params;
    const { admin } = req;
    const { message } = req.body;
    
    if (!message) {
      return res.status(400).json({ message: 'Message is required' });
    }
    
    const chat = await AiChat.findById(chatId);
    
    if (!chat) {
      return res.status(404).json({ message: 'AI chat not found' });
    }
    
    // Check ownership
    if (chat.userId.toString() !== admin._id.toString() || chat.userModel !== 'Admin') {
      return res.status(403).json({ message: 'Access denied' });
    }
    
    // Add user message to the chat
    chat.messages.push({
      content: message,
      role: 'user'
    });
    
    // Get AI response
    const aiResponse = await generateChatResponse(
      admin._id.toString(),
      message,
      admin.role === 'super-admin' ? 'super_admin' : 'admin'
    );
    
    // Add AI response to the chat
    chat.messages.push({
      content: aiResponse,
      role: 'ai',
      metadata: {
        modelName: config.gemini.defaultModel
      }
    });
    
    // Update chat
    chat.updatedAt = Date.now();
    await chat.save();
    
    res.json({
      chatId: chat._id,
      userMessage: message,
      aiResponse: aiResponse
    });
  } catch (error) {
    console.error('Error sending message to AI:', error);
    res.status(500).json({ message: 'Error sending message', error: error.message });
  }
};

// Clear chat history for user
const clearChat = async (req, res) => {
  try {
    const { admin } = req;
    
    // Clear the history in memory
    clearChatHistory(admin._id.toString());
    
    res.json({ message: 'Chat history cleared successfully' });
  } catch (error) {
    console.error('Error clearing chat history:', error);
    res.status(500).json({ message: 'Error clearing chat history', error: error.message });
  }
};

// Analyze a road damage image with AI assistance
const analyzeImage = async (req, res) => {
  try {
    const { admin } = req;
    const { imageBase64, damageInfo } = req.body;
    
    if (!imageBase64) {
      return res.status(400).json({ message: 'Image is required' });
    }
    
    // Analyze the image with Gemini
    const analysisResult = await analyzeRoadDamageImage(
      imageBase64, 
      damageInfo || {}
    );
    
    res.json(analysisResult);
  } catch (error) {
    console.error('Error analyzing image with AI:', error);
    res.status(500).json({ message: 'Error analyzing image', error: error.message });
  }
};

module.exports = {
  createAiChat,
  getAdminAiChats,
  getAiChat,
  sendMessage,
  clearChat,
  analyzeImage
};
