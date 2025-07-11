/**
 * Field Worker AI Chat Controller
 * Handles field worker interactions with the Gemini AI assistant
 */

const AiChat = require('../models/AiChat');
const { generateChatResponse, analyzeRoadDamageImage, clearChatHistory } = require('../utils/geminiClient');
const config = require('../config');

// Create a new AI chat conversation
const createAiChat = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { initialMessage } = req.body;
    
    if (!initialMessage) {
      return res.status(400).json({ message: 'Initial message is required' });
    }
    
    // Generate AI response for the initial message
    const aiResponse = await generateChatResponse(
      fieldWorker._id.toString(),
      initialMessage,
      'field_worker'
    );
    
    // Create a new AI chat
    const chat = new AiChat({
      userId: fieldWorker._id,
      userModel: 'FieldWorker',
      tenantId: fieldWorker.tenant,
      title: initialMessage.slice(0, 50) + (initialMessage.length > 50 ? '...' : ''),
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
    console.error('Error creating field worker AI chat:', error);
    res.status(500).json({ message: 'Error creating AI chat', error: error.message });
  }
};

// Get all AI chat conversations for the field worker
const getFieldWorkerAiChats = async (req, res) => {
  try {
    const { fieldWorker } = req;
    
    const chats = await AiChat.find({
      userId: fieldWorker._id,
      userModel: 'FieldWorker'
    })
      .sort({ updatedAt: -1 })
      .limit(50);
    
    res.json(chats);
  } catch (error) {
    console.error('Error fetching field worker AI chats:', error);
    res.status(500).json({ message: 'Error fetching AI chats', error: error.message });
  }
};

// Get a specific AI chat conversation
const getAiChat = async (req, res) => {
  try {
    const { chatId } = req.params;
    const { fieldWorker } = req;
    
    const chat = await AiChat.findById(chatId);
    
    if (!chat) {
      return res.status(404).json({ message: 'AI chat not found' });
    }
    
    // Check ownership
    if (chat.userId.toString() !== fieldWorker._id.toString() || chat.userModel !== 'FieldWorker') {
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
    const { fieldWorker } = req;
    const { message } = req.body;
    
    if (!message) {
      return res.status(400).json({ message: 'Message is required' });
    }
    
    const chat = await AiChat.findById(chatId);
    
    if (!chat) {
      return res.status(404).json({ message: 'AI chat not found' });
    }
    
    // Check ownership
    if (chat.userId.toString() !== fieldWorker._id.toString() || chat.userModel !== 'FieldWorker') {
      return res.status(403).json({ message: 'Access denied' });
    }
    
    // Add user message to the chat
    chat.messages.push({
      content: message,
      role: 'user'
    });
    
    // Get AI response
    const aiResponse = await generateChatResponse(
      fieldWorker._id.toString(),
      message,
      'field_worker'
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

// Clear chat history for field worker
const clearChat = async (req, res) => {
  try {
    const { fieldWorker } = req;
    
    // Clear the history in memory
    clearChatHistory(fieldWorker._id.toString());
    
    res.json({ message: 'Chat history cleared successfully' });
  } catch (error) {
    console.error('Error clearing chat history:', error);
    res.status(500).json({ message: 'Error clearing chat history', error: error.message });
  }
};

// Get AI assistance with damage assessment
const getDamageAssistance = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { imageBase64, location, preliminaryAssessment } = req.body;
    
    if (!imageBase64) {
      return res.status(400).json({ message: 'Image is required' });
    }
    
    // Analyze the image with Gemini
    const damageInfo = {
      location,
      damageType: preliminaryAssessment?.damageType,
      severity: preliminaryAssessment?.severity
    };
    
    const analysisResult = await analyzeRoadDamageImage(imageBase64, damageInfo);
    
    res.json(analysisResult);
  } catch (error) {
    console.error('Error getting damage assistance:', error);
    res.status(500).json({ message: 'Error analyzing damage', error: error.message });
  }
};

module.exports = {
  createAiChat,
  getFieldWorkerAiChats,
  getAiChat,
  sendMessage,
  clearChat,
  getDamageAssistance
};
