/**
 * Admin AI Chat Routes
 */

const express = require('express');
const adminAiChatController = require('../controllers/adminAiChatController');
const { protectAdmin } = require('../middleware/adminAuthMiddleware');

const router = express.Router();

// Apply admin authentication to all routes
router.use(protectAdmin);

// Create a new AI chat
router.post('/chats', adminAiChatController.createAiChat);

// Get all AI chats for the admin
router.get('/chats', adminAiChatController.getAdminAiChats);

// Get a specific AI chat
router.get('/chats/:chatId', adminAiChatController.getAiChat);

// Send a message to an AI chat
router.post('/chats/:chatId/messages', adminAiChatController.sendMessage);

// Clear chat history
router.post('/clear-history', adminAiChatController.clearChat);

// Analyze a road damage image with AI
router.post('/analyze-image', adminAiChatController.analyzeImage);

module.exports = router;
