/**
 * Field Worker AI Chat Routes
 */

const express = require('express');
const fieldWorkerAiChatController = require('../controllers/fieldWorkerAiChatController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');

const router = express.Router();

// Apply field worker authentication to all routes
router.use(protectFieldWorker);

// Create a new AI chat
router.post('/chats', fieldWorkerAiChatController.createAiChat);

// Get all AI chats for the field worker
router.get('/chats', fieldWorkerAiChatController.getFieldWorkerAiChats);

// Get a specific AI chat
router.get('/chats/:chatId', fieldWorkerAiChatController.getAiChat);

// Send a message to an AI chat
router.post('/chats/:chatId/messages', fieldWorkerAiChatController.sendMessage);

// Clear chat history
router.post('/clear-history', fieldWorkerAiChatController.clearChat);

// Get AI assistance for damage assessment
router.post('/damage-assistance', fieldWorkerAiChatController.getDamageAssistance);

module.exports = router;
