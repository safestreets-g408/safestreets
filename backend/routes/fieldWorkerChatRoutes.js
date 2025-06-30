const express = require('express');
const router = express.Router();
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const {
  getAdminChatList,
  getChatRoom,
  getChatMessages,
  sendMessage,
  markMessagesAsRead,
  shareReport
} = require('../controllers/fieldWorkerChatController');

// Get list of admins the field worker can chat with
router.get('/admins', protectFieldWorker, getAdminChatList);

// Get or create chat room with admin
router.post('/room', protectFieldWorker, getChatRoom);

// Get messages for a chat with an admin
router.get('/admin/:adminId/messages', protectFieldWorker, getChatMessages);

// Send message to an admin
router.post('/admin/:adminId/message', protectFieldWorker, sendMessage);

// Mark messages from an admin as read
router.put('/admin/:adminId/read', protectFieldWorker, markMessagesAsRead);

// Share a damage report with an admin
router.post('/admin/:adminId/share-report', protectFieldWorker, shareReport);

module.exports = router;
