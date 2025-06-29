const express = require('express');
const router = express.Router();
const { protectAdmin } = require('../middleware/adminAuthMiddleware');
const {
  getChatRoom,
  getAllChatRooms,
  getTenantChatRooms,
  getChatMessages,
  sendMessage,
  markMessagesAsRead,
  getChatRoomsByRole
} = require('../controllers/chatController');

// Test route to verify auth
router.get('/test', protectAdmin, (req, res) => {
  console.log('Chat test endpoint hit by:', req.admin.name, req.admin.role);
  res.json({ 
    message: 'Chat auth working', 
    admin: { 
      id: req.admin._id, 
      name: req.admin.name, 
      role: req.admin.role,
      tenant: req.admin.tenant ? {
        id: req.admin.tenant._id,
        name: req.admin.tenant.name
      } : null
    },
    timestamp: new Date().toISOString()
  });
});

// Get all chat rooms (for super admin)
router.get('/rooms', protectAdmin, getAllChatRooms);

// Get tenant chat rooms (for tenant admins - shows super admin contact)
router.get('/tenant-rooms', protectAdmin, getTenantChatRooms);

// Get or create chat room for a tenant
router.get('/room/:tenantId', protectAdmin, getChatRoom);

// Get messages for a chat room
router.get('/room/:tenantId/messages', protectAdmin, getChatMessages);

// Send a message
router.post('/room/:tenantId/message', protectAdmin, sendMessage);

// Mark messages as read
router.put('/room/:tenantId/read', protectAdmin, markMessagesAsRead);

// Get chat rooms based on user role
router.get('/rooms-by-role', protectAdmin, getChatRoomsByRole);

module.exports = router;
