const express = require('express');
const { 
  getFieldWorkerNotifications,
  markNotificationAsRead
} = require('../controllers/fieldWorkerNotificationController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const router = express.Router();

// Protected field worker notification routes
router.get('/notifications', protectFieldWorker, getFieldWorkerNotifications);
router.patch('/notifications/:notificationId/read', protectFieldWorker, markNotificationAsRead);

module.exports = router;
