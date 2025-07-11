const express = require('express');
const { 
  getFieldWorkerNotifications,
  markNotificationAsRead,
  registerDeviceToken,
  sendTestNotification
} = require('../controllers/fieldWorkerNotificationController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const router = express.Router();

// Protected field worker notification routes
router.get('/notifications', protectFieldWorker, getFieldWorkerNotifications);
router.patch('/notifications/:notificationId/read', protectFieldWorker, markNotificationAsRead);

// Device registration and push notifications
router.post('/register-device', protectFieldWorker, registerDeviceToken);
router.post('/test-notification', protectFieldWorker, sendTestNotification);

module.exports = router;
