const express = require('express');
const router = express.Router();
const { sendTenantCredentials, subscribeToNewsletter } = require('../controllers/emailController');
const { protectAdmin, restrictTo } = require('../middleware/adminAuthMiddleware');

// Only super-admin can send tenant credentials
router.post('/send-tenant-credentials', protectAdmin, restrictTo('super-admin'), sendTenantCredentials);

// Public route for newsletter subscription
router.post('/subscribe-newsletter', subscribeToNewsletter);

module.exports = router;
