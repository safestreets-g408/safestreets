const express = require('express');
const router = express.Router();
const { sendTenantCredentials } = require('../controllers/emailController');
const { protectAdmin, restrictTo } = require('../middleware/adminAuthMiddleware');

// Only super-admin can send tenant credentials
router.post('/send-tenant-credentials', protectAdmin, restrictTo('super-admin'), sendTenantCredentials);

module.exports = router;
