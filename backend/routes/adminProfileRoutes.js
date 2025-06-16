const express = require('express');
const { getAdminProfile, updateAdminProfile } = require('../controllers/adminProfileController');
const { protectAdmin } = require('../middleware/adminAuthMiddleware');
const router = express.Router();

// Routes for /api/admin/profile
router.get('/', protectAdmin, getAdminProfile);
router.put('/', protectAdmin, updateAdminProfile);

module.exports = router;
