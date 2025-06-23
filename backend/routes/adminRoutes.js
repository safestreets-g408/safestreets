const express = require('express');
const { registerAdmin, loginAdmin, logoutAdmin } = require('../controllers/adminAuthController');
const { protectAdmin } = require('../middleware/adminAuthMiddleware');
const router = express.Router();

router.post('/register', registerAdmin);
router.post('/login', loginAdmin);
router.post('/logout', protectAdmin, logoutAdmin);

module.exports = router;
