const express = require('express');
const { registerAdmin, loginAdmin, logoutAdmin, validateToken } = require('../controllers/adminAuthController');
const { protectAdmin } = require('../middleware/adminAuthMiddleware');
const router = express.Router();

router.post('/register', registerAdmin);
router.post('/login', loginAdmin);
router.post('/logout', protectAdmin, logoutAdmin);
router.get('/validate', protectAdmin, validateToken);

module.exports = router;
