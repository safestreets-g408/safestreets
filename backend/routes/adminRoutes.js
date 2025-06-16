const express = require('express');
const { registerAdmin, loginAdmin } = require('../controllers/adminAuthController');
const router = express.Router();

router.post('/register', registerAdmin);
router.post('/login', loginAdmin);

module.exports = router;
