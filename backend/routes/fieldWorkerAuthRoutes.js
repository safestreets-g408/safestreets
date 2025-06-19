const express = require('express');
const router = express.Router();
const { 
  registerFieldWorker,
  loginFieldWorker,
  getFieldWorkerProfile,
  updateFieldWorkerProfile,
  refreshToken
} = require('../controllers/fieldWorkerAuthController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');

// Public routes
router.post('/register', registerFieldWorker);
router.post('/login', loginFieldWorker);

// Protected routes
router.get('/profile', protectFieldWorker, getFieldWorkerProfile);
router.put('/profile', protectFieldWorker, updateFieldWorkerProfile);
router.post('/refresh-token', protectFieldWorker, refreshToken);

module.exports = router;
