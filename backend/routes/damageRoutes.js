const express = require('express');
const { uploadDamageReport, getDamageHistory, upload, getReports } = require('../controllers/damageController');
const protect = require('../middleware/authMiddleware');
const router = express.Router();

// Protected routes
router.post('/upload', protect, upload.single('image'), uploadDamageReport);
router.get('/history', protect, getDamageHistory);
router.get('/reports',getReports)

module.exports = router;
