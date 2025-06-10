const express = require('express');
const { 
  uploadDamageReport, 
  getDamageHistory, 
  getReports,
  getReportById,
  getReportImage,
  upload 
} = require('../controllers/damageController');
const protect = require('../middleware/authMiddleware');
const router = express.Router();

// Protected routes
router.post('/upload', protect, upload.single('image'), uploadDamageReport);
router.get('/history', protect, getDamageHistory);
router.get('/reports', protect, getReports);
router.get('/report/:reportId', protect, getReportById);
router.get('/report/:reportId/image/:type', protect, getReportImage);

module.exports = router;
