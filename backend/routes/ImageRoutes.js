const express = require('express');
const router = express.Router();
const { 
    uploadImage, 
    getImage, 
    getImageById, 
    getReports,
    getReportById,
    testAiServer 
} = require('../controllers/ImageController');
const { protectAdmin: protect } = require('../middleware/adminAuthMiddleware');

// Test endpoint
router.get('/test-ai-server', protect, testAiServer);

// Protected routes - removed multer middleware since we're using base64
router.post('/upload', protect, uploadImage);
router.get('/email/:email', protect, getImage);
router.get('/id/:imageId', protect, getImageById);

// Report routes
router.get('/reports', protect, getReports);
router.get('/reports/:reportId', protect, getReportById);

module.exports = router;