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
const { analyzeWithYolo } = require('../controllers/aiModelController');
const { protectAdmin, ensureTenantIsolation } = require('../middleware/adminAuthMiddleware');
const upload = require('../middleware/multerConfig');

// Test endpoint
router.get('/test-ai-server', protectAdmin, testAiServer);

// Protected routes with tenant isolation
router.post('/upload', protectAdmin, ensureTenantIsolation(), upload.single('image'), uploadImage);
router.post('/analyze-yolo', protectAdmin, ensureTenantIsolation(), upload.single('image'), analyzeWithYolo);
router.get('/email/:email', protectAdmin, ensureTenantIsolation(), getImage);
router.get('/id/:imageId', protectAdmin, ensureTenantIsolation(), getImageById);

// Report routes with tenant isolation
router.get('/reports', protectAdmin, ensureTenantIsolation(), getReports);
router.get('/reports/:reportId', protectAdmin, ensureTenantIsolation(), getReportById);

module.exports = router;