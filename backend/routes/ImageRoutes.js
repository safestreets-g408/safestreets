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
const { protectAdmin, ensureTenantIsolation } = require('../middleware/adminAuthMiddleware');

// Test endpoint
router.get('/test-ai-server', protectAdmin, testAiServer);

// Protected routes with tenant isolation
router.post('/upload', protectAdmin, ensureTenantIsolation(), uploadImage);
router.get('/email/:email', protectAdmin, ensureTenantIsolation(), getImage);
router.get('/id/:imageId', protectAdmin, ensureTenantIsolation(), getImageById);

// Report routes with tenant isolation
router.get('/reports', protectAdmin, ensureTenantIsolation(), getReports);
router.get('/reports/:reportId', protectAdmin, ensureTenantIsolation(), getReportById);

module.exports = router;