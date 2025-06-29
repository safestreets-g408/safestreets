const express = require('express');
const { 
  getFieldWorkerReports,
  updateRepairStatus,
  getFieldWorkerDashboard,
  uploadDamageReportByFieldWorker,
  getFilteredReports,
  getTaskAnalytics,
  getWeeklyReportStats,
  getReportStatusSummary,
  getNearbyReports
} = require('../controllers/fieldWorkerDamageController');
const { getReportImage } = require('../controllers/damageController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const router = express.Router();

// Protected field worker routes

// Dashboard and analytics endpoints
router.get('/dashboard', protectFieldWorker, getFieldWorkerDashboard);
router.get('/task-analytics', protectFieldWorker, getTaskAnalytics);
router.get('/weekly-stats', protectFieldWorker, getWeeklyReportStats);
router.get('/status-summary', protectFieldWorker, getReportStatusSummary);

// Report management endpoints
router.get('/reports', protectFieldWorker, getFieldWorkerReports);
router.get('/reports/filtered', protectFieldWorker, getFilteredReports);
router.patch('/reports/:reportId/status', protectFieldWorker, updateRepairStatus);
router.post('/ai-reports/upload', protectFieldWorker, uploadDamageReportByFieldWorker);
router.post('/reports/upload', protectFieldWorker, uploadDamageReportByFieldWorker);
router.get('/nearby', protectFieldWorker, getNearbyReports);

// Image access route for field workers
router.get('/report/:reportId/image/:type', getReportImage);

module.exports = router;
