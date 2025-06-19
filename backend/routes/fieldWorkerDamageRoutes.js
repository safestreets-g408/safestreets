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
router.post('/reports/upload', protectFieldWorker, uploadDamageReportByFieldWorker);
router.get('/nearby', protectFieldWorker, getNearbyReports);

module.exports = router;
