const express = require('express');
const { 
  getFieldWorkerReports,
  updateRepairStatus,
  getFieldWorkerDashboard,
  uploadDamageReportByFieldWorker
} = require('../controllers/fieldWorkerDamageController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const router = express.Router();

// Protected field worker routes
router.get('/dashboard', protectFieldWorker, getFieldWorkerDashboard);
router.get('/reports', protectFieldWorker, getFieldWorkerReports);
router.patch('/reports/:reportId/status', protectFieldWorker, updateRepairStatus);
router.post('/reports/upload', protectFieldWorker, uploadDamageReportByFieldWorker);

module.exports = router;
