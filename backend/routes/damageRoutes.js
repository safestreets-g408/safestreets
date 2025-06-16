const express = require('express');
const { 
  uploadDamageReport, 
  getDamageHistory, 
  getReports,
  getReportById,
  getReportImage,
  createFromAiReport,
  createAndAssignFromAiReport,
  getGeneratedFromAiReports,
  assignRepair,
  unassignRepair,
  updateRepairStatus,
  updateReport,
  deleteReport,
  upload 
} = require('../controllers/damageController');
const { protectAdmin: protect } = require('../middleware/adminAuthMiddleware');
const router = express.Router();

// Protected routes
router.post('/upload', protect, upload.single('image'), uploadDamageReport);
router.get('/history', protect, getDamageHistory);
router.get('/reports', protect, getReports);
router.get('/report/:reportId', protect, getReportById);
router.get('/report/:reportId/image/:type', protect, getReportImage);

// New routes for AI report integration and repair management
router.post('/reports/create-from-ai', protect, createFromAiReport);
router.post('/reports/create-and-assign', protect, createAndAssignFromAiReport);
router.get('/reports/generated-from-ai', protect, getGeneratedFromAiReports);
router.patch('/reports/:reportId/assign', protect, assignRepair);
router.patch('/reports/:reportId/unassign', protect, unassignRepair);
router.patch('/reports/:reportId/status', protect, updateRepairStatus);
router.put('/report/:reportId', protect, updateReport);
router.delete('/report/:reportId', protect, deleteReport);

module.exports = router;
