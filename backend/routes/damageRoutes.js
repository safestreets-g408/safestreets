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
const { protectAdmin, ensureTenantIsolation } = require('../middleware/adminAuthMiddleware');
const { enforceDamageTenantIsolation } = require('../middleware/tenantIsolationMiddleware');
const router = express.Router();

// Protected routes with tenant isolation
router.post('/upload', protectAdmin, ensureTenantIsolation(), upload.single('image'), uploadDamageReport);
router.get('/history', protectAdmin, ensureTenantIsolation(), enforceDamageTenantIsolation, getDamageHistory);
router.get('/reports', protectAdmin, ensureTenantIsolation(), enforceDamageTenantIsolation, getReports);
router.get('/report/:reportId', protectAdmin, ensureTenantIsolation(), enforceDamageTenantIsolation, getReportById);
// Allow image access with token in URL for <img> tag compatibility
router.get('/report/:reportId/image/:type', getReportImage); // Consider tenant isolation for this endpoint too

// New routes for AI report integration and repair management - adding tenant isolation
router.post('/reports/create-from-ai', protectAdmin, ensureTenantIsolation(), createFromAiReport);
router.post('/reports/create-and-assign', protectAdmin, ensureTenantIsolation(), createAndAssignFromAiReport);
router.get('/reports/generated-from-ai', protectAdmin, ensureTenantIsolation(), getGeneratedFromAiReports);
router.patch('/reports/:reportId/assign', protectAdmin, ensureTenantIsolation(), assignRepair);
router.patch('/reports/:reportId/unassign', protectAdmin, ensureTenantIsolation(), unassignRepair);
router.patch('/reports/:reportId/status', protectAdmin, ensureTenantIsolation(), updateRepairStatus);
router.put('/report/:reportId', protectAdmin, ensureTenantIsolation(), updateReport);
router.delete('/report/:reportId', protectAdmin, ensureTenantIsolation(), deleteReport);

module.exports = router;
