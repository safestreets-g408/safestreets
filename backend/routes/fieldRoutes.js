const express = require('express');
const router = express.Router();
const { 
    addFieldWorker, 
    getFieldWorkers, 
    updateFieldWorker,
    getFieldWorkerById,
    getFieldWorkerAssignments,
    sendDailyUpdates
} = require('../controllers/fieldWorkerController');
const { protectAdmin, ensureTenantIsolation } = require('../middleware/adminAuthMiddleware');

// Protected routes with tenant isolation
router.post('/add', protectAdmin, ensureTenantIsolation(), addFieldWorker);
router.get('/workers', protectAdmin, ensureTenantIsolation(), getFieldWorkers); 
router.get('/', protectAdmin, ensureTenantIsolation(), getFieldWorkers); 
router.get('/:workerId', protectAdmin, ensureTenantIsolation(), getFieldWorkerById);
router.put('/:workerId', protectAdmin, ensureTenantIsolation(), updateFieldWorker);

// Route to get assignments for a specific field worker
router.get('/:workerId/assignments', protectAdmin, ensureTenantIsolation(), getFieldWorkerAssignments);

// Route for sending daily updates to field workers
router.post('/daily-updates', protectAdmin, ensureTenantIsolation(), sendDailyUpdates);
// Route for sending test email to a specific field worker
router.post('/daily-updates/:workerId', protectAdmin, ensureTenantIsolation(), sendDailyUpdates);

module.exports = router;