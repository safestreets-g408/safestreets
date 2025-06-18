const express = require('express');
const router = express.Router();
const { 
    addFieldWorker, 
    getFieldWorkers, 
    updateFieldWorker,
    getFieldWorkerById 
} = require('../controllers/fieldWorkerController');
const { protectAdmin, ensureTenantIsolation } = require('../middleware/adminAuthMiddleware');

// Protected routes with tenant isolation
router.post('/add', protectAdmin, ensureTenantIsolation(), addFieldWorker);
router.get('/workers', protectAdmin, ensureTenantIsolation(), getFieldWorkers); 
router.get('/', protectAdmin, ensureTenantIsolation(), getFieldWorkers); 
router.get('/:workerId', protectAdmin, ensureTenantIsolation(), getFieldWorkerById);
router.put('/:workerId', protectAdmin, ensureTenantIsolation(), updateFieldWorker);

module.exports = router;