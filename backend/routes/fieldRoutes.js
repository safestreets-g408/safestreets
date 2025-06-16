const express = require('express');
const router = express.Router();
const { 
    addFieldWorker, 
    getFieldWorkers, 
    updateFieldWorker,
    getFieldWorkerById 
} = require('../controllers/fieldWorkerController');
const { protectAdmin: protect } = require('../middleware/adminAuthMiddleware');

// Protected routes
router.post('/add', protect, addFieldWorker);
router.get('/workers', protect, getFieldWorkers); 
router.get('/', protect, getFieldWorkers); 
router.get('/:workerId', protect, getFieldWorkerById);
router.put('/:workerId', protect, updateFieldWorker);

module.exports = router;