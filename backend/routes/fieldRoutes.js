const express = require('express');
const router = express.Router();
const { 
    addFieldWorker, 
    getFieldWorkers, 
    updateFieldWorker,
    getFieldWorkerById 
} = require('../controllers/fieldWorkerController');
const protect = require('../middleware/authMiddleware');

// Protected routes
router.post('/add', protect, addFieldWorker);
router.get('/', protect, getFieldWorkers);
router.get('/:workerId', protect, getFieldWorkerById);
router.put('/:workerId', protect, updateFieldWorker);

module.exports = router;