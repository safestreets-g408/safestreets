const express = require('express');
const { generateDamageSummary } = require('../controllers/damageController');
const router = express.Router();

// Public AI routes for testing - no authentication required
router.post('/generate-summary', generateDamageSummary); 

module.exports = router;
