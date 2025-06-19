const express = require('express');
const { getWeatherInfo } = require('../controllers/weatherController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const router = express.Router();

// Protected weather routes
router.get('/weather', protectFieldWorker, getWeatherInfo);

module.exports = router;
