const express = require('express');
const { getWeatherInfo } = require('../controllers/weatherController');
const { protectFieldWorker } = require('../middleware/fieldWorkerAuthMiddleware');
const cacheMiddleware = require('../middleware/cacheMiddleware');
const router = express.Router();

// Protected weather routes with 30-minute cache
router.get('/weather', protectFieldWorker, cacheMiddleware(1800, req => `weather:${req.query.lat}:${req.query.lon}`), getWeatherInfo);

module.exports = router;
