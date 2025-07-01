const express = require('express');
const { generateDamageSummary } = require('../controllers/damageController');
const { getModelInfo } = require('../controllers/aiModelController');
const router = express.Router();

// Public AI routes for testing - no authentication required
router.post('/generate-summary', generateDamageSummary); 

// Model information endpoint
router.get('/model-info', getModelInfo);

// AI Server health check
router.get('/health-check', async (req, res) => {
  try {
    // Try to connect to the AI server
    const axios = require('axios');
    const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:5000';
    
    const response = await axios.get(`${AI_SERVER_URL}/health`, {
      timeout: 5000 // 5 second timeout
    });
    
    if (response.status === 200 && response.data.status === 'healthy') {
      return res.status(200).json({
        success: true,
        message: 'AI server is running',
        aiServerResponse: response.data
      });
    } else {
      return res.status(200).json({
        success: false,
        message: 'AI server responded but may have issues',
        aiServerResponse: response.data
      });
    }
  } catch (error) {
    console.error('AI server health check failed:', error.message);
    return res.status(200).json({
      success: false,
      message: 'Could not connect to AI server',
      error: error.message
    });
  }
});

module.exports = router;
