const axios = require('axios');
const crypto = require('crypto');

// URL for the AI server
const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:5000';

// Simple in-memory cache for summaries
const summaryCache = {};
const CACHE_TTL = 3600000; // 1 hour in milliseconds

// Helper to generate a cache key
const generateCacheKey = (details) => {
  const str = `${details.location}-${details.damageType}-${details.severity}-${details.priority}`;
  return crypto.createHash('md5').update(str).digest('hex');
};

const generateDamageSummary = async (damageDetails) => {
  try {
    const { location, damageType, severity, priority } = damageDetails;
    
    // Validate required parameters
    if (!location || !damageType || !severity || !priority) {
      throw new Error('Missing required damage details');
    }

    // Check if we have a cached summary
    const cacheKey = generateCacheKey(damageDetails);
    const now = Date.now();
    
    if (summaryCache[cacheKey] && summaryCache[cacheKey].expiry > now) {
      console.log('Using cached summary');
      return summaryCache[cacheKey].summary;
    }
    
    console.log('Generating new summary from AI server');
    const startTime = Date.now();

    // Call the AI server endpoint with increased timeout
    const response = await axios.post(`${AI_SERVER_URL}/generate-summary`, {
      location,
      damageType,
      severity,
      priority
    }, {
      timeout: 15000 // 15 second timeout
    });

    console.log(`Summary generated in ${(Date.now() - startTime)/1000}s`);

    // Return the generated summary
    if (response.data && response.data.success && response.data.summary) {
      // Use formatted_summary if available, otherwise fall back to summary
      const formattedSummary = response.data.formatted_summary || response.data.summary;
      
      // Cache the result
      summaryCache[cacheKey] = {
        summary: formattedSummary,
        summaryType: response.data.summary_type || 'standard',
        expiry: now + CACHE_TTL
      };
      
      // Clean up old cache entries
      Object.keys(summaryCache).forEach(key => {
        if (summaryCache[key].expiry < now) {
          delete summaryCache[key];
        }
      });
      
      console.log(`Received ${response.data.summary_type || 'standard'} summary from AI server`);
      return formattedSummary;
    } else {
      throw new Error(response.data?.message || 'Failed to generate summary');
    }
  } catch (error) {
    console.error('Error generating damage summary:', error);
    // Return a basic fallback summary
    return `Road damage at ${damageDetails.location}: ${damageDetails.damageType} with ${damageDetails.severity} severity. Priority level: ${damageDetails.priority}.`;
  }
};

module.exports = {
  generateDamageSummary
};
