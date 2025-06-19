const axios = require('axios');

// URL for the AI server
const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:5000';

const generateDamageSummary = async (damageDetails) => {
  try {
    const { location, damageType, severity, priority } = damageDetails;
    
    // Validate required parameters
    if (!location || !damageType || !severity || !priority) {
      throw new Error('Missing required damage details');
    }

    // Call the AI server endpoint
    const response = await axios.post(`${AI_SERVER_URL}/generate-summary`, {
      location,
      damageType,
      severity,
      priority
    });

    // Return the generated summary
    if (response.data && response.data.success && response.data.summary) {
      return response.data.summary;
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
