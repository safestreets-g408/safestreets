

require('dotenv').config();

module.exports = {
  // Server configuration
  server: {
    port: process.env.PORT || 3001,
    env: process.env.NODE_ENV || 'development',
  },
  
  // Database configuration
  database: {
    url: process.env.MONGODB_URI || 'mongodb://localhost:27017/safestreets',
  },
  
  // JWT configuration
  jwt: {
    secret: process.env.JWT_SECRET || 'safestreets-secret-key',
    expiresIn: process.env.JWT_EXPIRES_IN || '7d',
  },
  
  // Storage configuration for images
  storage: {
    uploadDir: process.env.UPLOAD_DIR || 'uploads',
    imageResultsDir: process.env.IMAGE_RESULTS_DIR || 'results',
  },
  
  // AI Models server configuration
  aiModels: {
    url: process.env.AI_MODELS_SERVER_URL || 'http://localhost:5000',
    timeout: process.env.AI_REQUEST_TIMEOUT || 30000, // 30 seconds
  },
  
  // Weather API configuration
  weather: {
    apiKey: process.env.WEATHER_API_KEY || '',
    baseUrl: process.env.WEATHER_API_URL || 'https://api.weatherapi.com/v1',
  },
  
  // Cache configuration
  cache: {
    ttl: process.env.CACHE_TTL || 3600, // 1 hour in seconds
  },
  
  // Gemini AI configuration
  gemini: {
    apiKey: process.env.GEMINI_API_KEY || '',
    defaultModel: process.env.GEMINI_DEFAULT_MODEL || 'gemini-1.5-flash',
    proModel: process.env.GEMINI_PRO_MODEL || 'gemini-1.5-pro',
  }
};
