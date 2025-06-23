const { getCachedData, cacheData } = require('../utils/redisClient');

const cacheMiddleware = (duration = 3600, keyGenerator = null) => {
  return async (req, res, next) => {
    // Skip caching for non-GET requests
    if (req.method !== 'GET') {
      return next();
    }
    
    // Generate cache key
    const key = keyGenerator 
      ? keyGenerator(req) 
      : `cache:${req.originalUrl || req.url}${JSON.stringify(req.query)}`;
    
    // Set tenant ID in key if available (for multi-tenant isolation)
    const tenantKey = req.tenantId 
      ? `tenant:${req.tenantId}:${key}` 
      : key;

    try {
      // Try to get data from cache
      const cachedData = await getCachedData(tenantKey);
      
      if (cachedData) {
        console.log(`🚀 Cache hit: ${tenantKey}`);
        return res.status(200).json(cachedData);
      }
      
      // Cache miss - modify res.json to capture and cache the response
      const originalJson = res.json;
      res.json = function(data) {
        // Only cache successful responses
        if (res.statusCode >= 200 && res.statusCode < 300) {
          cacheData(tenantKey, data, duration).catch(err => 
            console.error(`Failed to cache data for ${tenantKey}:`, err)
          );
        }
        return originalJson.call(this, data);
      };
      
      next();
    } catch (error) {
      console.error('Cache middleware error:', error);
      next(); // Continue without caching
    }
  };
};

module.exports = cacheMiddleware;
