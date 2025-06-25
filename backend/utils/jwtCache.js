const { cacheData, getCachedData, deleteCachedData } = require('./redisClient');

const JWT_TOKEN_PREFIX = 'jwt:';
const JWT_BLACKLIST_PREFIX = 'blacklist:jwt:';
const DEFAULT_TOKEN_EXPIRY = 24 * 60 * 60; // 24 hours in seconds

const cacheUserToken = async (userId, token, expirySeconds = DEFAULT_TOKEN_EXPIRY) => {
  try {
    // Store token with user ID for quick lookup
    const tokenKey = `${JWT_TOKEN_PREFIX}${token}`;
    await cacheData(tokenKey, { userId, valid: true }, expirySeconds);
    
    // Also store in reverse lookup by user ID
    const userKey = `${JWT_TOKEN_PREFIX}user:${userId}`;
    const existingTokens = await getCachedData(userKey) || [];
    
    if (!existingTokens.includes(token)) {
      existingTokens.push(token);
      await cacheData(userKey, existingTokens, expirySeconds);
    }
    
    return true;
  } catch (error) {
    console.error('Error caching JWT token:', error);
    return false;
  }
};

const getTokenFromCache = async (token) => {
  try {
    console.log('Validating token:', token); // Add logging

    // Check if token is blacklisted
    const isBlacklisted = await getCachedData(`${JWT_BLACKLIST_PREFIX}${token}`);
    if (isBlacklisted) {
      console.log('Token is blacklisted:', token); // Add logging
      return null;
    }

    // Check if token exists in cache
    const tokenData = await getCachedData(`${JWT_TOKEN_PREFIX}${token}`);

    return tokenData && tokenData.valid ? tokenData : null;
  } catch (error) {
    console.error('Error retrieving cached JWT token:', error);
    return null;
  }
};

const invalidateUserTokens = async (userId) => {
  try {
    // Get all tokens for this user
    const userKey = `${JWT_TOKEN_PREFIX}user:${userId}`;
    const tokens = await getCachedData(userKey) || [];
    
    // Blacklist each token
    for (const token of tokens) {
      await cacheData(`${JWT_BLACKLIST_PREFIX}${token}`, true, DEFAULT_TOKEN_EXPIRY);
      await deleteCachedData(`${JWT_TOKEN_PREFIX}${token}`);
    }
    
    // Clear the user's token list
    await deleteCachedData(userKey);
    
    return true;
  } catch (error) {
    console.error('Error invalidating user tokens:', error);
    return false;
  }
};

const invalidateToken = async (token) => {
  try {
    const tokenKey = `${JWT_TOKEN_PREFIX}${token}`;
    const tokenData = await getCachedData(tokenKey);
    
    if (tokenData && tokenData.userId) {
      // Blacklist this specific token
      await cacheData(`${JWT_BLACKLIST_PREFIX}${token}`, true, DEFAULT_TOKEN_EXPIRY);
      
      // Remove from user's token list
      const userKey = `${JWT_TOKEN_PREFIX}user:${tokenData.userId}`;
      const tokens = await getCachedData(userKey) || [];
      const updatedTokens = tokens.filter(t => t !== token);
      
      if (updatedTokens.length > 0) {
        await cacheData(userKey, updatedTokens, DEFAULT_TOKEN_EXPIRY);
      } else {
        await deleteCachedData(userKey);
      }
    }
    
    // Delete the token from cache
    await deleteCachedData(tokenKey);
    
    return true;
  } catch (error) {
    console.error('Error invalidating token:', error);
    return false;
  }
};

module.exports = {
  cacheUserToken,
  getTokenFromCache,
  invalidateUserTokens,
  invalidateToken
};
