const { clearCachePattern } = require('../utils/redisClient');

const clearDamageReportCaches = async (tenantId) => {
  try {
    // Clear all damage report related caches for the tenant
    if (tenantId) {
      // Clear tenant-specific caches
      await clearCachePattern(`tenant:${tenantId}:cache:*damage*`);
      await clearCachePattern(`tenant:${tenantId}:cache:*report*`);
    } else {
      // Clear all damage caches if no tenant specified
      await clearCachePattern(`*damage*`);
      await clearCachePattern(`*report*`);
    }
    return true;
  } catch (error) {
    console.error('Failed to clear damage report caches:', error);
    return false;
  }
};

const clearSingleReportCache = async (reportId, tenantId) => {
  try {
    // Clear cache for specific report
    if (tenantId) {
      await clearCachePattern(`tenant:${tenantId}:cache:*report/${reportId}*`);
    } else {
      await clearCachePattern(`*report/${reportId}*`);
    }
    return true;
  } catch (error) {
    console.error(`Failed to clear cache for report ${reportId}:`, error);
    return false;
  }
};

module.exports = {
  clearDamageReportCaches,
  clearSingleReportCache
};
