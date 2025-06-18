const DamageReport = require('../models/DamageReport');

// Middleware to enforce tenant isolation for damage reports
const enforceDamageTenantIsolation = async (req, res, next) => {
  // Skip for super admins
  if (req.admin && req.admin.role === 'super-admin') {
    return next();
  }
  
  // If we have a tenantId from the ensureTenantIsolation middleware, use it
  if (req.tenantId) {
    req.query.tenant = req.tenantId;
    return next();
  }
  
  // If we have a fieldWorker, use their tenant
  if (req.fieldWorker && req.fieldWorker.tenant) {
    req.query.tenant = req.fieldWorker.tenant;
    return next();
  }
  
  // No tenant, reject request
  return res.status(403).json({ message: 'Not authorized to access this data' });
};

// Helper to check if an object belongs to a tenant
const verifyTenantOwnership = async (model, id, tenantId) => {
  const document = await model.findById(id);
  
  if (!document) {
    return { error: 'Resource not found', status: 404 };
  }
  
  // Check if document belongs to tenant
  if (document.tenant && document.tenant.toString() !== tenantId.toString()) {
    return { error: 'Not authorized to access this resource', status: 403 };
  }
  
  return { document, status: 200 };
};

module.exports = {
  enforceDamageTenantIsolation,
  verifyTenantOwnership
};
