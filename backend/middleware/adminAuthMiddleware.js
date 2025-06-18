const jwt = require('jsonwebtoken');
const Admin = require('../models/Admin');

// Protect admin routes
const protectAdmin = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ message: 'No token, authorization denied' });
    }
    
    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Check if admin still exists
    const admin = await Admin.findById(decoded.adminId).populate('tenant');
    if (!admin) {
      return res.status(401).json({ message: 'Not authorized. Admin no longer exists.' });
    }

    // Check if tenant is active (except for super-admin)
    if (admin.role !== 'super-admin' && admin.tenant && !admin.tenant.active) {
      return res.status(401).json({ message: 'Tenant account is inactive. Please contact the system administrator.' });
    }
    
    // Add admin info to request
    req.admin = admin;
    next();
  } catch (err) {
    res.status(401).json({ message: 'Token is not valid', error: err.message });
  }
};

// Restrict routes to specific admin roles
const restrictTo = (...roles) => {
  return (req, res, next) => {
    if (!roles.includes(req.admin.role)) {
      return res.status(403).json({ message: 'Not authorized. Insufficient permissions.' });
    }
    next();
  };
};

// Ensure tenant data isolation
const ensureTenantIsolation = (allowSuperAdmin = true) => {
  return (req, res, next) => {
    // Allow super admin to access all data if specified
    if (allowSuperAdmin && req.admin.role === 'super-admin') {
      return next();
    }
    
    // Ensure the admin has a tenant assigned
    if (!req.admin.tenant) {
      return res.status(403).json({ message: 'No tenant associated with this admin account.' });
    }
    
    // Add tenantId filter to all queries
    req.tenantId = req.admin.tenant._id;
    next();
  };
};

module.exports = { protectAdmin, restrictTo, ensureTenantIsolation };
