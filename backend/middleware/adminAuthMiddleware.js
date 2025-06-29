const jwt = require('jsonwebtoken');
const Admin = require('../models/Admin');
const { getTokenFromCache } = require('../utils/jwtCache');

// Protect admin routes
const protectAdmin = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    console.log('Auth middleware - Token received:', !!token);
    console.log('Auth middleware - JWT_SECRET exists:', !!process.env.JWT_SECRET);
    
    if (!token) {
      console.log('Auth middleware - No token provided');
      return res.status(401).json({ message: 'No token, authorization denied' });
    }
    
    if (!process.env.JWT_SECRET) {
      console.error('Auth middleware - JWT_SECRET not configured');
      return res.status(500).json({ message: 'Server configuration error' });
    }
    
    // Try to get from cache first, but fallback to direct JWT verification
    let decoded;
    let cachedToken;
    
    try {
      cachedToken = await getTokenFromCache(token);
    } catch (cacheError) {
      console.warn('Redis cache error, falling back to JWT verification:', cacheError.message);
    }
    
    if (cachedToken && cachedToken.userId) {
      // Use the cached user ID directly
      decoded = { adminId: cachedToken.userId };
      console.log('Auth middleware - Using cached token for user:', cachedToken.userId);
    } else {
      // If not in cache or cache failed, verify JWT signature and decode
      try {
        decoded = jwt.verify(token, process.env.JWT_SECRET);
        console.log('Auth middleware - JWT decoded successfully for admin:', decoded.adminId);
      } catch (jwtError) {
        console.error('JWT verification failed:', jwtError.message);
        return res.status(401).json({ message: 'Invalid token', details: jwtError.message });
      }
    }
    
    // Check if admin still exists
    const admin = await Admin.findById(decoded.adminId).populate('tenant');
    if (!admin) {
      console.log('Admin not found for ID:', decoded.adminId);
      return res.status(401).json({ message: 'Not authorized. Admin no longer exists.' });
    }

    console.log('Admin authenticated:', admin.name, 'Role:', admin.role, 'Tenant:', admin.tenant?.name);
    
    // Check if tenant is active (except for super-admin)
    if (admin.role !== 'super-admin' && admin.tenant && !admin.tenant.active) {
      return res.status(401).json({ message: 'Tenant account is inactive. Please contact the system administrator.' });
    }
    
    // Add admin info to request
    req.admin = admin;
    next();
  } catch (err) {
    console.error('Auth middleware error:', err);
    res.status(401).json({ message: 'Authentication failed', error: err.message });
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
