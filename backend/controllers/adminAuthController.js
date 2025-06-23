const Admin = require('../models/Admin');
const Tenant = require('../models/Tenant');
const jwt = require('jsonwebtoken');
const { cacheUserToken, invalidateToken } = require('../utils/jwtCache');

// Register Admin
const registerAdmin = async (req, res) => {
  const { name, email, password, role, tenantId } = req.body;
  
  try {
    // Check if admin exists
    const adminExists = await Admin.findOne({ email });
    if (adminExists) {
      return res.status(400).json({ message: 'Admin account already exists with this email' });
    }
    
    // Validate inputs based on role
    if (role !== 'super-admin' && !tenantId) {
      return res.status(400).json({ message: 'Tenant ID is required for non-super-admin users' });
    }
    
    // If tenantId is provided, check if tenant exists and is active
    if (tenantId) {
      const tenant = await Tenant.findById(tenantId);
      if (!tenant) {
        return res.status(404).json({ message: 'Tenant not found' });
      }
      
      if (!tenant.active) {
        return res.status(400).json({ message: 'Cannot create admin for inactive tenant' });
      }
      
      // Check if max admins limit is reached
      const adminCount = await Admin.countDocuments({ tenant: tenantId });
      if (adminCount >= tenant.settings.maxAdmins) {
        return res.status(400).json({ message: `Maximum number of admins (${tenant.settings.maxAdmins}) reached for this tenant` });
      }
    }
    
    // Create admin with or without tenant association
    const adminData = {
      name,
      email,
      password,
      role: role || 'admin'
    };
    
    // Add tenant if not super-admin
    if (role !== 'super-admin' && tenantId) {
      adminData.tenant = tenantId;
    }
    
    const admin = await Admin.create(adminData);
    const token = jwt.sign({ adminId: admin._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    
    // Return admin data (excluding password)
    const responseData = {
      _id: admin._id,
      name: admin.name,
      email: admin.email,
      role: admin.role,
      profile: admin.profile || {}
    };
    
    // Add tenant info if exists
    if (admin.tenant) {
      responseData.tenant = admin.tenant;
    }
    
    res.status(201).json({ token, admin: responseData });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Login Admin
const loginAdmin = async (req, res) => {
  const { email, password } = req.body;
  try {
    // Find admin with tenant data
    const admin = await Admin.findOne({ email }).populate('tenant');
    console.log('Login attempt for:', email);
    console.log('Admin found:', admin ? 'yes' : 'no');
    
    if (!admin) {
      return res.status(400).json({ message: 'Invalid credentials - admin not found' });
    }
    
    const isMatch = await admin.matchPassword(password);
    console.log('Password match result:', isMatch);
    
    if (!isMatch) {
      return res.status(400).json({ message: 'Invalid credentials - password mismatch' });
    }
    
    // Check if tenant is active for non-super-admin users
    if (admin.role !== 'super-admin' && admin.tenant && !admin.tenant.active) {
      return res.status(403).json({ message: 'Your tenant account is inactive. Please contact the system administrator.' });
    }

    // Update last active timestamp
    admin.profile = admin.profile || {};
    admin.profile.lastActive = Date.now();
    await admin.save();

    const token = jwt.sign({ adminId: admin._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    
    // Cache token in Redis for faster validation
    await cacheUserToken(admin._id.toString(), token, 60 * 60); // 1 hour
    
    // Return admin data (excluding password)
    const adminData = {
      _id: admin._id,
      name: admin.name,
      email: admin.email,
      role: admin.role,
      profile: admin.profile || {}
    };
    
    // Add tenant info if exists
    if (admin.tenant) {
      adminData.tenant = {
        _id: admin.tenant._id,
        name: admin.tenant.name,
        code: admin.tenant.code,
        settings: admin.tenant.settings
      };
    }
    
    res.status(200).json({ token, admin: adminData });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Logout Admin
const logoutAdmin = async (req, res) => {
  try {
    // Get token from header
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(400).json({ message: 'No token provided' });
    }
    
    // Invalidate token in Redis
    await invalidateToken(token);
    
    res.status(200).json({ message: 'Logged out successfully' });
  } catch (error) {
    console.error('Logout error:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

module.exports = { registerAdmin, loginAdmin, logoutAdmin };
