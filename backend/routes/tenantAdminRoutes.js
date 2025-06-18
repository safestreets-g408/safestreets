const express = require('express');
const { protectAdmin, restrictTo } = require('../middleware/adminAuthMiddleware');
const Admin = require('../models/Admin');
const FieldWorker = require('../models/FieldWorker');
const DamageReport = require('../models/DamageReport');

const router = express.Router();

// Protect all routes
router.use(protectAdmin);

// Get tenant admins
router.get('/tenants/:tenantId/admins', restrictTo('super-admin', 'tenant-owner'), async (req, res) => {
  try {
    const { tenantId } = req.params;
    
    // For super-admin, allow access to any tenant
    // For tenant-owner, restrict to their own tenant
    if (req.admin.role === 'tenant-owner' && req.admin.tenant.toString() !== tenantId) {
      return res.status(403).json({ message: 'Not authorized to access this tenant' });
    }
    
    const admins = await Admin.find({ tenant: tenantId })
      .select('-password')
      .sort({ createdAt: -1 });
    
    res.status(200).json(admins);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Get tenant field workers
router.get('/tenants/:tenantId/field-workers', restrictTo('super-admin', 'tenant-owner', 'admin'), async (req, res) => {
  try {
    const { tenantId } = req.params;
    
    // Enforce tenant isolation for non-super-admins
    if (req.admin.role !== 'super-admin') {
      if (!req.admin.tenant || req.admin.tenant.toString() !== tenantId) {
        return res.status(403).json({ message: 'Not authorized to access this tenant' });
      }
    }
    
    const workers = await FieldWorker.find({ tenant: tenantId })
      .select('-password')
      .sort({ createdAt: -1 });
    
    res.status(200).json(workers);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Get tenant reports
router.get('/tenants/:tenantId/reports', restrictTo('super-admin', 'tenant-owner', 'admin'), async (req, res) => {
  try {
    const { tenantId } = req.params;
    
    // Enforce tenant isolation for non-super-admins
    if (req.admin.role !== 'super-admin') {
      if (!req.admin.tenant || req.admin.tenant.toString() !== tenantId) {
        return res.status(403).json({ message: 'Not authorized to access this tenant' });
      }
    }
    
    const reports = await DamageReport.find({ tenant: tenantId })
      .sort({ createdAt: -1 })
      .limit(50);
    
    res.status(200).json(reports);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Create admin user within a tenant
router.post('/tenants/:tenantId/admins', restrictTo('super-admin', 'tenant-owner'), async (req, res) => {
  try {
    const { tenantId } = req.params;
    const { name, email, password } = req.body;
    
    // For tenant-owner, restrict to their own tenant
    if (req.admin.role === 'tenant-owner' && req.admin.tenant.toString() !== tenantId) {
      return res.status(403).json({ message: 'Not authorized to add admins to this tenant' });
    }
    
    // Check if admin email already exists
    const existingAdmin = await Admin.findOne({ email });
    if (existingAdmin) {
      return res.status(400).json({ message: 'Admin with this email already exists' });
    }
    
    const admin = await Admin.create({
      name,
      email,
      password,
      role: 'admin', // Only allow creating regular admins, not tenant owners
      tenant: tenantId
    });
    
    // Return admin data (excluding password)
    const adminData = {
      _id: admin._id,
      name: admin.name,
      email: admin.email,
      role: admin.role
    };
    
    res.status(201).json(adminData);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Create field worker within a tenant
router.post('/tenants/:tenantId/field-workers', restrictTo('super-admin', 'tenant-owner', 'admin'), async (req, res) => {
  try {
    const { tenantId } = req.params;
    const { name, workerId, specialization, region } = req.body;
    
    // Enforce tenant isolation for non-super-admins
    if (req.admin.role !== 'super-admin') {
      if (!req.admin.tenant || req.admin.tenant.toString() !== tenantId) {
        return res.status(403).json({ message: 'Not authorized to add field workers to this tenant' });
      }
    }
    
    // Check if worker ID already exists
    const existingWorker = await FieldWorker.findOne({ workerId });
    if (existingWorker) {
      return res.status(400).json({ message: 'Field worker with this ID already exists' });
    }
    
    // Generate email and default password
    const email = `${workerId.toLowerCase()}@safestreets.com`;
    const generatedPassword = `${workerId}${Math.floor(1000 + Math.random() * 9000)}`;
    
    const fieldWorker = await FieldWorker.create({
      name,
      workerId,
      email,
      password: generatedPassword, // This will be hashed by pre-save hook
      specialization,
      region,
      tenant: tenantId
    });
    
    // Return field worker data (excluding password)
    const workerData = {
      _id: fieldWorker._id,
      name: fieldWorker.name,
      workerId: fieldWorker.workerId,
      email: fieldWorker.email,
      specialization: fieldWorker.specialization,
      region: fieldWorker.region,
      generatedPassword // Include this only for initial setup
    };
    
    res.status(201).json(workerData);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

module.exports = router;
