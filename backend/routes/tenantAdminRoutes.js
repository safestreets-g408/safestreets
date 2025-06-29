const express = require('express');
const { protectAdmin, restrictTo } = require('../middleware/adminAuthMiddleware');
const Admin = require('../models/Admin');
const FieldWorker = require('../models/FieldWorker');
const DamageReport = require('../models/DamageReport');

const router = express.Router();

// Protect all routes
router.use(protectAdmin);

// Get all admins across all tenants (super-admin only)
router.get('/all', restrictTo('super-admin'), async (req, res) => {
  try {
    const admins = await Admin.find()
      .populate('tenant', 'name')
      .select('-password')
      .sort({ createdAt: -1 });
    
    res.status(200).json(admins);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Get all field workers across all tenants (super-admin only)
router.get('/field-workers/all', restrictTo('super-admin'), async (req, res) => {
  try {
    const fieldWorkers = await FieldWorker.find()
      .populate('tenant', 'name')
      .select('-password')
      .sort({ createdAt: -1 });
    
    res.status(200).json(fieldWorkers);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

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

// Update admin user
router.put('/tenants/:tenantId/admins/:adminId', restrictTo('super-admin', 'tenant-owner'), async (req, res) => {
  try {
    const { tenantId, adminId } = req.params;
    const { name, email, password } = req.body;
    
    // For tenant-owner, restrict to their own tenant
    if (req.admin.role === 'tenant-owner' && req.admin.tenant.toString() !== tenantId) {
      return res.status(403).json({ message: 'Not authorized to update admins in this tenant' });
    }
    
    // Find the admin to update
    const admin = await Admin.findOne({ _id: adminId, tenant: tenantId });
    
    if (!admin) {
      return res.status(404).json({ message: 'Admin not found' });
    }
    
    // Don't allow updating a tenant-owner's account through this endpoint
    if (admin.role === 'tenant-owner') {
      return res.status(403).json({ message: 'Cannot modify tenant owner account' });
    }
    
    // Update fields
    if (name) admin.name = name;
    if (email) admin.email = email;
    if (password) admin.password = password;
    
    await admin.save();
    
    // Return updated admin data (excluding password)
    const adminData = {
      _id: admin._id,
      name: admin.name,
      email: admin.email,
      role: admin.role,
      tenant: admin.tenant
    };
    
    res.status(200).json(adminData);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Delete admin user
router.delete('/tenants/:tenantId/admins/:adminId', restrictTo('super-admin', 'tenant-owner'), async (req, res) => {
  try {
    const { tenantId, adminId } = req.params;
    
    // For tenant-owner, restrict to their own tenant
    if (req.admin.role === 'tenant-owner' && req.admin.tenant.toString() !== tenantId) {
      return res.status(403).json({ message: 'Not authorized to delete admins from this tenant' });
    }
    
    // Find the admin to delete
    const admin = await Admin.findOne({ _id: adminId, tenant: tenantId });
    
    if (!admin) {
      return res.status(404).json({ message: 'Admin not found' });
    }
    
    // Don't allow deleting a tenant-owner
    if (admin.role === 'tenant-owner') {
      return res.status(403).json({ message: 'Cannot delete tenant owner account' });
    }
    
    // Don't allow self-deletion
    if (admin._id.toString() === req.admin._id.toString()) {
      return res.status(400).json({ message: 'Cannot delete your own account' });
    }
    
    await Admin.deleteOne({ _id: adminId });
    
    res.status(200).json({ message: 'Admin deleted successfully' });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Update field worker
router.put('/tenants/:tenantId/field-workers/:workerId', restrictTo('super-admin', 'tenant-owner', 'admin'), async (req, res) => {
  try {
    const { tenantId, workerId } = req.params;
    const { name, email, password, phone, specialization, region, active } = req.body;
    
    // Enforce tenant isolation for non-super-admins
    if (req.admin.role !== 'super-admin') {
      if (!req.admin.tenant || req.admin.tenant.toString() !== tenantId) {
        return res.status(403).json({ message: 'Not authorized to update field workers in this tenant' });
      }
    }
    
    // Find the field worker to update
    const fieldWorker = await FieldWorker.findOne({ _id: workerId, tenant: tenantId });
    
    if (!fieldWorker) {
      return res.status(404).json({ message: 'Field worker not found' });
    }
    
    // Update fields
    if (name) fieldWorker.name = name;
    if (email) fieldWorker.email = email;
    if (password) fieldWorker.password = password;
    if (phone !== undefined) fieldWorker.phone = phone;
    if (specialization) fieldWorker.specialization = specialization;
    if (region) fieldWorker.region = region;
    if (active !== undefined) fieldWorker.active = active;
    
    await fieldWorker.save();
    
    // Return updated field worker data (excluding password)
    const workerData = {
      _id: fieldWorker._id,
      name: fieldWorker.name,
      workerId: fieldWorker.workerId,
      email: fieldWorker.email,
      phone: fieldWorker.phone,
      specialization: fieldWorker.specialization,
      region: fieldWorker.region,
      active: fieldWorker.active,
      tenant: fieldWorker.tenant
    };
    
    res.status(200).json(workerData);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

// Delete field worker
router.delete('/tenants/:tenantId/field-workers/:workerId', restrictTo('super-admin', 'tenant-owner', 'admin'), async (req, res) => {
  try {
    const { tenantId, workerId } = req.params;
    
    // Enforce tenant isolation for non-super-admins
    if (req.admin.role !== 'super-admin') {
      if (!req.admin.tenant || req.admin.tenant.toString() !== tenantId) {
        return res.status(403).json({ message: 'Not authorized to delete field workers from this tenant' });
      }
    }
    
    // Find the field worker to delete
    const fieldWorker = await FieldWorker.findOne({ _id: workerId, tenant: tenantId });
    
    if (!fieldWorker) {
      return res.status(404).json({ message: 'Field worker not found' });
    }
    
    await FieldWorker.deleteOne({ _id: workerId });
    
    res.status(200).json({ message: 'Field worker deleted successfully' });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
});

module.exports = router;
