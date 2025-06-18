const Tenant = require('../models/Tenant');
const Admin = require('../models/Admin');
const mongoose = require('mongoose');

// Create tenant
const createTenant = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { 
      name, 
      code, 
      description, 
      settings, 
      adminName, 
      adminEmail, 
      adminPassword,
      fieldWorkers // Optional array of field workers to create initially
    } = req.body;

    // Check if tenant code already exists
    const existingTenant = await Tenant.findOne({ code }).session(session);
    if (existingTenant) {
      await session.abortTransaction();
      session.endSession();
      return res.status(400).json({ message: 'Tenant with this code already exists' });
    }

    // Check if admin email already exists
    const existingAdmin = await Admin.findOne({ email: adminEmail }).session(session);
    if (existingAdmin) {
      await session.abortTransaction();
      session.endSession();
      return res.status(400).json({ message: 'Admin with this email already exists' });
    }

    // Create tenant
    const tenant = await Tenant.create([{
      name,
      code,
      description,
      settings,
      createdBy: req.admin._id,
      updatedAt: Date.now()
    }], { session });

    // Create tenant owner admin
    const admin = await Admin.create([{
      name: adminName,
      email: adminEmail,
      password: adminPassword,
      role: 'tenant-owner',
      tenant: tenant[0]._id
    }], { session });
    
    // Create field workers if provided
    const createdFieldWorkers = [];
    if (fieldWorkers && Array.isArray(fieldWorkers) && fieldWorkers.length > 0) {
      // Check if we have the FieldWorker model
      const FieldWorker = require('../models/FieldWorker');
      
      // Validate against the max field workers setting
      if (fieldWorkers.length > settings.maxFieldWorkers) {
        await session.abortTransaction();
        session.endSession();
        return res.status(400).json({ 
          message: `Cannot create more than ${settings.maxFieldWorkers} field workers for this tenant` 
        });
      }
      
      // Check for duplicate emails
      const fieldWorkerEmails = fieldWorkers.map(fw => fw.email);
      const uniqueEmails = new Set(fieldWorkerEmails);
      if (uniqueEmails.size !== fieldWorkerEmails.length) {
        await session.abortTransaction();
        session.endSession();
        return res.status(400).json({ message: 'Duplicate field worker emails are not allowed' });
      }
      
      // Create each field worker
      for (const worker of fieldWorkers) {
        const newWorker = await FieldWorker.create([{
          name: worker.name,
          workerId: worker.workerId,
          email: worker.email,
          password: worker.password,
          phone: worker.phone || '',
          specialization: worker.specialization,
          region: worker.region,
          tenant: tenant[0]._id
        }], { session });
        
        createdFieldWorkers.push({
          _id: newWorker[0]._id,
          name: newWorker[0].name,
          email: newWorker[0].email
        });
      }
    }

    await session.commitTransaction();
    session.endSession();

    res.status(201).json({
      tenant: {
        _id: tenant[0]._id,
        name: tenant[0].name,
        code: tenant[0].code,
        description: tenant[0].description,
        settings: tenant[0].settings,
        active: tenant[0].active,
        createdAt: tenant[0].createdAt
      },
      admin: {
        _id: admin[0]._id,
        name: admin[0].name,
        email: admin[0].email,
        role: admin[0].role
      },
      fieldWorkers: createdFieldWorkers // Include any field workers created
    });
  } catch (err) {
    await session.abortTransaction();
    session.endSession();
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Get all tenants (super admin only)
const getAllTenants = async (req, res) => {
  try {
    const tenants = await Tenant.find()
      .select('name code description active settings createdAt updatedAt')
      .sort({ createdAt: -1 });
    
    res.status(200).json(tenants);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Get tenant by ID (super admin only)
const getTenantById = async (req, res) => {
  try {
    const tenant = await Tenant.findById(req.params.id);
    if (!tenant) {
      return res.status(404).json({ message: 'Tenant not found' });
    }
    
    res.status(200).json(tenant);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Update tenant
const updateTenant = async (req, res) => {
  try {
    const { name, description, settings, active } = req.body;
    
    const tenant = await Tenant.findById(req.params.id);
    if (!tenant) {
      return res.status(404).json({ message: 'Tenant not found' });
    }
    
    tenant.name = name || tenant.name;
    tenant.description = description || tenant.description;
    tenant.active = active !== undefined ? active : tenant.active;
    
    if (settings) {
      tenant.settings = {
        ...tenant.settings,
        ...settings
      };
    }
    
    tenant.updatedAt = Date.now();
    
    const updatedTenant = await tenant.save();
    
    res.status(200).json({
      _id: updatedTenant._id,
      name: updatedTenant.name,
      code: updatedTenant.code,
      description: updatedTenant.description,
      settings: updatedTenant.settings,
      active: updatedTenant.active,
      createdAt: updatedTenant.createdAt,
      updatedAt: updatedTenant.updatedAt
    });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Delete tenant
const deleteTenant = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();
  
  try {
    const tenant = await Tenant.findById(req.params.id).session(session);
    if (!tenant) {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({ message: 'Tenant not found' });
    }
    
    // Delete all related data in a transaction to ensure consistency
    await Admin.deleteMany({ tenant: tenant._id }).session(session);
    
    await tenant.deleteOne();
    
    await session.commitTransaction();
    session.endSession();
    
    res.status(200).json({ message: 'Tenant deleted successfully' });
  } catch (err) {
    await session.abortTransaction();
    session.endSession();
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

module.exports = {
  createTenant,
  getAllTenants,
  getTenantById,
  updateTenant,
  deleteTenant
};
