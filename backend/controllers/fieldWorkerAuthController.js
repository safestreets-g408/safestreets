const FieldWorker = require('../models/FieldWorker');
const jwt = require('jsonwebtoken');

// Register Field Worker
const registerFieldWorker = async (req, res) => {
  const { name, workerId, specialization, region, email, password } = req.body;
  try {
    const workerExists = await FieldWorker.findOne({ 
      $or: [{ email }, { workerId }] 
    });
    
    if (workerExists) {
      return res.status(400).json({ 
        message: 'Field worker already exists with this email or worker ID' 
      });
    }

    const fieldWorker = await FieldWorker.create({ 
      name, 
      workerId, 
      specialization, 
      region,
      email,
      password
    });
    
    const token = jwt.sign(
      { fieldWorkerId: fieldWorker._id }, 
      process.env.JWT_SECRET, 
      { expiresIn: '24h' }
    );
    
    // Return field worker data (excluding password)
    const fieldWorkerData = {
      _id: fieldWorker._id,
      name: fieldWorker.name,
      workerId: fieldWorker.workerId,
      email: fieldWorker.email,
      specialization: fieldWorker.specialization,
      region: fieldWorker.region,
      activeAssignments: fieldWorker.activeAssignments,
      profile: fieldWorker.profile || {}
    };
    
    res.status(201).json({ token, fieldWorker: fieldWorkerData });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Login Field Worker
const loginFieldWorker = async (req, res) => {
  const { email, password } = req.body;
  
  try {
    const fieldWorker = await FieldWorker.findOne({ email });
    
    if (!fieldWorker || !(await fieldWorker.matchPassword(password))) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Update last active timestamp
    fieldWorker.profile = fieldWorker.profile || {};
    fieldWorker.profile.lastActive = Date.now();
    await fieldWorker.save();

    const token = jwt.sign(
      { fieldWorkerId: fieldWorker._id }, 
      process.env.JWT_SECRET, 
      { expiresIn: '24h' }
    );
    
    // Return field worker data (excluding password)
    const fieldWorkerData = {
      _id: fieldWorker._id,
      name: fieldWorker.name,
      workerId: fieldWorker.workerId,
      email: fieldWorker.email,
      specialization: fieldWorker.specialization,
      region: fieldWorker.region,
      activeAssignments: fieldWorker.activeAssignments,
      profile: fieldWorker.profile || {}
    };
    
    res.status(200).json({ token, fieldWorker: fieldWorkerData });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Get Field Worker Profile
const getFieldWorkerProfile = async (req, res) => {
  try {
    const fieldWorker = await FieldWorker.findById(req.fieldWorker.id).select('-password');
    
    if (!fieldWorker) {
      return res.status(404).json({ message: 'Field worker not found' });
    }
    
    res.status(200).json({ fieldWorker });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Update Field Worker Profile
const updateFieldWorkerProfile = async (req, res) => {
  try {
    const { name, specialization, region, profile } = req.body;
    
    const fieldWorker = await FieldWorker.findByIdAndUpdate(
      req.fieldWorker.id,
      { 
        name, 
        specialization, 
        region,
        profile: { ...profile }
      },
      { new: true, runValidators: true }
    ).select('-password');
    
    if (!fieldWorker) {
      return res.status(404).json({ message: 'Field worker not found' });
    }
    
    res.status(200).json({ fieldWorker });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

module.exports = { 
  registerFieldWorker, 
  loginFieldWorker, 
  getFieldWorkerProfile,
  updateFieldWorkerProfile
};
