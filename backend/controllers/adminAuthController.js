const Admin = require('../models/Admin');
const jwt = require('jsonwebtoken');

// Register Admin
const registerAdmin = async (req, res) => {
  const { name, email, password, role } = req.body;
  try {
    const adminExists = await Admin.findOne({ email });
    if (adminExists) {
      return res.status(400).json({ message: 'Admin account already exists with this email' });
    }

    const admin = await Admin.create({ name, email, password, role });
    const token = jwt.sign({ adminId: admin._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    
    // Return admin data (excluding password)
    const adminData = {
      _id: admin._id,
      name: admin.name,
      email: admin.email,
      role: admin.role,
      profile: admin.profile || {}
    };
    
    res.status(201).json({ token, admin: adminData });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Login Admin
const loginAdmin = async (req, res) => {
  const { email, password } = req.body;
  try {
    const admin = await Admin.findOne({ email });
    if (!admin || !(await admin.matchPassword(password))) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Update last active timestamp
    admin.profile = admin.profile || {};
    admin.profile.lastActive = Date.now();
    await admin.save();

    const token = jwt.sign({ adminId: admin._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    
    // Return admin data (excluding password)
    const adminData = {
      _id: admin._id,
      name: admin.name,
      email: admin.email,
      role: admin.role,
      profile: admin.profile || {}
    };
    
    res.status(200).json({ token, admin: adminData });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

module.exports = { registerAdmin, loginAdmin };
