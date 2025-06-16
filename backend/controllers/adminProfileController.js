const Admin = require('../models/Admin');

// Get admin profile
const getAdminProfile = async (req, res) => {
  try {
    const admin = await Admin.findById(req.admin.id).select('-password');
    if (!admin) {
      return res.status(404).json({ message: 'Admin not found' });
    }
    res.status(200).json(admin);
  } catch (err) {
    console.error('Error getting admin profile:', err.message);
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Update admin profile
const updateAdminProfile = async (req, res) => {
  const { name, email, profile } = req.body;
  
  // Build profile object
  const profileFields = {};
  if (name) profileFields.name = name;
  if (email) profileFields.email = email;
  
  // Build profile object
  profileFields.profile = {};
  if (profile) {
    if (profile.position) profileFields.profile.position = profile.position;
    if (profile.phone) profileFields.profile.phone = profile.phone;
    if (profile.location) profileFields.profile.location = profile.location;
    if (profile.department) profileFields.profile.department = profile.department;
    if (profile.bio) profileFields.profile.bio = profile.bio;
    if (profile.skills) profileFields.profile.skills = profile.skills;
    if (profile.avatar) profileFields.profile.avatar = profile.avatar;
    profileFields.profile.lastActive = Date.now();
  }
  
  try {
    let admin = await Admin.findById(req.admin.id);
    
    if (!admin) {
      return res.status(404).json({ message: 'Admin not found' });
    }
    
    // Check if email is being updated and it's not the same as current
    if (email && email !== admin.email) {
      const existingAdmin = await Admin.findOne({ email });
      if (existingAdmin) {
        return res.status(400).json({ message: 'Email is already in use' });
      }
    }
    
    // Update admin
    admin = await Admin.findByIdAndUpdate(
      req.admin.id,
      { $set: profileFields },
      { new: true }
    ).select('-password');
    
    res.json(admin);
  } catch (err) {
    console.error('Error updating admin profile:', err.message);
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

module.exports = { getAdminProfile, updateAdminProfile };
