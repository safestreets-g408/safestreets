const User = require('../models/User');

// Get user profile
const getProfile = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-password');
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    res.status(200).json(user);
  } catch (err) {
    console.error('Error getting profile:', err.message);
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Update user profile
const updateProfile = async (req, res) => {
  const { name, email, profile } = req.body;
  
  // Build profile object
  const profileFields = {};
  if (name) profileFields.name = name;
  if (email) profileFields.email = email;
  
  // Build profile object
  profileFields.profile = {};
  if (profile) {
    if (profile.role) profileFields.profile.role = profile.role;
    if (profile.phone) profileFields.profile.phone = profile.phone;
    if (profile.location) profileFields.profile.location = profile.location;
    if (profile.department) profileFields.profile.department = profile.department;
    if (profile.bio) profileFields.profile.bio = profile.bio;
    if (profile.skills) profileFields.profile.skills = profile.skills;
    if (profile.avatar) profileFields.profile.avatar = profile.avatar;
    profileFields.profile.lastActive = Date.now();
  }
  
  try {
    let user = await User.findById(req.user.id);
    
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    // Check if email is being updated and it's not the same as current
    if (email && email !== user.email) {
      const existingUser = await User.findOne({ email });
      if (existingUser) {
        return res.status(400).json({ message: 'Email is already in use' });
      }
    }
    
    // Update user
    user = await User.findByIdAndUpdate(
      req.user.id,
      { $set: profileFields },
      { new: true }
    ).select('-password');
    
    res.json(user);
  } catch (err) {
    console.error('Error updating profile:', err.message);
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

module.exports = { getProfile, updateProfile };
