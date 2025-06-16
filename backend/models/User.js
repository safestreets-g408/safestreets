const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const UserSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  type: {
    type: String,
    enum: ['user', 'admin'],
    default: 'user'
  },
  profile: {
    role: { type: String, default: 'Staff Member' },
    phone: { type: String },
    location: { type: String },
    department: { type: String },
    joinDate: { type: Date, default: Date.now },
    bio: { type: String },
    skills: [{ type: String }],
    avatar: { type: String },
    lastActive: { type: Date, default: Date.now }
  }
});

// Encrypt password before saving
UserSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// Match password
UserSchema.methods.matchPassword = async function (enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('User', UserSchema);
