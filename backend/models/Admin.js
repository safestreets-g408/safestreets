const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const AdminSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: {
    type: String,
    enum: ['admin', 'super-admin', 'tenant-owner'],
    default: 'admin'
  },
  tenant: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Tenant',
    required: function() {
      return this.role !== 'super-admin'; // Tenant is required except for super-admin
    }
  },
  profile: {
    position: { type: String, default: 'Administrator' },
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
AdminSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// Match password
AdminSchema.methods.matchPassword = async function (enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('Admin', AdminSchema);
