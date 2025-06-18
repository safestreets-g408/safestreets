const mongoose = require('mongoose');

const TenantSchema = new mongoose.Schema({
  name: { 
    type: String, 
    required: true,
    trim: true
  },
  code: { 
    type: String, 
    required: true, 
    unique: true,
    trim: true
  },
  description: { 
    type: String,
    trim: true 
  },
  active: { 
    type: Boolean, 
    default: true 
  },
  settings: {
    logo: { type: String },
    primaryColor: { type: String, default: '#1976d2' },
    secondaryColor: { type: String, default: '#f50057' },
    customDomain: { type: String },
    maxFieldWorkers: { type: Number, default: 10 },
    maxAdmins: { type: Number, default: 2 }
  },
  createdAt: { 
    type: Date, 
    default: Date.now 
  },
  createdBy: { 
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin'
  },
  updatedAt: { 
    type: Date
  }
});

module.exports = mongoose.model('Tenant', TenantSchema);
