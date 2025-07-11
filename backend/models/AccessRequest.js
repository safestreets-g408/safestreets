const mongoose = require('mongoose');

const AccessRequestSchema = new mongoose.Schema(
  {
    organizationName: { type: String, required: true },
    contactName: { type: String, required: true },
    email: { type: String, required: true },
    phone: { type: String, required: true },
    region: { type: String, required: true },
    reason: { type: String, required: true },
    status: {
      type: String,
      enum: ['pending', 'approved', 'rejected'],
      default: 'pending'
    },
    reviewedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Admin'
    },
    reviewNotes: { type: String },
    reviewedAt: { type: Date },
    tenantCreated: {
      type: Boolean,
      default: false
    }
  },
  {
    timestamps: true
  }
);

module.exports = mongoose.model('AccessRequest', AccessRequestSchema);
