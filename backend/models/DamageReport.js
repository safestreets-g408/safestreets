const mongoose = require('mongoose');

const ImageSchema = new mongoose.Schema({
  data: Buffer,
  contentType: String
});

const DamageReportSchema = new mongoose.Schema({
  reportId: { type: String, required: true },
  region: { type: String, required: true },
  beforeImage: ImageSchema,
  afterImage: ImageSchema,
  damageType: { type: String, required: true },
  severity: { type: String, required: true },
  priority: { type: String, required: true },
  action: { type: String, required: true },
  description: { type: String },
  status: { type: String, default: 'Pending' },
  assignedTo: { type: String, default: 'Unassigned' },
  assignedAt: { type: Date },
  resolvedAt: { type: Date },
  reporter: { type: String, required: true },
  location: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('DamageReport', DamageReportSchema);
