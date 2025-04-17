const mongoose = require('mongoose');

const DamageReportSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  imagePath: { type: String, required: true },
  damageType: { type: String, required: true },       // e.g., Pothole
  severity: { type: String, required: true },         // e.g., High
  priority: { type: String, required: true },         // e.g., Urgent
  action: { type: String, required: true }, 
  region : { type: String, required: true},
  description: { type: String },
  reporter: { type: String, required: true },
  location: { type: String, required: true },          // e.g., Immediate repair
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('DamageReport', DamageReportSchema);
