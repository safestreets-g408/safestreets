const mongoose = require('mongoose');

const DamageReportSchema = new mongoose.Schema({
  reportId : {type: String, required: true},
  region : { type: String, required: true},
  imagePath: { type: String },
  damageType: { type: String, required: true },       
  severity: { type: String, required: true },         
  priority: { type: String, required: true },         
  action: { type: String, required: true }, 
  description: { type: String },
  reporter: { type: String, required: true },
  location: { type: String, required: true },          
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('DamageReport', DamageReportSchema);
