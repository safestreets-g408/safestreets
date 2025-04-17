const DamageReport = require('../models/DamageReport');
const multer = require('multer');
const path = require('path');

// Configure Multer Storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploads');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage });

// Upload image and classify damage
const uploadDamageReport = async (req, res) => {
  const { damageType, severity, priority, action } = req.body;
  const userId = req.user.userId;

  if (!req.file) {
    return res.status(400).json({ message: 'No image uploaded' });
  }

  try {
    // Save damage report to DB
    const newReport = new DamageReport({
      user: userId,
      imagePath: req.file.path,
      damageType,
      severity,
      priority,
      action,
      region,
      location,
      description,
      reporter
    });

    await newReport.save();
    res.status(201).json({ message: 'Damage report uploaded', report: newReport });
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

// Fetch damage history for authenticated user
const getDamageHistory = async (req, res) => {
  try {
    const reports = await DamageReport.find({ user: req.user.userId });
    res.status(200).json(reports);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err.message });
  }
};

module.exports = { uploadDamageReport, getDamageHistory, upload };
