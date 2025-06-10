const DamageReport = require('../models/DamageReport');
const upload = require('../middleware/multerConfig');
const path = require('path');


// Upload image and classify damage
const uploadDamageReport = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No image uploaded' });
    }

    const { damageType, severity, priority, action, region, location, description, reporter } = req.body;

    // Validate required fields
    const requiredFields = ['damageType', 'severity', 'priority', 'region', 'location', 'reporter'];
    const missingFields = requiredFields.filter(field => !req.body[field]);
    
    if (missingFields.length > 0) {
      return res.status(400).json({ 
        message: 'Missing required fields', 
        fields: missingFields 
      });
    }

    const reportId = 'DR-' + Date.now();
    const newReport = new DamageReport({
      reportId,
      beforeImage: {
        data: req.file.buffer,
        contentType: req.file.mimetype
      },
      damageType,
      severity,
      priority,
      action: action || 'Pending Review',
      region,
      location,
      description: description || '',
      reporter,
      status: 'Pending'
    });

    await newReport.save();
    
    res.status(201).json({ 
      message: 'Damage report uploaded successfully',
      report: {
        id: newReport._id,
        reportId: newReport.reportId,
        status: newReport.status,
        createdAt: newReport.createdAt
      }
    });
  } catch (err) {
    console.error('Error uploading damage report:', err);
    res.status(500).json({ 
      message: 'Failed to upload damage report', 
      error: err.message 
    });
  }
};

// Get all reports with optional filtering
const getReports = async (req, res) => {
  try {
    const { status, region, severity, startDate, endDate } = req.query;
    
    let query = {};
    
    if (status) query.status = status;
    if (region) query.region = region;
    if (severity) query.severity = severity;
    if (startDate || endDate) {
      query.createdAt = {};
      if (startDate) query.createdAt.$gte = new Date(startDate);
      if (endDate) query.createdAt.$lte = new Date(endDate);
    }

    const reports = await DamageReport.find(query)
      .select('-beforeImage.data -afterImage.data') // Exclude image data from response
      .sort({ createdAt: -1 });
      
    res.status(200).json(reports);
  } catch (err) {
    console.error('Error fetching reports:', err);
    res.status(500).json({ 
      message: 'Failed to fetch reports', 
      error: err.message 
    });
  }
};

// Get damage report by ID
const getReportById = async (req, res) => {
  try {
    const report = await DamageReport.findOne({ reportId: req.params.reportId })
      .select('-beforeImage.data -afterImage.data');
      
    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }
    
    res.status(200).json(report);
  } catch (err) {
    console.error('Error fetching report:', err);
    res.status(500).json({ 
      message: 'Failed to fetch report', 
      error: err.message 
    });
  }
};

// Get report image
const getReportImage = async (req, res) => {
  try {
    const { reportId, type } = req.params; // type can be 'before' or 'after'
    const report = await DamageReport.findOne({ reportId });
    
    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    const image = type === 'after' ? report.afterImage : report.beforeImage;
    if (!image || !image.data) {
      return res.status(404).json({ message: 'Image not found' });
    }

    res.set('Content-Type', image.contentType);
    res.send(image.data);
  } catch (err) {
    console.error('Error fetching image:', err);
    res.status(500).json({ 
      message: 'Failed to fetch image', 
      error: err.message 
    });
  }
};

// Get damage history for a specific user
const getDamageHistory = async (req, res) => {
  try {
    const reports = await DamageReport.find({ reporter: req.user.userId })
      .select('-beforeImage.data -afterImage.data')
      .sort({ createdAt: -1 });
      
    res.status(200).json(reports);
  } catch (err) {
    console.error('Error fetching damage history:', err);
    res.status(500).json({ 
      message: 'Failed to fetch damage history', 
      error: err.message 
    });
  }
};

module.exports = { 
  uploadDamageReport, 
  getDamageHistory, 
  getReports,
  getReportById,
  getReportImage,
  upload 
};
