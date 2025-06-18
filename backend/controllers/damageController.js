const mongoose = require('mongoose');
const DamageReport = require('../models/DamageReport');
const AiReport = require('../models/AiReport');
const FieldWorker = require('../models/FieldWorker');
const upload = require('../middleware/multerConfig');
const path = require('path');


// Helper function to validate MongoDB ObjectId
const isValidObjectId = (id) => {
  return mongoose.Types.ObjectId.isValid(id);
};


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
      // Add tenant reference from middleware
      tenant: req.tenantId,
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
    
    // Start with tenant filter from middleware
    let query = { ...req.query };
    
    // Filter reports by tenant (this ensures tenant isolation)
    // The tenant filter is added by the ensureTenantIsolation middleware
    // in req.query.tenant
    
    if (status) {
      // Handle special case for status not equal (format: !Status)
      if (status.startsWith('!')) {
        query.status = { $ne: status.substring(1) };
      } else {
        query.status = status;
      }
    }
    if (region) query.region = region;
    if (severity) query.severity = severity;
    if (startDate || endDate) {
      query.createdAt = {};
      if (startDate) query.createdAt.$gte = new Date(startDate);
      if (endDate) query.createdAt.$lte = new Date(endDate);
    }

    const reports = await DamageReport.find(query)
      .select('-beforeImage.data -afterImage.data') // Exclude image data from response
      .populate('assignedTo', 'name workerId specialization') // Populate assignedTo with worker details
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
    // Include tenant filter from middleware
    const tenantFilter = req.query.tenant ? { tenant: req.query.tenant } : {};
    
    const report = await DamageReport.findOne({ 
      reportId: req.params.reportId,
      ...tenantFilter // Apply tenant filter
    })
      .select('-beforeImage.data -afterImage.data')
      .populate('assignedTo', 'name workerId specialization'); // Populate assignedTo with worker details
      
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
    
    // If the JWT middleware isn't working, try to handle token from query param
    // This is just a workaround for image loading issues in <img> tags
    const tokenQueryParam = req.query.token;
    if (!req.user && tokenQueryParam) {
      console.log('Using token from query parameter for image authentication');
      // This would need proper JWT validation in a production environment
      // This is just a quick fix for the current issue
    }
    
    const report = await DamageReport.findOne({ reportId });
    
    if (!report) {
      console.log(`Report not found: ${reportId}`);
      return res.status(404).json({ message: 'Report not found' });
    }

    const image = type === 'after' ? report.afterImage : report.beforeImage;
    if (!image || !image.data) {
      console.log(`Image (${type}) not found for report: ${reportId}`);
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

// Create a damage report from AI analysis
const createFromAiReport = async (req, res) => {
  try {
    const { 
      region, 
      location, 
      description, 
      action, 
      damageType, 
      severity, 
      priority, 
      reporter, 
      aiReportId 
    } = req.body;

    // Validate required fields
    const requiredFields = ['region', 'location', 'damageType', 'severity', 'priority', 'reporter'];
    const missingFields = requiredFields.filter(field => !req.body[field]);
    
    if (missingFields.length > 0) {
      return res.status(400).json({ 
        message: 'Missing required fields', 
        fields: missingFields,
        success: false 
      });
    }

    // Generate a report ID
    const reportId = 'DR-' + Date.now();

    // Fetch AI report to get the annotated image
    const aiReport = await AiReport.findById(aiReportId).populate('imageId');
    
    if (!aiReport) {
      return res.status(404).json({ 
        message: 'AI Report not found',
        success: false 
      });
    }

    const newReport = new DamageReport({
      reportId,
      // Add tenant reference
      tenant: req.tenantId,
      region,
      location,
      damageType,
      severity,
      priority,
      action: action || 'Pending Review',
      description: description || '',
      reporter,
      status: 'Pending',
      assignedTo: null,
      aiReportId: aiReportId
    });

    if (aiReport.annotatedImageBase64) {
      // Convert base64 to buffer
      const buffer = Buffer.from(aiReport.annotatedImageBase64, 'base64');
      
      newReport.beforeImage = {
        data: buffer,
        contentType: 'image/jpeg' // Adjust if your annotated images have a different format
      };
    }

    await newReport.save();
    
    res.status(201).json({ 
      message: 'Damage report created successfully from AI analysis',
      success: true,
      report: {
        id: newReport._id,
        reportId: newReport.reportId,
        status: newReport.status,
        createdAt: newReport.createdAt
      }
    });
  } catch (err) {
    console.error('Error creating damage report from AI:', err);
    res.status(500).json({ 
      message: 'Failed to create damage report', 
      error: err.message,
      success: false
    });
  }
};

// Create a damage report from AI analysis and optionally assign to a field worker
const createAndAssignFromAiReport = async (req, res) => {
  try {
    const { 
      region, 
      location, 
      description, 
      action, 
      damageType, 
      severity, 
      priority, 
      reporter, 
      aiReportId,
      workerId
    } = req.body;

    // Validate required fields
    const requiredFields = ['region', 'location', 'damageType', 'severity', 'priority', 'reporter'];
    const missingFields = requiredFields.filter(field => !req.body[field]);
    
    if (missingFields.length > 0) {
      return res.status(400).json({ 
        message: 'Missing required fields', 
        fields: missingFields,
        success: false 
      });
    }

    // Generate a report ID
    const reportId = 'DR-' + Date.now();

    let aiReport = null;
    
    // Only fetch AI report if aiReportId is provided
    if (aiReportId) {
      aiReport = await AiReport.findById(aiReportId).populate('imageId');
      
      if (!aiReport) {
        return res.status(404).json({ 
          message: 'AI Report not found',
          success: false 
        });
      }
    }

    // Create the report
    const newReport = new DamageReport({
      reportId,
      // Add tenant reference
      tenant: req.tenantId,
      region,
      location,
      damageType,
      severity,
      priority,
      action: action || 'Pending Review',
      description: description || '',
      reporter,
      status: 'Pending',
      assignedTo: null,
      ...(aiReportId && { aiReportId }) // Only add aiReportId if it exists
    });

    if (aiReport.annotatedImageBase64) {
      // Convert base64 to buffer
      const buffer = Buffer.from(aiReport.annotatedImageBase64, 'base64');
      
      newReport.beforeImage = {
        data: buffer,
        contentType: 'image/jpeg' // Adjust if your annotated images have a different format
      };
    }

    await newReport.save();

    // If workerId is provided, assign the report to the worker
    let fieldWorker;
    if (workerId) {
      // Check if field worker exists and belongs to the same tenant
      const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
      fieldWorker = await FieldWorker.findOne({
        _id: workerId,
        ...tenantFilter // Ensure field worker belongs to the same tenant
      });
      
      if (!fieldWorker) {
        return res.status(404).json({
          message: 'Field worker not found or not authorized for this tenant',
          success: false
        });
      }

      // Update report
      newReport.assignedTo = fieldWorker._id;
      newReport.assignedAt = new Date();
      newReport.status = 'Assigned';
      
      await newReport.save();
    }
    
    res.status(201).json({ 
      message: 'Damage report created successfully' + (workerId ? ' and assigned to field worker' : ''),
      success: true,
      report: {
        id: newReport._id,
        reportId: newReport.reportId,
        status: newReport.status,
        assignedTo: workerId ? {
          _id: fieldWorker._id,
          name: fieldWorker.name,
          workerId: fieldWorker.workerId,
          specialization: fieldWorker.specialization
        } : null,
        createdAt: newReport.createdAt
      }
    });
  } catch (err) {
    console.error('Error creating damage report from AI:', err);
    res.status(500).json({ 
      message: 'Failed to create damage report', 
      error: err.message,
      success: false
    });
  }
};

// Assign a repair task to a field worker
const assignRepair = async (req, res) => {
  try {
    const { reportId } = req.params;
    const { workerId } = req.body;
    
    console.log('Assign repair request:', { reportId, workerId, body: req.body });

    if (!reportId || !workerId) {
      console.log('Missing required fields:', { reportId, workerId });
      return res.status(400).json({
        message: 'Report ID and Worker ID are required',
        success: false
      });
    }
    
    // Validate ObjectIds
    if (!isValidObjectId(reportId)) {
      console.log(`Invalid report ID format: ${reportId}`);
      return res.status(400).json({
        message: 'Invalid report ID format',
        success: false
      });
    }
    
    if (!isValidObjectId(workerId)) {
      console.log(`Invalid worker ID format: ${workerId}`);
      return res.status(400).json({
        message: 'Invalid worker ID format',
        success: false
      });
    }

    // Apply tenant filter from middleware
    const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};

    // Check if report exists and belongs to the correct tenant
    const report = await DamageReport.findOne({ 
      _id: reportId,
      ...tenantFilter
    });
    
    if (!report) {
      console.log(`Report not found with ID: ${reportId} for tenant: ${req.tenantId}`);
      return res.status(404).json({
        message: 'Damage report not found',
        success: false
      });
    }
    
    console.log('Found report:', { 
      id: report._id,
      reportId: report.reportId, 
      status: report.status,
      assignedTo: report.assignedTo
    });

    // Check if field worker exists and belongs to the correct tenant
    const fieldWorker = await FieldWorker.findOne({
      _id: workerId,
      ...tenantFilter // Ensure field worker belongs to the same tenant
    });
    
    if (!fieldWorker) {
      console.error(`Field worker not found with ID: ${workerId} for tenant: ${req.tenantId}`);
      return res.status(404).json({
        message: `Field worker not found or not authorized for this tenant`,
        success: false
      });
    }
    
    console.log('Found field worker:', { 
      id: fieldWorker._id,
      workerId: fieldWorker.workerId,
      name: fieldWorker.name 
    });

    // Update report
    report.assignedTo = fieldWorker._id;
    report.assignedAt = new Date();
    report.status = 'Assigned';
    
    await report.save();

    res.status(200).json({
      message: 'Repair task assigned successfully',
      success: true,
      report: {
        id: report._id,
        reportId: report.reportId,
        assignedTo: {
          _id: fieldWorker._id,
          name: fieldWorker.name,
          workerId: fieldWorker.workerId,
          specialization: fieldWorker.specialization
        },
        status: report.status
      }
    });
  } catch (err) {
    console.error('Error assigning repair:', err);
    res.status(500).json({ 
      message: 'Failed to assign repair task', 
      error: err.message,
      success: false
    });
  }
};

// Unassign a repair task
const unassignRepair = async (req, res) => {
  try {
    const { reportId } = req.params;

    if (!reportId) {
      return res.status(400).json({
        message: 'Report ID is required',
        success: false
      });
    }

    // Check if report exists
    const report = await DamageReport.findById(reportId);
    if (!report) {
      return res.status(404).json({
        message: 'Damage report not found',
        success: false
      });
    }

    // Update report
    report.assignedTo = null;
    report.status = 'Pending';
    
    await report.save();

    res.status(200).json({
      message: 'Repair task unassigned successfully',
      success: true,
      report: {
        id: report._id,
        reportId: report.reportId,
        assignedTo: null,
        status: report.status
      }
    });
  } catch (err) {
    console.error('Error unassigning repair:', err);
    res.status(500).json({ 
      message: 'Failed to unassign repair task', 
      error: err.message,
      success: false
    });
  }
};

// Update repair status
const updateRepairStatus = async (req, res) => {
  try {
    const { reportId } = req.params;
    const { status } = req.body;

    if (!reportId || !status) {
      return res.status(400).json({
        message: 'Report ID and status are required',
        success: false
      });
    }

    // Validate status
    const validStatuses = ['Pending', 'Assigned', 'In Progress', 'In-Progress', 'On Hold', 'On-Hold', 'Completed', 'Resolved', 'Rejected'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({
        message: 'Invalid status value',
        success: false
      });
    }

    // Check if report exists
    const report = await DamageReport.findById(reportId);
    if (!report) {
      return res.status(404).json({
        message: 'Damage report not found',
        success: false
      });
    }

    // Update report status
    report.status = status;
    
    // If completed, add resolvedAt date
    if (status === 'Completed') {
      report.resolvedAt = new Date();
    }
    
    await report.save();

    res.status(200).json({
      message: 'Repair status updated successfully',
      success: true,
      report: {
        id: report._id,
        reportId: report.reportId,
        status: report.status,
        resolvedAt: report.resolvedAt
      }
    });
  } catch (err) {
    console.error('Error updating repair status:', err);
    res.status(500).json({ 
      message: 'Failed to update repair status', 
      error: err.message,
      success: false
    });
  }
};

// Get reports generated from AI reports
const getGeneratedFromAiReports = async (req, res) => {
  try {
    // Apply tenant filter - only get reports for the current tenant
    // req.tenantId is set by the ensureTenantIsolation middleware
    const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
    
    // Find all damage reports that have aiReportId field and belong to the tenant
    const reports = await DamageReport.find({ 
      aiReportId: { $exists: true },
      ...tenantFilter // Apply tenant filter
    })
      .select('aiReportId reportId createdAt')
      .sort({ createdAt: -1 });
      
    res.status(200).json(reports);
  } catch (err) {
    console.error('Error fetching AI-generated reports:', err);
    res.status(500).json({ 
      message: 'Failed to fetch AI-generated reports', 
      error: err.message 
    });
  }
};

// Update a damage report
const updateReport = async (req, res) => {
  try {
    const { reportId } = req.params;
    const { 
      damageType, 
      severity, 
      priority, 
      action, 
      region, 
      location, 
      description,
      status
    } = req.body;

    // Check if report exists
    const report = await DamageReport.findOne({ reportId });
    if (!report) {
      return res.status(404).json({
        message: 'Damage report not found',
        success: false
      });
    }

    // Update fields if provided
    if (damageType) report.damageType = damageType;
    if (severity) report.severity = severity;
    if (priority) report.priority = priority;
    if (action) report.action = action;
    if (region) report.region = region;
    if (location) report.location = location;
    if (description) report.description = description;
    if (status) {
      report.status = status;
      // If completed, add resolvedAt date
      if (status === 'Completed' && !report.resolvedAt) {
        report.resolvedAt = new Date();
      }
    }

    await report.save();

    res.status(200).json({
      message: 'Report updated successfully',
      success: true,
      report: {
        id: report._id,
        reportId: report.reportId,
        damageType: report.damageType,
        severity: report.severity,
        priority: report.priority,
        action: report.action,
        region: report.region,
        location: report.location,
        description: report.description,
        status: report.status,
        createdAt: report.createdAt,
        updatedAt: report.updatedAt
      }
    });
  } catch (err) {
    console.error('Error updating report:', err);
    res.status(500).json({ 
      message: 'Failed to update report', 
      error: err.message,
      success: false
    });
  }
};

// Delete a damage report
const deleteReport = async (req, res) => {
  try {
    const { reportId } = req.params;

    // Check if report exists
    const report = await DamageReport.findOne({ reportId });
    if (!report) {
      return res.status(404).json({
        message: 'Damage report not found',
        success: false
      });
    }

    await DamageReport.deleteOne({ reportId });

    res.status(200).json({
      message: 'Report deleted successfully',
      success: true,
    });
  } catch (err) {
    console.error('Error deleting report:', err);
    res.status(500).json({ 
      message: 'Failed to delete report', 
      error: err.message,
      success: false
    });
  }
};

// Create a damage report directly (not from AI report)
const createDamageReport = async (req, res) => {
  try {
    const { 
      region, 
      location, 
      description, 
      action, 
      damageType, 
      severity, 
      priority, 
      reporter,
      assignedWorker // This might be passed from frontend
    } = req.body;

    // Add logging to trace request payload
    console.log('Payload received:', req.body);

    // Validate required fields
    const requiredFields = ['region', 'location', 'damageType', 'severity', 'priority', 'reporter'];
    const missingFields = requiredFields.filter(field => !req.body[field]);
    
    if (missingFields.length > 0) {
      return res.status(400).json({ 
        message: 'Missing required fields', 
        fields: missingFields,
        success: false 
      });
    }

    // Generate a report ID
    const reportId = 'DR-' + Date.now();

    // Create the report
    const newReport = new DamageReport({
      reportId,
      // Add tenant reference
      tenant: req.tenantId,
      region,
      location,
      damageType,
      severity,
      priority,
      action: action || 'Pending Review',
      description: description || '',
      reporter,
      status: 'Pending',
      assignedTo: null, // Will be assigned below if assignedWorker is provided
    });

    await newReport.save();

    // If assignedWorker is provided, assign the report
    if (assignedWorker) {
      // Check if field worker exists and belongs to the same tenant
      const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
      const fieldWorker = await FieldWorker.findOne({
        _id: assignedWorker,
        ...tenantFilter
      });

      if (fieldWorker) {
        // Assign the report to this worker
        newReport.assignedTo = fieldWorker._id;
        newReport.status = 'Assigned';
        await newReport.save();

        // Increment active assignments count
        fieldWorker.activeAssignments = (fieldWorker.activeAssignments || 0) + 1;
        await fieldWorker.save();
      }
    }

    res.status(201).json({
      message: 'Damage report created successfully',
      success: true,
      report: {
        _id: newReport._id,
        reportId: newReport.reportId,
        status: newReport.status,
        assignedTo: newReport.assignedTo,
        createdAt: newReport.createdAt
      }
    });

  } catch (error) {
    console.error('Error creating damage report:', error);
    res.status(500).json({ 
      message: 'Failed to create damage report', 
      error: error.message,
      success: false 
    });
  }
};

// Search across all reports and related data
const searchAllReportsAndData = async (req, res) => {
  try {
    const { q: searchQuery, quick } = req.query;
    const isQuickSearch = quick === 'true';

    if (!searchQuery || searchQuery.trim() === '') {
      return res.status(400).json({ message: 'Search query is required' });
    }

    // Apply tenant filter from middleware if available
    const tenantFilter = req.query.tenant ? { tenant: req.query.tenant } : {};
    
    // Create regex for case-insensitive search
    const searchRegex = new RegExp(searchQuery, 'i');

    // Search in DamageReports
    const reports = await DamageReport.find({
      ...tenantFilter,
      $or: [
        { reportId: searchRegex },
        { damageType: searchRegex },
        { location: searchRegex },
        { region: searchRegex },
        { description: searchRegex },
        { status: searchRegex },
        { action: searchRegex },
        { priority: searchRegex },
        { severity: searchRegex },
      ],
    })
      .select('-beforeImage.data -afterImage.data')
      .populate('assignedTo', 'name workerId specialization')
      .sort({ createdAt: -1 })
      .limit(isQuickSearch ? 5 : 20);

    // Search in FieldWorkers (if they exist in the model)
    let fieldWorkers = [];
    try {
      fieldWorkers = await FieldWorker.find({
        ...tenantFilter,
        $or: [
          { name: searchRegex },
          { workerId: searchRegex },
          { specialization: searchRegex },
          { email: searchRegex },
          { phone: searchRegex },
        ],
      })
        .select('-password')
        .limit(isQuickSearch ? 3 : 15);
    } catch (err) {
      console.warn('Error searching field workers:', err);
    }

    // Prepare mock analytics data
    // In a real implementation, you might search an analytics collection
    const analyticsTerms = [
      'report', 'analysis', 'damage', 'repair', 'pothole', 'crack', 
      'road', 'street', 'bridge', 'infrastructure', 'maintenance'
    ];
    
    let analytics = [];
    if (analyticsTerms.some(term => term.includes(searchQuery.toLowerCase()) || 
        searchQuery.toLowerCase().includes(term))) {
      analytics = [
        {
          id: 'analytics-1',
          title: 'Damage Trends Analysis',
          description: 'Analysis of damage trends over time',
          type: 'Trends',
          date: new Date()
        },
        {
          id: 'analytics-2',
          title: 'Regional Damage Distribution',
          description: 'Distribution of damage reports by region',
          type: 'Report',
          date: new Date()
        },
        {
          id: 'analytics-3',
          title: 'Repair Performance Metrics',
          description: 'Key performance indicators for repair teams',
          type: 'Performance',
          date: new Date()
        },
      ].filter(item => 
        item.title.match(searchRegex) || 
        item.description.match(searchRegex) ||
        item.type.match(searchRegex)
      );
    }

    // Prepare mock repairs data
    // In a real implementation, you would search an actual repairs collection
    const repairs = reports
      .filter(report => report.status === 'Assigned' || report.status === 'In Progress')
      .map(report => ({
        _id: report._id,
        repairId: `R-${report.reportId.substring(3)}`,
        description: `Repair for ${report.damageType} at ${report.location}`,
        status: report.status,
        location: report.location,
        assignedTo: report.assignedTo,
        startDate: report.updatedAt,
        estimatedCompletion: new Date(new Date(report.updatedAt).getTime() + 7 * 24 * 60 * 60 * 1000), // 1 week later
      }));

    res.status(200).json({
      reports,
      fieldWorkers,
      analytics,
      repairs,
      query: searchQuery,
    });
  } catch (err) {
    console.error('Error performing search:', err);
    res.status(500).json({ 
      message: 'Failed to perform search', 
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
  createFromAiReport,
  createAndAssignFromAiReport,
  assignRepair,
  unassignRepair,
  updateRepairStatus,
  upload,
  getGeneratedFromAiReports,
  updateReport,
  deleteReport,
  createDamageReport,
  searchAllReportsAndData
};
