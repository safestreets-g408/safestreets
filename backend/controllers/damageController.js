const mongoose = require('mongoose');
const DamageReport = require('../models/DamageReport');
const AiReport = require('../models/AiReport');
const FieldWorker = require('../models/FieldWorker');
const upload = require('../middleware/multerConfig');
const path = require('path');
const { generateDamageSummary } = require('../utils/aiUtils');
const { clearDamageReportCaches, clearSingleReportCache } = require('../utils/cacheUtils');
const NotificationService = require('../utils/notificationService');


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
    
    // Generate a summary using AI if description is missing
    let finalDescription = description;
    let formattedDescription = null;
    let descriptionType = 'standard';
    
    if (!description || description.trim() === '') {
      try {
        console.log('Generating professional AI summary for damage report');
        
        // The generateDamageSummary function now returns a professionally formatted summary
        const aiGeneratedSummary = await generateDamageSummary({
          location,
          damageType,
          severity,
          priority
        });
        
        console.log('Professional AI summary generated successfully');
        
        // Check if the description is in markdown format (starts with ## or has markdown headers)
        if (aiGeneratedSummary.includes('##') || aiGeneratedSummary.includes('**')) {
          formattedDescription = aiGeneratedSummary;
          descriptionType = 'professional';
          
          // Create a simplified plain text version for backward compatibility
          // Extract just the first paragraph or sentence for simple description
          const plainTextDescription = aiGeneratedSummary
            .replace(/##.*?\n/g, '')  // Remove headers
            .replace(/\*\*/g, '')     // Remove bold markers
            .replace(/\n\n/g, ' ')    // Replace double line breaks with space
            .split('.')[0] + '.';     // Get first sentence
            
          finalDescription = plainTextDescription;
          
          console.log('Created both formatted and plain text versions of description');
        } else {
          finalDescription = aiGeneratedSummary;
        }
      } catch (summaryError) {
        console.error('Error generating AI summary:', summaryError);
        // Use a simple default if AI generation fails
        finalDescription = `Road damage at ${location}: ${damageType} with ${severity} severity. Priority level: ${priority}.`;
        descriptionType = 'standard';
      }
    }

    const reportId = 'DR-' + Date.now();
    // Process location data if possible
    let processedLocation = location;
    if (location) {
      try {
        // Check if it's already a JSON string or object
        if (typeof location === 'string') {
          try {
            // Try to parse as JSON
            JSON.parse(location);
            // If no error, it's already valid JSON
            processedLocation = location;
          } catch (e) {
            // Check if it contains coordinates
            const coordMatch = location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
            if (coordMatch) {
              // If it contains coordinates, structure it as a JSON string
              processedLocation = JSON.stringify({
                coordinates: [parseFloat(coordMatch[2]), parseFloat(coordMatch[1])],
                address: location
              });
            }
          }
        } else if (typeof location === 'object') {
          // Ensure it's stored as a JSON string
          processedLocation = JSON.stringify(location);
        }
      } catch (e) {
        console.error('Error processing location data:', e);
        // Keep the original value if there's an error
      }
    }
    
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
      location: processedLocation,
      description: finalDescription,
      formattedDescription: formattedDescription,
      descriptionType: descriptionType,
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
    let query = {};
    
    // Filter reports by tenant (this ensures tenant isolation)
    // The tenant filter is added by the ensureTenantIsolation middleware
    if (req.tenantId) {
      query.tenant = req.tenantId;
    } else if (req.query.tenant) {
      query.tenant = req.query.tenant;
    }
    
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
      
    // Process location data before sending response
    const processedReports = reports.map(report => {
      const reportObj = report.toObject();
      
      // Process location data if it exists
      if (reportObj.location) {
        try {
          // Try to parse location as JSON if it's a string
          if (typeof reportObj.location === 'string') {
            try {
              const parsedLocation = JSON.parse(reportObj.location);
              reportObj.location = parsedLocation;
            } catch (e) {
              // Check if it contains coordinates
              const coordMatch = reportObj.location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
              if (coordMatch) {
                // If it contains coordinates, structure it as an object
                reportObj.location = {
                  coordinates: [parseFloat(coordMatch[2]), parseFloat(coordMatch[1])],
                  address: reportObj.location
                };
              }
              // Otherwise keep it as is
            }
          }
        } catch (e) {
          console.error('Error processing location data:', e);
        }
      }
      
      return reportObj;
    });
      
    res.status(200).json(processedReports);
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
    // Create tenant filter from middleware
    const tenantFilter = {};
    if (req.tenantId) {
      tenantFilter.tenant = req.tenantId;
    } else if (req.query.tenant) {
      tenantFilter.tenant = req.query.tenant;
    }
    
    const report = await DamageReport.findOne({ 
      reportId: req.params.reportId,
      ...tenantFilter // Apply tenant filter
    })
      .select('-beforeImage.data -afterImage.data')
      .populate('assignedTo', 'name workerId specialization'); // Populate assignedTo with worker details
      
    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }
    
    // Process location data before sending response
    const reportObj = report.toObject();
    
    // Process location data if it exists
    if (reportObj.location) {
      try {
        // Try to parse location as JSON if it's a string
        if (typeof reportObj.location === 'string') {
          try {
            const parsedLocation = JSON.parse(reportObj.location);
            reportObj.location = parsedLocation;
          } catch (e) {
            // Check if it contains coordinates
            const coordMatch = reportObj.location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
            if (coordMatch) {
              // If it contains coordinates, structure it as an object
              reportObj.location = {
                coordinates: [parseFloat(coordMatch[2]), parseFloat(coordMatch[1])],
                address: reportObj.location
              };
            }
            // Otherwise keep it as is
          }
        }
      } catch (e) {
        console.error('Error processing location data:', e);
      }
    }
    
    res.status(200).json(reportObj);
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
    
    // Handle authentication via query parameter for field workers
    const tokenQueryParam = req.query.token;
    let isAuthenticated = false;
    
    if (tokenQueryParam) {
      try {
        // Import JWT for token verification
        const jwt = require('jsonwebtoken');
        const decoded = jwt.verify(tokenQueryParam, process.env.JWT_SECRET);
        
        // Check if this is a field worker token
        if (decoded.fieldWorkerId) {
          console.log(`Field worker authenticated for image access: ${decoded.fieldWorkerId}`);
          isAuthenticated = true;
        } else if (decoded.adminId || decoded.userId) {
          console.log(`Admin/User authenticated for image access`);
          isAuthenticated = true;
        }
      } catch (tokenError) {
        console.log('Invalid token provided for image access:', tokenError.message);
        // Continue without authentication for backward compatibility
      }
    }
    
    // For admin routes, check if user is already authenticated
    if (req.user) {
      isAuthenticated = true;
    }
    
    // Find report - use _id if it looks like a MongoDB ObjectId, otherwise use reportId
    let report;
    const isObjectId = /^[0-9a-fA-F]{24}$/.test(reportId);
    
    if (isObjectId) {
      report = await DamageReport.findById(reportId);
    } else {
      report = await DamageReport.findOne({ reportId });
    }
    
    if (!report) {
      console.log(`Report not found: ${reportId}`);
      return res.status(404).json({ message: 'Report not found' });
    }

    const image = type === 'after' ? report.afterImage : report.beforeImage;
    if (!image || !image.data) {
      console.log(`Image (${type}) not found for report: ${reportId}`);
      return res.status(404).json({ message: 'Image not found' });
    }

    // Set appropriate headers for image caching
    res.set({
      'Content-Type': image.contentType,
      'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
      'Access-Control-Allow-Origin': '*', // Allow cross-origin requests for images
      'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    });
    
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
    console.log('Finding AI Report with ID:', aiReportId);
    
    const aiReport = await AiReport.findById(aiReportId).populate('imageId');
    
    if (!aiReport) {
      console.error('AI Report not found with ID:', aiReportId);
      return res.status(404).json({ 
        message: 'AI Report not found',
        success: false 
      });
    }
    
    console.log('Found AI Report:', {
      id: aiReport._id,
      predictionClass: aiReport.predictionClass,
      damageType: aiReport.damageType,
      imageExists: !!aiReport.annotatedImageBase64,
      imageLength: aiReport.annotatedImageBase64 ? aiReport.annotatedImageBase64.length : 0
    });
    
    // Generate a summary using AI if description is missing
    let finalDescription = description;
    if (!description || description.trim() === '') {
      try {
        console.log('Generating AI summary for damage report');
        finalDescription = await generateDamageSummary({
          location,
          damageType,
          severity,
          priority
        });
        console.log('AI summary generated successfully');
      } catch (summaryError) {
        console.error('Error generating AI summary:', summaryError);
        // Use a simple default if AI generation fails
        finalDescription = `Road damage at ${location}: ${damageType} with ${severity} severity. Priority level: ${priority}.`;
      }
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
      description: finalDescription,
      reporter,
      status: 'Pending',
      assignedTo: null,
      aiReportId: aiReportId
    });

    try {
      if (aiReport.annotatedImageBase64) {
        console.log('Found annotated image, converting to buffer...');
        
        // Convert base64 to buffer
        const buffer = Buffer.from(aiReport.annotatedImageBase64, 'base64');
        console.log('Buffer created with length:', buffer.length);
        
        newReport.beforeImage = {
          data: buffer,
          contentType: 'image/jpeg' // Adjust if your annotated images have a different format
        };
        
        console.log('beforeImage set on the damage report');
      } else {
        console.warn('No annotatedImageBase64 found in the AI report');
        
        // Check if the AI report has an imageId with data
        if (aiReport.imageId && aiReport.imageId.data) {
          console.log('Found imageId with data, using this as beforeImage');
          newReport.beforeImage = {
            data: aiReport.imageId.data,
            contentType: aiReport.imageId.contentType || 'image/jpeg'
          };
        }
      }
    } catch (imageError) {
      console.error('Error processing image:', imageError);
    }

    console.log('Saving damage report with aiReportId:', aiReportId);
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

      console.log(`Assigning report to field worker: ${fieldWorker._id} (${fieldWorker.name})`);
      
      // Update report with assignment details
      newReport.assignedTo = fieldWorker._id;
      newReport.assignedAt = new Date();
      newReport.status = 'Assigned';
      
      // Save the report with assignment details
      await newReport.save();
      
      // Verify the status was properly saved by retrieving it again
      const savedReport = await DamageReport.findById(newReport._id);
      console.log(`Saved report status: ${savedReport.status}, assignedTo: ${savedReport.assignedTo}`);
      
      // Update any existing report caches
      try {
        await clearSingleReportCache(newReport._id.toString());
        console.log('Report cache cleared');
      } catch (cacheError) {
        console.error('Error clearing report cache:', cacheError);
      }
      
      // Send notification to field worker about new assignment
      if (fieldWorker.deviceTokens && fieldWorker.deviceTokens.length > 0) {
        try {
          // Get the latest token
          const tokens = fieldWorker.deviceTokens.map(device => device.token);
          
          // Send notification
          await NotificationService.sendPushNotifications(
            tokens,
            'New Task Assigned',
            `You have been assigned to inspect ${newReport.damageType} damage in ${newReport.location.address || newReport.region}`,
            {
              type: 'task',
              taskId: newReport._id.toString(),
              reportId: newReport.reportId,
              screenName: 'TaskDetails',
              params: { id: newReport._id.toString() }
            }
          );
        } catch (notificationError) {
          console.error('Error sending assignment notification:', notificationError);
          // Don't fail the request if notification fails
        }
      }
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
    report.status = 'Assigned'; // Set status first before saving
    
    console.log(`Updating report ${report._id} status to "Assigned" and worker to ${fieldWorker._id}`);
    
    // Save the report first to ensure status is updated
    await report.save();
    
    // Double-check that the status was saved properly
    const updatedReport = await DamageReport.findById(report._id);
    console.log(`Verification - Report status after save: ${updatedReport.status}, assignedTo: ${updatedReport.assignedTo}`);
    
    // Clear the cache for this report to ensure latest data is shown
    try {
      await clearSingleReportCache(report._id.toString());
      console.log(`Cache cleared for report ${report._id}`);
    } catch (cacheError) {
      console.error('Error clearing report cache:', cacheError);
    }
    
    // Send notification to field worker about the assignment
    if (fieldWorker.deviceTokens && fieldWorker.deviceTokens.length > 0) {
      try {
        // Get all tokens
        const tokens = fieldWorker.deviceTokens.map(device => device.token);
        
        // Send notification
        await NotificationService.sendPushNotifications(
          tokens,
          'New Assignment',
          `You have been assigned to report #${report.reportId}: ${report.damageType} damage in ${report.location.address || report.region}`,
          {
            type: 'task',
            taskId: report._id.toString(),
            reportId: report.reportId,
            screenName: 'TaskDetails',
            params: { id: report._id.toString() }
          }
        );
        
        console.log(`Notification sent to field worker ${fieldWorker.name}`);
      } catch (notificationError) {
        console.error('Error sending assignment notification:', notificationError);
        // Don't fail the request if notification fails
      }
    }

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
      return res.status(400).json({ 
        message: 'Search query is required',
        reports: [],
        fieldWorkers: [],
        analytics: [],
        repairs: [],
        query: '' 
      });
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
      error: err.message,
      reports: [],
      fieldWorkers: [],
      analytics: [],
      repairs: [],
      query: searchQuery || '' 
    });
  }
};

// Get all active field workers to assign
const getActiveFieldWorkers = async (req, res) => {
  try {
    // Apply tenant filter from middleware
    const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
    
    // Find all active field workers for the tenant
    const workers = await FieldWorker.find({ 
      ...tenantFilter,
      status: 'Active' // Only active workers
    })
      .select('name workerId specialization')
      .sort({ name: 1 });
      
    res.status(200).json(workers);
  } catch (err) {
    console.error('Error fetching active field workers:', err);
    res.status(500).json({ 
      message: 'Failed to fetch active field workers', 
      error: err.message 
    });
  }
};

// Generate a damage summary using AI
const handleGenerateDamageSummary = async (req, res) => {
  try {
    const { location, damageType, severity, priority } = req.body;
    
    // Validate required fields
    const requiredFields = ['location', 'damageType', 'severity', 'priority'];
    const missingFields = requiredFields.filter(field => !req.body[field]);
    
    if (missingFields.length > 0) {
      return res.status(400).json({
        message: 'Missing required fields',
        fields: missingFields,
        success: false
      });
    }
    
    // Call the AI utility to generate summary
    const summary = await generateDamageSummary({
      location,
      damageType,
      severity,
      priority
    });
    
    res.status(200).json({
      summary,
      success: true
    });
  } catch (err) {
    console.error('Error generating damage summary:', err);
    res.status(500).json({
      message: 'Failed to generate damage summary',
      error: err.message,
      success: false
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
  searchAllReportsAndData,
  getActiveFieldWorkers,
  generateDamageSummary: handleGenerateDamageSummary
};
