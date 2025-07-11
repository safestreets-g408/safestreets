const mongoose = require('mongoose');
const DamageReport = require('../models/DamageReport');
const FieldWorker = require('../models/FieldWorker');
const AiReport = require('../models/AiReport');
const NotificationService = require('../utils/notificationService');

// Get field worker's assigned reports
const getFieldWorkerReports = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    
    // Find reports assigned to this field worker
    const reports = await DamageReport.find({ 
      assignedTo: fieldWorkerId 
    }).sort({ assignedAt: -1 });

    res.status(200).json(reports);
  } catch (error) {
    console.error('Error fetching field worker reports:', error);
    res.status(500).json({ message: 'Error fetching reports', error: error.message });
  }
};

// Update repair status
const updateRepairStatus = async (req, res) => {
  try {
    const { reportId } = req.params;
    const { status, notes } = req.body;
    const fieldWorkerId = req.fieldWorker.id;

    // Validate status
    const validStatuses = ['pending', 'in_progress', 'completed', 'on_hold', 'cancelled'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ 
        message: 'Invalid status', 
        validStatuses 
      });
    }

    // Find the report and verify it's assigned to this field worker
    const report = await DamageReport.findOne({ 
      _id: reportId, 
      assignedTo: fieldWorkerId 
    });

    if (!report) {
      return res.status(404).json({ 
        message: 'Report not found or not assigned to you' 
      });
    }

    // Update the report
    report.repairStatus = status;
    if (notes) {
      report.repairNotes = notes;
    }

    // Update main status based on repair status
    if (status === 'in_progress') {
      report.status = 'In Progress';
    } else if (status === 'completed') {
      report.status = 'Completed';
    } else if (status === 'on_hold') {
      report.status = 'On Hold';
    } else if (status === 'cancelled') {
      report.status = 'Cancelled';
    }
    
    // Add status update to history
    if (!report.statusHistory) {
      report.statusHistory = [];
    }
    
    report.statusHistory.push({
      status,
      updatedBy: fieldWorkerId,
      updatedAt: new Date(),
      notes
    });

    // If completed, update field worker's stats
    if (status === 'completed') {
      await FieldWorker.findByIdAndUpdate(fieldWorkerId, {
        $inc: { 'profile.totalReportsHandled': 1, activeAssignments: -1 }
      });
    } else if (status === 'in_progress' && report.repairStatus === 'pending') {
      // Starting work on a pending report
      // No change to activeAssignments as it was already counted when assigned
    }

    await report.save();

    // Send notification to field worker on status change
    try {
      const fieldWorker = await FieldWorker.findById(fieldWorkerId);
      
      if (fieldWorker && fieldWorker.deviceTokens && fieldWorker.deviceTokens.length > 0) {
        // Get all tokens
        const tokens = fieldWorker.deviceTokens.map(device => device.token);
        
        // Generate appropriate message based on status
        let title = 'Report Status Updated';
        let body = '';
        
        switch (status) {
          case 'in_progress':
            body = `You've started work on report #${report.reportId}`;
            break;
          case 'completed':
            body = `Report #${report.reportId} has been marked as completed`;
            break;
          case 'on_hold':
            body = `Report #${report.reportId} has been put on hold`;
            break;
          case 'cancelled':
            body = `Report #${report.reportId} has been cancelled`;
            break;
          default:
            body = `Status for report #${report.reportId} has been updated`;
        }
        
        // Send notification
        await NotificationService.sendPushNotifications(
          tokens,
          title,
          body,
          {
            type: 'report',
            reportId: report.reportId,
            taskId: report._id.toString(),
            screenName: 'TaskDetails',
            params: { id: report._id.toString() },
            status
          }
        );
      }
    } catch (notificationError) {
      console.error('Error sending status update notification:', notificationError);
      // Don't fail the request if notification fails
    }

    res.status(200).json({ 
      message: 'Repair status updated successfully', 
      report 
    });
  } catch (error) {
    console.error('Error updating repair status:', error);
    res.status(500).json({ message: 'Error updating status', error: error.message });
  }
};

// Get field worker dashboard data
const getFieldWorkerDashboard = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    
    // Get query parameters for filtering
    const { 
      startDate, 
      endDate, 
      damageType, 
      priority, 
      status 
    } = req.query;
    
    // Build query
    const query = { assignedTo: fieldWorkerId };
    
    // Add filters if provided
    if (startDate && endDate) {
      query.createdAt = { 
        $gte: new Date(startDate), 
        $lte: new Date(endDate) 
      };
    }
    
    if (damageType) {
      query.damageType = damageType;
    }
    
    if (priority) {
      query.priority = priority;
    }
    
    if (status) {
      query.repairStatus = status;
    }
    
    // Get field worker's assigned reports
    const reports = await DamageReport.find(query);

    // Calculate statistics
    const now = new Date();
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const oneMonthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    
    // Enhanced stats with more metrics
    const stats = {
      totalAssigned: reports.length,
      reportsThisWeek: reports.filter(r => 
        new Date(r.assignedAt) >= oneWeekAgo
      ).length,
      reportsThisMonth: reports.filter(r => 
        new Date(r.assignedAt) >= oneMonthAgo
      ).length,
      repairsCompleted: reports.filter(r => 
        r.repairStatus === 'completed'
      ).length,
      pendingIssues: reports.filter(r => 
        r.repairStatus === 'pending' || r.repairStatus === 'in_progress'
      ).length,
      highPriorityIssues: reports.filter(r => 
        r.priority === 'High'
      ).length,
      completionRate: reports.length > 0 ? 
        reports.filter(r => r.repairStatus === 'completed').length / reports.length : 0,
      averageCompletionTime: calculateAverageCompletionTime(reports),
      byDamageType: calculateDamageTypeCounts(reports),
      byStatus: calculateStatusCounts(reports),
    };

    // Get recent reports for notifications, sorted by assigned date
    const recentReports = reports
      .sort((a, b) => new Date(b.assignedAt) - new Date(a.assignedAt))
      .slice(0, 5);
    
    // Get urgent reports that need attention (high priority or overdue)
    const urgentReports = reports
      .filter(r => 
        (r.priority === 'High' && r.repairStatus !== 'completed') ||
        (r.estimatedCompletionDate && new Date(r.estimatedCompletionDate) < now && r.repairStatus !== 'completed')
      )
      .slice(0, 3);

    res.status(200).json({
      stats,
      recentReports,
      urgentReports,
      fieldWorker: {
        name: req.fieldWorker.name,
        workerId: req.fieldWorker.workerId,
        specialization: req.fieldWorker.specialization,
        region: req.fieldWorker.region,
        phone: req.fieldWorker.phone,
        email: req.fieldWorker.email,
        activeAssignments: req.fieldWorker.activeAssignments || 0,
        completedReports: req.fieldWorker.profile?.totalReportsHandled || 0
      }
    });
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    res.status(500).json({ message: 'Error fetching dashboard data', error: error.message });
  }
};

// Get filtered reports
const getFilteredReports = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    
    // Get query parameters
    const { 
      status, 
      priority, 
      damageType, 
      startDate, 
      endDate,
      sortBy = 'assignedAt',
      sortOrder = 'desc',
      limit = 20,
      offset = 0
    } = req.query;
    
    // Build query
    const query = { assignedTo: fieldWorkerId };
    
    if (status) {
      query.repairStatus = status;
    }
    
    if (priority) {
      query.priority = priority;
    }
    
    if (damageType) {
      query.damageType = damageType;
    }
    
    if (startDate && endDate) {
      query.createdAt = { 
        $gte: new Date(startDate), 
        $lte: new Date(endDate) 
      };
    }
    
    // Determine sort direction
    const sortDirection = sortOrder === 'asc' ? 1 : -1;
    
    // Execute query with pagination
    const reports = await DamageReport.find(query)
      .sort({ [sortBy]: sortDirection })
      .skip(parseInt(offset))
      .limit(parseInt(limit));
    
    // Get total count for pagination
    const total = await DamageReport.countDocuments(query);
    
    res.status(200).json({
      reports,
      pagination: {
        total,
        limit: parseInt(limit),
        offset: parseInt(offset),
        hasMore: parseInt(offset) + reports.length < total
      }
    });
  } catch (error) {
    console.error('Error fetching filtered reports:', error);
    res.status(500).json({ message: 'Error fetching filtered reports', error: error.message });
  }
};

// Get task analytics
const getTaskAnalytics = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    
    // Get all reports assigned to this field worker
    const reports = await DamageReport.find({ assignedTo: fieldWorkerId });
    
    // Current time period calculations
    const now = new Date();
    const today = new Date(now.setHours(0, 0, 0, 0));
    const startOfWeek = new Date(now);
    startOfWeek.setDate(now.getDate() - now.getDay());
    startOfWeek.setHours(0, 0, 0, 0);
    const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    
    // Analytics data
    const analytics = {
      performanceOverTime: calculatePerformanceOverTime(reports),
      completionRateByDamageType: calculateCompletionRateByType(reports),
      averageCompletionTimeByPriority: calculateAvgTimeByPriority(reports),
      tasksByTimeOfDay: calculateTasksByTimeOfDay(reports),
      tasksToday: reports.filter(r => 
        new Date(r.assignedAt) >= today
      ).length,
      tasksThisWeek: reports.filter(r => 
        new Date(r.assignedAt) >= startOfWeek
      ).length,
      tasksThisMonth: reports.filter(r => 
        new Date(r.assignedAt) >= startOfMonth
      ).length
    };
    
    res.status(200).json(analytics);
  } catch (error) {
    console.error('Error fetching task analytics:', error);
    res.status(500).json({ message: 'Error fetching task analytics', error: error.message });
  }
};

// Get weekly report statistics
const getWeeklyReportStats = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    
    // Get all reports assigned to this field worker
    const reports = await DamageReport.find({ assignedTo: fieldWorkerId });
    
    // Get current date and calculate the start of the last 7 days
    const now = new Date();
    const days = [];
    
    // Create array for the last 7 days
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(now.getDate() - i);
      date.setHours(0, 0, 0, 0);
      days.push({
        date: date,
        assigned: 0,
        completed: 0
      });
    }
    
    // Calculate daily counts
    reports.forEach(report => {
      const assignedDate = new Date(report.assignedAt);
      assignedDate.setHours(0, 0, 0, 0);
      
      const completedStatusUpdate = report.statusHistory?.find(
        update => update.status === 'completed'
      );
      
      let completedDate = null;
      if (completedStatusUpdate) {
        completedDate = new Date(completedStatusUpdate.updatedAt);
        completedDate.setHours(0, 0, 0, 0);
      }
      
      // Count assigned reports
      days.forEach(day => {
        if (assignedDate.getTime() === day.date.getTime()) {
          day.assigned++;
        }
        
        // Count completed reports
        if (completedDate && completedDate.getTime() === day.date.getTime()) {
          day.completed++;
        }
      });
    });
    
    // Format for response
    const formattedDays = days.map(day => ({
      date: day.date.toISOString().split('T')[0],
      day: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][day.date.getDay()],
      assigned: day.assigned,
      completed: day.completed
    }));
    
    res.status(200).json(formattedDays);
  } catch (error) {
    console.error('Error fetching weekly report stats:', error);
    res.status(500).json({ message: 'Error fetching weekly report stats', error: error.message });
  }
};

// Get report status summary
const getReportStatusSummary = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    const { period = 'week' } = req.query;
    
    // Determine date range based on period
    const now = new Date();
    let startDate;
    
    switch (period) {
      case 'day':
        startDate = new Date(now);
        startDate.setHours(0, 0, 0, 0);
        break;
      case 'week':
        startDate = new Date(now);
        startDate.setDate(now.getDate() - 7);
        break;
      case 'month':
        startDate = new Date(now);
        startDate.setMonth(now.getMonth() - 1);
        break;
      case 'year':
        startDate = new Date(now);
        startDate.setFullYear(now.getFullYear() - 1);
        break;
      default:
        startDate = new Date(now);
        startDate.setDate(now.getDate() - 7);
    }
    
    // Get reports within the date range
    const reports = await DamageReport.find({
      assignedTo: fieldWorkerId,
      assignedAt: { $gte: startDate }
    });
    
    // Calculate status counts
    const statusCounts = {
      pending: 0,
      in_progress: 0,
      completed: 0,
      cancelled: 0
    };
    
    reports.forEach(report => {
      if (statusCounts.hasOwnProperty(report.repairStatus)) {
        statusCounts[report.repairStatus]++;
      }
    });
    
    // Calculate additional metrics
    const totalReports = reports.length;
    const summary = {
      periodLabel: getPeriodLabel(period),
      totalReports,
      statusCounts,
      percentages: {
        pending: totalReports ? ((statusCounts.pending / totalReports) * 100).toFixed(1) : 0,
        in_progress: totalReports ? ((statusCounts.in_progress / totalReports) * 100).toFixed(1) : 0,
        completed: totalReports ? ((statusCounts.completed / totalReports) * 100).toFixed(1) : 0,
        cancelled: totalReports ? ((statusCounts.cancelled / totalReports) * 100).toFixed(1) : 0
      },
      startDate: startDate.toISOString(),
      endDate: now.toISOString()
    };
    
    res.status(200).json(summary);
  } catch (error) {
    console.error('Error fetching report status summary:', error);
    res.status(500).json({ message: 'Error fetching report status summary', error: error.message });
  }
};

// Get nearby reports
const getNearbyReports = async (req, res) => {
  try {
    const { lat, lon, radius = 5 } = req.query;
    
    if (!lat || !lon) {
      return res.status(400).json({ message: 'Latitude and longitude are required' });
    }
    
    // Convert radius to meters (1km = 1000m)
    const radiusInMeters = parseInt(radius) * 1000;
    
    // Find reports near the given coordinates
    const reports = await DamageReport.find({
      assignedTo: req.fieldWorker.id,
      coordinates: {
        $near: {
          $geometry: {
            type: 'Point',
            coordinates: [parseFloat(lon), parseFloat(lat)]
          },
          $maxDistance: radiusInMeters
        }
      }
    }).limit(20);
    
    res.status(200).json(reports);
  } catch (error) {
    console.error('Error fetching nearby reports:', error);
    res.status(500).json({ message: 'Error fetching nearby reports', error: error.message });
  }
};

// Upload damage report by field worker (for new discoveries)
const uploadDamageReportByFieldWorker = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    const tenantId = req.fieldWorker.tenant;
    const { 
      damageType, 
      severity, 
      priority, 
      location, 
      description,
      predictionClass,
      annotatedImageBase64,
      imageData: frontendImageData, // Frontend sends this instead of annotatedImageBase64
      yoloDetections, // New field for YOLOv8 detections
      yoloDetectionCount // New field for YOLOv8 detection count
    } = req.body;

    console.log('Upload request received from field worker:', fieldWorkerId);
    console.log('Tenant ID from auth:', tenantId);
    console.log('Request body keys:', Object.keys(req.body));
    console.log('Has frontendImageData:', !!frontendImageData);
    console.log('Has annotatedImageBase64:', !!annotatedImageBase64);
    console.log('Has YOLOv8 detections:', !!yoloDetections);
    console.log('YOLO detection count:', yoloDetectionCount);
    console.log('Location data received:', JSON.stringify(location, null, 2));
    console.log('Location type:', typeof location);
    console.log('Description received:', description);

    // Validate required fields for AiReport schema
    if (!damageType || !severity) {
      return res.status(400).json({ 
        message: 'Missing required fields: damageType and severity are required',
        received: { 
          damageType: damageType || 'missing', 
          severity: severity || 'missing',
          predictionClass: predictionClass || 'missing',
          hasAnnotatedImageBase64: !!annotatedImageBase64,
          hasFrontendImageData: !!frontendImageData,
          annotatedImageBase64Length: annotatedImageBase64?.length || 0,
          frontendImageDataLength: frontendImageData?.length || 0
        }
      });
    }

    // Use the correct image data field - frontend sends imageData, not annotatedImageBase64
    const imageData = annotatedImageBase64 || frontendImageData || '';
    const finalPredictionClass = predictionClass || damageType;

    console.log('Using imageData length:', imageData.length);
    console.log('Final predictionClass:', finalPredictionClass);

    // Process location data properly - handle different possible structures
    let locationData = {
      coordinates: undefined,
      address: 'Location not specified'
    };

    if (location) {
      console.log('Location object structure:', JSON.stringify(location, null, 2));
      
      // Handle case where location is an object with coordinates and address
      if (typeof location === 'object' && location.coordinates && Array.isArray(location.coordinates) && location.coordinates.length === 2) {
        locationData.coordinates = location.coordinates;
        locationData.address = location.address || 'Address not available';
      }
      // Handle case where location might be a string (old format)
      else if (typeof location === 'string') {
        console.log('Location is a string, trying to parse coordinates');
        const coordMatch = location.match(/(-?\d+\.?\d*),\s*(-?\d+\.?\d*)/);
        if (coordMatch) {
          const lat = parseFloat(coordMatch[1]);
          const lng = parseFloat(coordMatch[2]);
          if (!isNaN(lat) && !isNaN(lng)) {
            locationData.coordinates = [lng, lat]; // MongoDB uses [longitude, latitude]
          }
        }
        // For string location, check if we have separate address field
        if (req.body.address) {
          locationData.address = req.body.address;
        }
      }
      // Handle case where location has an address but no coordinates
      else if (typeof location === 'object' && location.address) {
        locationData.address = location.address;
      }
    }
    
    // Fallback to separate address field if location object doesn't have address
    if (req.body.address && locationData.address === 'Location not specified') {
      locationData.address = req.body.address;
    }
    
    console.log('Processed location data:', JSON.stringify(locationData, null, 2));

    // Create a temporary image record to satisfy the imageId requirement
    const Image = require('../models/Image');
    
    // Convert base64 image to buffer if we have image data
    let imageBuffer = null;
    let contentType = 'image/jpeg'; // Default content type
    
    if (imageData && imageData.length > 0) {
      try {
        imageBuffer = Buffer.from(imageData, 'base64');
      } catch (bufferError) {
        console.warn('Failed to convert base64 to buffer:', bufferError);
        // Continue with null buffer
      }
    }
    
    const tempImage = new Image({
      tenant: tenantId,
      data: imageBuffer, // Use the correct field name 'data' instead of 'imageData'
      contentType: contentType,
      result: 'Completed'
    });
    const savedImage = await tempImage.save();

    // Process YOLOv8 detections if available
    const finalYoloDetections = Array.isArray(yoloDetections) ? yoloDetections : [];
    const finalYoloCount = yoloDetectionCount || finalYoloDetections.length || 0;

    // Create AI report with proper schema fields
    const newAiReport = new AiReport({
      imageId: savedImage._id,
      tenant: tenantId,
      predictionClass: finalPredictionClass,
      damageType,
      severity: severity.toUpperCase(),
      priority: typeof priority === 'number' ? Math.max(1, Math.min(10, priority)) : 5, // Ensure priority is 1-10
      location: locationData,
      annotatedImageBase64: imageData,
      // Add YOLOv8 fields
      yoloDetections: finalYoloDetections,
      yoloDetectionCount: finalYoloCount
    });

    const savedReport = await newAiReport.save();
    
    console.log('AI report saved successfully:', savedReport._id);

    res.status(201).json({
      message: 'AI report submitted successfully',
      report: savedReport
    });
  } catch (error) {
    console.error('Error uploading AI report:', error);
    res.status(500).json({ 
      message: 'Error uploading report', 
      error: error.message,
      details: error.errors ? Object.keys(error.errors) : undefined
    });
  }
};

// Upload after image for completed report
const uploadAfterImage = async (req, res) => {
  try {
    const { reportId } = req.params;
    const fieldWorkerId = req.fieldWorker.id;

    // Check if file was uploaded
    if (!req.file) {
      return res.status(400).json({ message: 'No image file provided' });
    }

    // Find the report and verify it's assigned to this field worker
    const report = await DamageReport.findOne({ 
      _id: reportId, 
      assignedTo: fieldWorkerId 
    });

    if (!report) {
      return res.status(404).json({ 
        message: 'Report not found or not assigned to you' 
      });
    }

    // Verify the report is completed
    if (report.repairStatus !== 'completed') {
      return res.status(400).json({ 
        message: 'Can only upload after images for completed reports' 
      });
    }

    // Update the after image
    report.afterImage = {
      data: req.file.buffer,
      contentType: req.file.mimetype
    };

    // Mark the report as resolved when after image is uploaded
    report.status = 'Resolved';
    report.resolvedAt = new Date();

    await report.save();

    res.status(200).json({ 
      message: 'After image uploaded successfully',
      reportId: report._id
    });
  } catch (error) {
    console.error('Error uploading after image:', error);
    res.status(500).json({ message: 'Error uploading after image', error: error.message });
  }
};

// Get specific damage report by ID
const getDamageReportById = async (req, res) => {
  try {
    const { reportId } = req.params;
    const fieldWorkerId = req.fieldWorker.id;

    // Find the report and verify it's assigned to this field worker
    const report = await DamageReport.findOne({ 
      _id: reportId, 
      assignedTo: fieldWorkerId 
    }).populate('assignedTo', 'name workerId specialization');

    if (!report) {
      return res.status(404).json({ 
        message: 'Report not found or not assigned to you' 
      });
    }

    // Don't send the image data in the main response for performance
    const reportData = report.toObject();
    delete reportData.beforeImage.data;
    delete reportData.afterImage?.data;

    res.status(200).json(reportData);
  } catch (error) {
    console.error('Error fetching damage report:', error);
    res.status(500).json({ message: 'Error fetching damage report', error: error.message });
  }
};

// Get AI reports assigned to field worker
const getFieldWorkerAiReports = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    
    // Get query parameters
    const { 
      status, 
      priority, 
      damageType, 
      startDate, 
      endDate,
      sortBy = 'createdAt',
      sortOrder = 'desc',
      limit = 20,
      offset = 0
    } = req.query;
    
    // Build query - AI reports might be assigned or in field worker's region
    const query = {
      $or: [
        { assignedTo: fieldWorkerId },
        { 
          // If no assignedTo field, check if it's in the field worker's region
          assignedTo: { $exists: false },
          // Assuming field worker has a region field
          region: req.fieldWorker.region 
        }
      ]
    };
    
    if (status) {
      query.status = status;
    }
    
    if (priority) {
      query.priority = { $gte: parseInt(priority) };
    }
    
    if (damageType) {
      query.damageType = damageType;
    }
    
    if (startDate && endDate) {
      query.createdAt = { 
        $gte: new Date(startDate), 
        $lte: new Date(endDate) 
      };
    }
    
    // Determine sort direction
    const sortDirection = sortOrder === 'asc' ? 1 : -1;
    
    // Execute query with pagination
    const reports = await AiReport.find(query)
      .sort({ [sortBy]: sortDirection })
      .skip(parseInt(offset))
      .limit(parseInt(limit));
    
    // Get total count for pagination
    const total = await AiReport.countDocuments(query);
    
    res.status(200).json({
      reports,
      pagination: {
        total,
        limit: parseInt(limit),
        offset: parseInt(offset),
        hasMore: parseInt(offset) + reports.length < total
      }
    });
  } catch (error) {
    console.error('Error fetching AI reports:', error);
    res.status(500).json({ message: 'Error fetching AI reports', error: error.message });
  }
};

module.exports = {
  getFieldWorkerReports,
  updateRepairStatus,
  getFieldWorkerDashboard,
  uploadDamageReportByFieldWorker,
  getFilteredReports,
  getTaskAnalytics,
  getWeeklyReportStats,
  getReportStatusSummary,
  getNearbyReports,
  uploadAfterImage,
  getDamageReportById,
  getFieldWorkerAiReports
};
