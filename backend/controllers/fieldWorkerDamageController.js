const mongoose = require('mongoose');
const DamageReport = require('../models/DamageReport');
const FieldWorker = require('../models/FieldWorker');

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
    const validStatuses = ['pending', 'in_progress', 'completed', 'cancelled'];
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

// Helper functions
const calculateAverageCompletionTime = (reports) => {
  const completedReports = reports.filter(report => {
    // Find 'completed' status in history
    const completedUpdate = report.statusHistory?.find(
      update => update.status === 'completed'
    );
    
    return completedUpdate && report.assignedAt;
  });
  
  if (completedReports.length === 0) return 0;
  
  // Calculate average time to completion in hours
  const totalHours = completedReports.reduce((sum, report) => {
    const assignedAt = new Date(report.assignedAt);
    
    const completedUpdate = report.statusHistory.find(
      update => update.status === 'completed'
    );
    
    const completedAt = new Date(completedUpdate.updatedAt);
    
    const hoursToComplete = (completedAt - assignedAt) / (1000 * 60 * 60);
    return sum + hoursToComplete;
  }, 0);
  
  return totalHours / completedReports.length;
};

const calculateDamageTypeCounts = (reports) => {
  const typeCounts = {};
  
  reports.forEach(report => {
    const type = report.damageType || 'Unknown';
    typeCounts[type] = (typeCounts[type] || 0) + 1;
  });
  
  return typeCounts;
};

const calculateStatusCounts = (reports) => {
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
  
  return statusCounts;
};

const calculatePerformanceOverTime = (reports) => {
  // Group reports by month
  const reportsByMonth = {};
  
  reports.forEach(report => {
    const assignedDate = new Date(report.assignedAt);
    const monthKey = `${assignedDate.getFullYear()}-${assignedDate.getMonth() + 1}`;
    
    if (!reportsByMonth[monthKey]) {
      reportsByMonth[monthKey] = {
        assigned: 0,
        completed: 0,
        month: assignedDate.toLocaleString('default', { month: 'short' }),
        year: assignedDate.getFullYear()
      };
    }
    
    reportsByMonth[monthKey].assigned++;
    
    if (report.repairStatus === 'completed') {
      reportsByMonth[monthKey].completed++;
    }
  });
  
  // Convert to array and sort by date
  return Object.values(reportsByMonth)
    .sort((a, b) => {
      if (a.year !== b.year) return a.year - b.year;
      return a.month.localeCompare(b.month);
    })
    .slice(-6); // Last 6 months
};

const calculateCompletionRateByType = (reports) => {
  const reportsByType = {};
  
  reports.forEach(report => {
    const type = report.damageType || 'Unknown';
    
    if (!reportsByType[type]) {
      reportsByType[type] = { total: 0, completed: 0 };
    }
    
    reportsByType[type].total++;
    
    if (report.repairStatus === 'completed') {
      reportsByType[type].completed++;
    }
  });
  
  // Calculate completion rate for each type
  Object.keys(reportsByType).forEach(type => {
    const { total, completed } = reportsByType[type];
    reportsByType[type].rate = total > 0 ? (completed / total) * 100 : 0;
  });
  
  return reportsByType;
};

const calculateAvgTimeByPriority = (reports) => {
  const reportsByPriority = {
    Low: { total: 0, totalHours: 0 },
    Medium: { total: 0, totalHours: 0 },
    High: { total: 0, totalHours: 0 }
  };
  
  reports.forEach(report => {
    if (!report.assignedAt || report.repairStatus !== 'completed') return;
    
    const priority = report.priority || 'Medium';
    if (!reportsByPriority[priority]) return;
    
    const completedUpdate = report.statusHistory?.find(
      update => update.status === 'completed'
    );
    
    if (!completedUpdate) return;
    
    const assignedAt = new Date(report.assignedAt);
    const completedAt = new Date(completedUpdate.updatedAt);
    const hoursToComplete = (completedAt - assignedAt) / (1000 * 60 * 60);
    
    reportsByPriority[priority].total++;
    reportsByPriority[priority].totalHours += hoursToComplete;
  });
  
  // Calculate average for each priority
  Object.keys(reportsByPriority).forEach(priority => {
    const { total, totalHours } = reportsByPriority[priority];
    reportsByPriority[priority].avgHours = total > 0 ? totalHours / total : 0;
  });
  
  return reportsByPriority;
};

const calculateTasksByTimeOfDay = (reports) => {
  const hours = Array(24).fill(0);
  
  reports.forEach(report => {
    if (!report.assignedAt) return;
    
    const assignedAt = new Date(report.assignedAt);
    const hour = assignedAt.getHours();
    
    hours[hour]++;
  });
  
  return hours.map((count, hour) => ({
    hour,
    count,
    label: `${hour % 12 || 12} ${hour < 12 ? 'AM' : 'PM'}`
  }));
};

const getPeriodLabel = (period) => {
  switch (period) {
    case 'day': return 'Today';
    case 'week': return 'This Week';
    case 'month': return 'This Month';
    case 'year': return 'This Year';
    default: return 'This Week';
  }
};

// Upload damage report by field worker (for new discoveries)
const uploadDamageReportByFieldWorker = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    const { 
      damageType, 
      severity, 
      priority, 
      location, 
      description,
      coordinates 
    } = req.body;

    // Validate required fields
    const requiredFields = ['damageType', 'severity', 'location'];
    const missingFields = requiredFields.filter(field => !req.body[field]);
    
    if (missingFields.length > 0) {
      return res.status(400).json({ 
        message: 'Missing required fields', 
        fields: missingFields 
      });
    }

    const reportId = 'FW-' + Date.now();
    
    const newReport = new DamageReport({
      reportId,
      damageType,
      severity,
      priority: priority || 'Medium',
      region: req.fieldWorker.region,
      location,
      description: description || '',
      reporter: `${req.fieldWorker.name} (Field Worker)`,
      reporterType: 'field_worker',
      reportedBy: fieldWorkerId,
      status: 'Reported by Field Worker',
      coordinates: coordinates || null,
      createdAt: new Date()
    });

    await newReport.save();

    res.status(201).json({
      message: 'Damage report submitted successfully',
      report: newReport
    });
  } catch (error) {
    console.error('Error uploading damage report:', error);
    res.status(500).json({ message: 'Error uploading report', error: error.message });
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
  getNearbyReports
};
