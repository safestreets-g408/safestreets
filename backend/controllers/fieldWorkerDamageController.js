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
    
    // Get field worker's assigned reports
    const reports = await DamageReport.find({ 
      assignedTo: fieldWorkerId 
    });

    // Calculate statistics
    const now = new Date();
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    
    const stats = {
      totalAssigned: reports.length,
      reportsThisWeek: reports.filter(r => 
        new Date(r.assignedAt) >= oneWeekAgo
      ).length,
      repairsCompleted: reports.filter(r => 
        r.repairStatus === 'completed'
      ).length,
      pendingIssues: reports.filter(r => 
        r.repairStatus === 'pending' || r.repairStatus === 'in_progress'
      ).length,
      completionRate: reports.length > 0 ? 
        reports.filter(r => r.repairStatus === 'completed').length / reports.length : 0
    };

    // Get recent reports for notifications
    const recentReports = reports
      .sort((a, b) => new Date(b.assignedAt) - new Date(a.assignedAt))
      .slice(0, 5);

    res.status(200).json({
      stats,
      recentReports,
      fieldWorker: {
        name: req.fieldWorker.name,
        workerId: req.fieldWorker.workerId,
        specialization: req.fieldWorker.specialization,
        region: req.fieldWorker.region
      }
    });
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    res.status(500).json({ message: 'Error fetching dashboard data', error: error.message });
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
  uploadDamageReportByFieldWorker
};
