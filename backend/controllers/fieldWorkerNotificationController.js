const mongoose = require('mongoose');
const DamageReport = require('../models/DamageReport');
const FieldWorker = require('../models/FieldWorker');
const axios = require('axios');
const NotificationService = require('../utils/notificationService');

// Get all notifications for a field worker
const getFieldWorkerNotifications = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    const { unreadOnly } = req.query;
    
    // Get field worker's assigned reports to generate notifications from
    const reports = await DamageReport.find({ assignedTo: fieldWorkerId })
      .sort({ assignedAt: -1 })
      .limit(30); // Limit to most recent 30 reports for performance
    
    // Generate notifications from reports
    const notifications = generateNotificationsFromReports(reports, fieldWorkerId);
    
    // Filter if unreadOnly parameter is provided
    const filteredNotifications = unreadOnly === 'true' 
      ? notifications.filter(notification => !notification.read)
      : notifications;
    
    res.status(200).json(filteredNotifications);
  } catch (error) {
    console.error('Error fetching field worker notifications:', error);
    res.status(500).json({ message: 'Error fetching notifications', error: error.message });
  }
};

// Mark a notification as read
const markNotificationAsRead = async (req, res) => {
  try {
    const { notificationId } = req.params;
    const fieldWorkerId = req.fieldWorker.id;
    
    // Since we're generating notifications dynamically, we just need to track which ones are read
    // In a real system with persisted notifications, you would update a database record
    
    // For this implementation, we'll return a success response
    // In a production app, you would actually update a notifications collection
    res.status(200).json({ 
      message: 'Notification marked as read',
      notificationId
    });
  } catch (error) {
    console.error('Error marking notification as read:', error);
    res.status(500).json({ message: 'Error marking notification as read', error: error.message });
  }
};

// Register a device token for push notifications
const registerDeviceToken = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    const { pushToken, deviceType } = req.body;
    
    if (!pushToken) {
      return res.status(400).json({ message: 'Push token is required' });
    }
    
    // Update the field worker with the new device token
    const fieldWorker = await FieldWorker.findById(fieldWorkerId);
    
    if (!fieldWorker) {
      return res.status(404).json({ message: 'Field worker not found' });
    }
    
    // Create device info object
    const deviceInfo = {
      token: pushToken,
      type: deviceType || 'unknown',
      lastRegistered: new Date()
    };
    
    // Update field worker's device tokens
    if (!fieldWorker.deviceTokens || !Array.isArray(fieldWorker.deviceTokens)) {
      fieldWorker.deviceTokens = [deviceInfo];
    } else {
      // Check if this token already exists
      const existingTokenIndex = fieldWorker.deviceTokens.findIndex(
        device => device.token === pushToken
      );
      
      if (existingTokenIndex >= 0) {
        // Update existing token
        fieldWorker.deviceTokens[existingTokenIndex] = deviceInfo;
      } else {
        // Add new token
        fieldWorker.deviceTokens.push(deviceInfo);
      }
    }
    
    await fieldWorker.save();
    
    res.status(200).json({
      success: true,
      message: 'Device token registered successfully'
    });
  } catch (error) {
    console.error('Error registering device token:', error);
    res.status(500).json({
      message: 'Error registering device token',
      error: error.message
    });
  }
};

// Send a push notification to a device
const sendPushNotification = async (token, title, body, data = {}) => {
  try {
    // Use Expo push notification service
    const message = {
      to: token,
      title,
      body,
      data,
      sound: 'default',
      badge: 1,
    };
    
    await axios.post('https://exp.host/--/api/v2/push/send', message, {
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      }
    });
    
    return true;
  } catch (error) {
    console.error('Error sending push notification:', error);
    return false;
  }
};

// Send a test notification to the field worker
const sendTestNotification = async (req, res) => {
  try {
    const fieldWorkerId = req.fieldWorker.id;
    const fieldWorker = await FieldWorker.findById(fieldWorkerId);
    
    if (!fieldWorker || !fieldWorker.deviceTokens || fieldWorker.deviceTokens.length === 0) {
      return res.status(404).json({ message: 'No device tokens registered for this field worker' });
    }
    
    // Get the latest token
    const latestToken = fieldWorker.deviceTokens[fieldWorker.deviceTokens.length - 1].token;
    
    // Send the test notification
    const success = await sendPushNotification(
      latestToken,
      'Test Notification',
      `Hello ${fieldWorker.name}, this is a test notification!`,
      { type: 'system' }
    );
    
    if (success) {
      res.status(200).json({
        success: true,
        message: 'Test notification sent successfully'
      });
    } else {
      res.status(500).json({
        message: 'Failed to send test notification'
      });
    }
  } catch (error) {
    console.error('Error sending test notification:', error);
    res.status(500).json({
      message: 'Error sending test notification',
      error: error.message
    });
  }
};

// Helper function to send notification to all devices of a field worker
const notifyFieldWorker = async (fieldWorkerId, title, body, data = {}) => {
  try {
    const fieldWorker = await FieldWorker.findById(fieldWorkerId);
    
    if (!fieldWorker || !fieldWorker.deviceTokens || fieldWorker.deviceTokens.length === 0) {
      return false;
    }
    
    // Send to all registered devices
    const promises = fieldWorker.deviceTokens.map(deviceInfo => 
      sendPushNotification(deviceInfo.token, title, body, data)
    );
    
    await Promise.all(promises);
    return true;
  } catch (error) {
    console.error(`Error notifying field worker ${fieldWorkerId}:`, error);
    return false;
  }
};

// Helper function to generate notifications from reports
const generateNotificationsFromReports = (reports, fieldWorkerId) => {
  const notifications = [];
  const now = new Date();
  
  // Recent assignments
  const recentReports = reports
    .filter(report => report.assignedAt)
    .sort((a, b) => new Date(b.assignedAt) - new Date(a.assignedAt))
    .slice(0, 5);
  
  recentReports.forEach((report) => {
    notifications.push({
      id: `assignment_${report._id}`,
      title: 'New Assignment',
      message: `You've been assigned to repair ${report.damageType} on ${report.location}`,
      time: formatTimeAgo(new Date(report.assignedAt)),
      read: false,
      type: 'info',
      icon: 'clipboard-text',
      reportId: report._id,
      createdAt: report.assignedAt
    });
  });
  
  // Completed repairs
  const completedReports = reports
    .filter(report => report.repairStatus === 'completed')
    .slice(0, 3);
  
  completedReports.forEach((report) => {
    const completedUpdate = report.statusHistory?.find(
      update => update.status === 'completed'
    );
    
    if (completedUpdate) {
      notifications.push({
        id: `completed_${report._id}`,
        title: 'Repair Completed',
        message: `Great job! Repair on ${report.location} marked as completed`,
        time: formatTimeAgo(new Date(completedUpdate.updatedAt)),
        read: true,
        type: 'success',
        icon: 'check-circle',
        reportId: report._id,
        createdAt: completedUpdate.updatedAt
      });
    }
  });
  
  // High priority assignments
  const highPriorityReports = reports
    .filter(report => 
      report.priority === 'High' && 
      report.repairStatus !== 'completed' &&
      report.repairStatus !== 'cancelled'
    )
    .slice(0, 3);
  
  highPriorityReports.forEach((report) => {
    notifications.push({
      id: `high_priority_${report._id}`,
      title: 'High Priority Task',
      message: `${report.damageType} on ${report.location} requires urgent attention`,
      time: formatTimeAgo(new Date(report.assignedAt)),
      read: false,
      type: 'warning',
      icon: 'alert-circle',
      reportId: report._id,
      createdAt: report.assignedAt
    });
  });
  
  // Approaching deadlines
  const approachingDeadlines = reports
    .filter(report => {
      if (!report.estimatedCompletionDate || report.repairStatus === 'completed' || report.repairStatus === 'cancelled') {
        return false;
      }
      
      const deadline = new Date(report.estimatedCompletionDate);
      const timeDiff = deadline - now;
      const daysDiff = timeDiff / (1000 * 60 * 60 * 24);
      
      // Within 2 days of deadline
      return daysDiff >= 0 && daysDiff <= 2;
    })
    .slice(0, 2);
  
  approachingDeadlines.forEach((report) => {
    const deadline = new Date(report.estimatedCompletionDate);
    const timeDiff = deadline - now;
    const daysDiff = Math.ceil(timeDiff / (1000 * 60 * 60 * 24));
    
    let deadlineText = '';
    if (daysDiff === 0) {
      deadlineText = 'today';
    } else if (daysDiff === 1) {
      deadlineText = 'tomorrow';
    } else {
      deadlineText = `in ${daysDiff} days`;
    }
    
    notifications.push({
      id: `deadline_${report._id}`,
      title: 'Approaching Deadline',
      message: `Repair on ${report.location} is due ${deadlineText}`,
      time: formatTimeAgo(now),
      read: false,
      type: 'warning',
      icon: 'clock-alert',
      reportId: report._id,
      createdAt: now
    });
  });
  
  // Sort notifications by creation date
  return notifications
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
};

// Helper function to format time ago
const formatTimeAgo = (date) => {
  const now = new Date();
  const diffInMinutes = Math.floor((now - date) / (1000 * 60));
  
  if (diffInMinutes < 60) {
    return `${diffInMinutes} minutes ago`;
  } else if (diffInMinutes < 1440) {
    const hours = Math.floor(diffInMinutes / 60);
    return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  } else {
    const days = Math.floor(diffInMinutes / 1440);
    return `${days} day${days > 1 ? 's' : ''} ago`;
  }
};

module.exports = {
  getFieldWorkerNotifications,
  markNotificationAsRead,
  registerDeviceToken,
  sendTestNotification
};
