import { getValidAuthToken } from './auth';
import { getBaseUrl } from '../config';

// Get field worker assigned reports (tasks)
export const getFieldWorkerTasks = async () => {
  try {
    const token = await getValidAuthToken();
    const baseUrl = await getBaseUrl();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const response = await fetch(`${baseUrl}/fieldworker/damage/reports`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to fetch tasks');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Get field worker tasks error:', error);
    throw error;
  }
};

// Update task (damage report) status
export const updateTaskStatus = async (reportId, status, notes = '') => {
  try {
    const token = await getValidAuthToken();
    const baseUrl = await getBaseUrl();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const response = await fetch(`${baseUrl}/fieldworker/damage/reports/${reportId}/status`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        status, 
        notes 
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to update task status');
    }

    const data = await response.json();
    return data.report;
  } catch (error) {
    console.error('Update task status error:', error);
    throw error;
  }
};

// Upload after image for completed task
export const uploadAfterImage = async (reportId, imageUri) => {
  try {
    const token = await getValidAuthToken();
    const baseUrl = await getBaseUrl();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    // Create FormData for image upload
    const formData = new FormData();
    
    // Extract filename from URI
    const filename = imageUri.split('/').pop();
    const fileType = filename.split('.').pop();
    
    formData.append('afterImage', {
      uri: imageUri,
      type: `image/${fileType}`,
      name: filename,
    });

    const response = await fetch(`${baseUrl}/fieldworker/damage/reports/${reportId}/after-image`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to upload after image');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Upload after image error:', error);
    throw error;
  }
};

// Transform damage report to task format for UI consistency
export const transformReportToTask = (report) => {
  return {
    id: report._id,
    title: `Repair ${report.damageType.toLowerCase()} - ${report.location}`,
    description: report.description || `${report.damageType} repair needed`,
    status: report.repairStatus || 'pending',
    priority: report.priority?.toLowerCase() || 'medium',
    location: report.location,
    dueDate: report.estimatedCompletionDate || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // Default to 7 days from now
    estimatedDuration: getEstimatedDuration(report.damageType, report.severity),
    reportId: report.reportId,
    damageType: report.damageType,
    severity: report.severity,
    assignedAt: report.assignedAt,
    hasAfterImage: !!report.afterImage
  };
};

// Get estimated duration based on damage type and severity
const getEstimatedDuration = (damageType, severity) => {
  const durations = {
    'pothole': {
      'Low': '1-2 hours',
      'Medium': '2-4 hours',
      'High': '4-6 hours'
    },
    'sidewalk crack': {
      'Low': '30 minutes',
      'Medium': '1-2 hours',
      'High': '2-3 hours'
    },
    'road surface': {
      'Low': '2-3 hours',
      'Medium': '4-6 hours',
      'High': '6-8 hours'
    },
    'street light': {
      'Low': '1 hour',
      'Medium': '1-2 hours',
      'High': '2-3 hours'
    },
    'sign damage': {
      'Low': '30 minutes',
      'Medium': '1 hour',
      'High': '1-2 hours'
    }
  };

  return durations[damageType?.toLowerCase()]?.[severity] || '1-2 hours';
};
