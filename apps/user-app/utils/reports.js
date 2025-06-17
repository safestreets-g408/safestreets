 import { getAuthToken } from './auth';
import { API_BASE_URL } from '../config';

// Fetch user reports
export const fetchReports = async (params = {}) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    // Convert params object to query string
    const queryParams = new URLSearchParams();
    Object.keys(params).forEach(key => {
      if (params[key] !== null && params[key] !== undefined) {
        queryParams.append(key, params[key]);
      }
    });
    
    const queryString = queryParams.toString();
    const url = `${API_BASE_URL}/fieldworker/reports${queryString ? `?${queryString}` : ''}`;
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch reports');
    }
    
    return data.reports;
  } catch (error) {
    console.error('Error fetching reports:', error);
    throw error;
  }
};

// Fetch single report details
export const fetchReportDetails = async (reportId) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${API_BASE_URL}/fieldworker/reports/${reportId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch report details');
    }
    
    return data.report;
  } catch (error) {
    console.error('Error fetching report details:', error);
    throw error;
  }
};

// Update report status
export const updateReportStatus = async (reportId, status, notes = '') => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${API_BASE_URL}/fieldworker/reports/${reportId}/status`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ status, notes }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to update report status');
    }
    
    return data.report;
  } catch (error) {
    console.error('Error updating report status:', error);
    throw error;
  }
};

// Submit new report
export const submitNewReport = async (reportData, imageUri) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const formData = new FormData();
    
    // Add report data
    Object.keys(reportData).forEach(key => {
      formData.append(key, reportData[key]);
    });
    
    // Add image if available
    if (imageUri) {
      // Get file name from URI
      const uriParts = imageUri.split('.');
      const fileType = uriParts[uriParts.length - 1];
      
      formData.append('image', {
        uri: imageUri,
        name: `photo.${fileType}`,
        type: `image/${fileType}`,
      });
    }
    
    const response = await fetch(`${API_BASE_URL}/fieldworker/reports`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to submit report');
    }
    
    return data.report;
  } catch (error) {
    console.error('Error submitting report:', error);
    throw error;
  }
};
