import { getValidAuthToken } from './auth';
import { API_BASE_URL } from '../config';

export const getUserReports = async (options = {}) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    // Build query parameters
    const params = new URLSearchParams();
    if (options.page) params.append('page', options.page);
    if (options.limit) params.append('limit', options.limit);
    if (options.sortBy) params.append('sortBy', options.sortBy);
    if (options.sortOrder) params.append('sortOrder', options.sortOrder);
    
    const queryString = params.toString() ? `?${params.toString()}` : '';
    const url = `${API_BASE_URL}/fieldworker/damage/reports${queryString}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    // Handle error response
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to fetch reports');
    }

    // Parse and return successful response
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Get user reports error:', error);
    throw error;
  }
};

export const getFilteredUserReports = async (filters = {}) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    // Build query parameters from filters
    const params = new URLSearchParams();
    Object.keys(filters).forEach(key => {
      if (filters[key] !== undefined && filters[key] !== null) {
        params.append(key, filters[key]);
      }
    });

    const queryString = params.toString() ? `?${params.toString()}` : '';
    const url = `${API_BASE_URL}/fieldworker/damage/reports/filtered${queryString}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    // Handle error response
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to fetch filtered reports');
    }

    // Parse and return successful response
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Get filtered user reports error:', error);
    throw error;
  }
};

export const getReportDetails = async (reportId) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const url = `${API_BASE_URL}/fieldworker/damage/reports/${reportId}`;
    console.log('Fetching report details from:', url);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', response.headers);

    // Handle error response
    if (!response.ok) {
      let errorMessage = `HTTP Error: ${response.status}`;
      
      try {
        // Try to parse error response as JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } else {
          // If not JSON, get text response
          const errorText = await response.text();
          console.error('Non-JSON error response:', errorText.substring(0, 200));
          errorMessage = `Server returned ${response.status}: ${errorText.substring(0, 100)}`;
        }
      } catch (parseError) {
        console.error('Error parsing error response:', parseError);
        errorMessage = `Failed to parse error response (${response.status})`;
      }
      
      throw new Error(errorMessage);
    }

    // Parse successful response
    try {
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const textResponse = await response.text();
        console.error('Expected JSON but got:', contentType, textResponse.substring(0, 200));
        throw new Error('Server returned non-JSON content');
      }

      const data = await response.json();
      console.log('Successfully parsed response data:', data);
      return data;
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      // Try to get raw response for debugging
      try {
        const rawText = await response.text();
        console.error('Raw response text:', rawText.substring(0, 500));
      } catch (textError) {
        console.error('Could not get raw response text:', textError);
      }
      throw new Error('Failed to parse server response as JSON');
    }
  } catch (error) {
    console.error('Get report details error:', error);
    throw error;
  }
};

export const updateReportStatus = async (reportId, status, notes = '') => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const url = `${API_BASE_URL}/fieldworker/damage/reports/${reportId}/status`;

    const response = await fetch(url, {
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

    // Handle error response
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to update report status');
    }

    // Parse and return successful response
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Update report status error:', error);
    throw error;
  }
};

export const getReportById = async (reportId) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    console.log('Fetching report with token:', token.substring(0, 20) + '...');
    
    // Try first endpoint format
    let response = await fetch(`${API_BASE_URL}/fieldworker/damage/report/${reportId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });

    // If first endpoint doesn't work, try alternative endpoint format
    if (response.status === 404) {
      response = await fetch(`${API_BASE_URL}/fieldworker/damage/reports/${reportId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
    }

    // Check if the response is OK
    if (!response.ok) {
      // Try to parse error as JSON, but handle non-JSON responses
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const errorData = await response.json();
        throw new Error(errorData.message || `Server error: ${response.status}`);
      } else {
        // For non-JSON responses (like HTML error pages)
        const errorText = await response.text();
        console.error('Non-JSON error response:', errorText.substring(0, 200));
        throw new Error(`Server returned non-JSON response: ${response.status}`);
      }
    }

    // Check content type before parsing
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      const textResponse = await response.text();
      console.error('Expected JSON but got:', contentType, textResponse.substring(0, 200));
      throw new Error('Server returned non-JSON content');
    }

    // Parse response as JSON with error handling
    let data;
    try {
      const text = await response.text();
      data = JSON.parse(text);
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      throw new Error('Failed to parse server response as JSON');
    }
    
    // Ensure consistency in ID fields
    if (data && !data.id && data._id) {
      data.id = data._id;
    }
    
    return data;
  } catch (error) {
    console.error('Fetch report details error:', error);
    throw error;
  }
};