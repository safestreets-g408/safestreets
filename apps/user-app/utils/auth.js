import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = 'http://192.168.13.215:5030/api'; // Updated to use machine IP for Expo development
// For production, use your actual backend URL:
// const API_BASE_URL = 'http://your-production-server.com/api';

// Auth token management
export const storeAuthToken = async (token) => {
  try {
    await AsyncStorage.setItem('fieldWorkerToken', token);
  } catch (error) {
    console.error('Error storing auth token:', error);
  }
};

export const getAuthToken = async () => {
  try {
    return await AsyncStorage.getItem('fieldWorkerToken');
  } catch (error) {
    console.error('Error getting auth token:', error);
    return null;
  }
};

export const removeAuthToken = async () => {
  try {
    await AsyncStorage.removeItem('fieldWorkerToken');
  } catch (error) {
    console.error('Error removing auth token:', error);
  }
};

// Store field worker data
export const storeFieldWorkerData = async (fieldWorkerData) => {
  try {
    await AsyncStorage.setItem('fieldWorkerData', JSON.stringify(fieldWorkerData));
  } catch (error) {
    console.error('Error storing field worker data:', error);
  }
};

export const getFieldWorkerData = async () => {
  try {
    const data = await AsyncStorage.getItem('fieldWorkerData');
    return data ? JSON.parse(data) : null;
  } catch (error) {
    console.error('Error getting field worker data:', error);
    return null;
  }
};

export const removeFieldWorkerData = async () => {
  try {
    await AsyncStorage.removeItem('fieldWorkerData');
  } catch (error) {
    console.error('Error removing field worker data:', error);
  }
};

// API calls
export const loginFieldWorker = async (email, password) => {
  try {
    const response = await fetch(`${API_BASE_URL}/fieldworker/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Login failed');
    }

    // Store token and field worker data
    await storeAuthToken(data.token);
    await storeFieldWorkerData(data.fieldWorker);

    return data;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

export const getFieldWorkerProfile = async () => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/auth/profile`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get profile');
    }

    return data.fieldWorker;
  } catch (error) {
    console.error('Get profile error:', error);
    throw error;
  }
};

export const updateFieldWorkerProfile = async (profileData) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/auth/profile`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(profileData),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to update profile');
    }

    // Update stored field worker data
    await storeFieldWorkerData(data.fieldWorker);

    return data.fieldWorker;
  } catch (error) {
    console.error('Update profile error:', error);
    throw error;
  }
};

export const logout = async () => {
  try {
    await removeAuthToken();
    await removeFieldWorkerData();
  } catch (error) {
    console.error('Logout error:', error);
  }
};

// Dashboard API calls
export const getDashboardStats = async () => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/damage/dashboard`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get dashboard stats');
    }

    return data;
  } catch (error) {
    console.error('Get dashboard stats error:', error);
    throw error;
  }
};

export const getFieldWorkerReports = async () => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/damage/reports`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get field worker reports');
    }

    return data;
  } catch (error) {
    console.error('Get field worker reports error:', error);
    throw error;
  }
};

export const updateRepairStatus = async (reportId, status, notes) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/damage/reports/${reportId}/status`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ status, notes }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to update repair status');
    }

    return data;
  } catch (error) {
    console.error('Update repair status error:', error);
    throw error;
  }
};

export const uploadDamageReport = async (reportData, imageUri) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const formData = new FormData();
    
    // Add image
    if (imageUri) {
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'damage_report.jpg',
      });
    }
    
    // Add other data
    Object.keys(reportData).forEach(key => {
      formData.append(key, reportData[key]);
    });

    const response = await fetch(`${API_BASE_URL}/damage/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to upload damage report');
    }

    return data;
  } catch (error) {
    console.error('Upload damage report error:', error);
    throw error;
  }
};

export const getReportById = async (reportId) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/damage/report/${reportId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get report');
    }

    return data;
  } catch (error) {
    console.error('Get report error:', error);
    throw error;
  }
};
