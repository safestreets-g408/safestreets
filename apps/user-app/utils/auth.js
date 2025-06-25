import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL, getBaseUrl } from '../config';
import { Alert, Platform } from 'react-native';

// Utility function to check network connectivity
export const checkNetworkConnectivity = async () => {
  try {
    // Try a lightweight fetch to a reliable endpoint
    const response = await fetch('https://www.google.com', { 
      method: 'HEAD',
      timeout: 5000 
    });
    return response.ok;
  } catch (error) {
    console.log('Network connectivity test failed:', error);
    return false;
  }
};

// Utility function to show network error
export const showNetworkError = (message = 'Network connection error') => {
  Alert.alert(
    'Connection Error',
    message,
    [{ text: 'OK', onPress: () => console.log('OK Pressed') }]
  );
};

export const storeAuthToken = async (token) => {
  try {
    await AsyncStorage.setItem('fieldWorkerToken', token);
  } catch (error) {
    console.error('Error storing auth token:', error);
  }
};

export const getAuthToken = async () => {
  try {
    const token = await AsyncStorage.getItem('fieldWorkerToken');
    console.log('Retrieved token from storage:', token); // Add logging
    return token;
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
    // Get the latest API URL (in case it was updated)
    const currentApiUrl = API_BASE_URL;
    console.log(`API URL being used:`, currentApiUrl);
    
    // Create an AbortController to handle request timeouts
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      console.error('Login request timed out after 20 seconds');
    }, 20000); // Increased timeout to 20 seconds
    
    const loginUrl = `${currentApiUrl}/fieldworker/auth/login`;
    console.log(`Attempting to login with API URL: ${loginUrl}`);
    console.log(`Sending POST request to: ${loginUrl}`);
    
    // Test if network is available first
    let networkAvailable = false;
    try {
      const networkCheck = await fetch('https://www.google.com', { method: 'HEAD', timeout: 5000 })
        .then(res => res.ok)
        .catch(() => false);
      
      networkAvailable = networkCheck;
      if (!networkAvailable) {
        console.log('No internet connection detected');
        throw new Error('No internet connection. Please check your network settings.');
      }
    } catch (netError) {
      // Continue anyway - server might be reachable even if google.com isn't
      console.log('Network check error, continuing anyway:', netError.message);
    }
    
    // Make the actual login request
    const response = await fetch(loginUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ email, password }),
      signal: controller.signal
    }).catch(fetchError => {
      console.error('Fetch error details:', fetchError);
      
      // More detailed error message with API URL info
      let errorMessage = `Network request failed: ${fetchError.message}`;
      
      if (loginUrl.includes('localhost') || loginUrl.includes('127.0.0.1')) {
        errorMessage += `\n\nYou're using localhost (${loginUrl}), which doesn't work with Expo Go on physical devices. Please use the Settings screen to change the API URL to your computer's actual IP address.`;
      } else {
        errorMessage += `\n\nServer URL: ${loginUrl}`;
      }
      
      throw new Error(errorMessage);
    });
    
    // Clear the timeout since the request completed
    clearTimeout(timeoutId);
    
    console.log('Login response status:', response.status);
    
    // Check if the response is ok before trying to parse JSON
    if (!response.ok) {
      let errorMessage = '';
      try {
        const errorText = await response.text();
        console.error('Login error response:', errorText);
        try {
          // Try to parse as JSON
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.message || errorText;
        } catch {
          // If not JSON, use the raw text
          errorMessage = errorText;
        }
      } catch (e) {
        errorMessage = 'Unknown server error';
      }
      throw new Error(`Login failed with status ${response.status}: ${errorMessage}`);
    }

    const data = await response.json();
    console.log('Login successful, storing credentials');

    // Store token and field worker data
    await storeAuthToken(data.token);
    await storeFieldWorkerData(data.fieldWorker);

    return data;
  } catch (error) {
    console.error('Login error:', error);
    
    // Add a link to the settings screen in error messages
    if (error.message.includes('Network request failed') || 
        error.message.includes('connect') || 
        error.message.includes('ECONNREFUSED') ||
        error.message.includes('timed out')) {
      
      throw new Error(
        `Cannot connect to server. Please check your internet connection and that the server is running.\n\nTry changing the API URL in Settings.`
      );
    }
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

// Token refresh functionality
export const refreshAuthToken = async () => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      console.log('No auth token found for refresh');
      return null;
    }

    // First check network connectivity
    const isConnected = await checkNetworkConnectivity();
    if (!isConnected) {
      console.log('No network connectivity, cannot refresh token');
      return token; // Return existing token if no connectivity
    }

    try {
      const response = await fetch(`${API_BASE_URL}/fieldworker/auth/refresh-token`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        // Add timeout to prevent long waits
        timeout: 10000
      });

      if (!response.ok) {
        // If token refresh fails with auth error, clear tokens
        if (response.status === 401 || response.status === 403) {
          console.log('Auth error during token refresh, clearing tokens');
          await removeAuthToken();
          await removeFieldWorkerData();
          return null;
        }
        
        // For other errors, log but return existing token
        console.log(`Token refresh failed with status: ${response.status}`);
        return token;
      }

      const data = await response.json();

      // Store new token
      if (data && data.token) {
        console.log('Received and storing new token');
        await storeAuthToken(data.token);
        return data.token;
      } else {
        console.log('No token in refresh response');
        return token;
      }
    } catch (fetchError) {
      console.error('Fetch error during token refresh:', fetchError);
      return token; // Return existing token on network errors
    }
  } catch (error) {
    console.error('Token refresh error:', error);
    return null;
  }
};

// Enhanced auth token getter with built-in refresh attempt
export const getValidAuthToken = async () => {
  try {
    // Simply return stored token without pre-emptive refresh
    const token = await getAuthToken();
    if (!token) {
      console.log('No token found in storage');
      return null;
    }
    return token;
  } catch (error) {
    console.error('Error getting valid auth token:', error);
    return null;
  }
};
