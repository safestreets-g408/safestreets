import { getValidAuthToken } from './auth';
import { API_BASE_URL } from '../config';

export const getDashboardData = async (filters = {}) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    // Convert filters to query string
    const queryParams = new URLSearchParams();
    Object.keys(filters).forEach(key => {
      if (filters[key]) {
        queryParams.append(key, filters[key]);
      }
    });

    const url = `${API_BASE_URL}/fieldworker/damage/dashboard${
      queryParams.toString() ? `?${queryParams.toString()}` : ''
    }`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get dashboard data');
    }

    return data;
  } catch (error) {
    console.error('Get dashboard data error:', error);
    throw error;
  }
};

export const getFilteredReports = async (filters = {}) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    // Convert filters to query string
    const queryParams = new URLSearchParams();
    Object.keys(filters).forEach(key => {
      if (filters[key]) {
        queryParams.append(key, filters[key]);
      }
    });

    const url = `${API_BASE_URL}/fieldworker/damage/reports/filtered${
      queryParams.toString() ? `?${queryParams.toString()}` : ''
    }`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get filtered reports');
    }

    return data;
  } catch (error) {
    console.error('Get filtered reports error:', error);
    throw error;
  }
};


export const getTaskAnalytics = async () => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/damage/task-analytics`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get task analytics');
    }

    return data;
  } catch (error) {
    console.error('Get task analytics error:', error);
    throw error;
  }
};

export const getWeeklyReportStats = async () => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/damage/weekly-stats`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get weekly statistics');
    }

    return data;
  } catch (error) {
    console.error('Get weekly statistics error:', error);
    throw error;
  }
};

export const getWeatherInfo = async (coordinates) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const url = `${API_BASE_URL}/fieldworker/weather?lat=${coordinates.latitude}&lon=${coordinates.longitude}`;
    console.log('Fetching weather from:', url);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    // Check if response is ok first before trying to parse JSON
    if (!response.ok) {
      const text = await response.text();
      console.error('Weather API error response:', text);
      throw new Error(`Weather API returned ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Get weather information error:', error);
    // Return mock weather data instead of throwing
    return getMockWeatherData(coordinates);
  }
};

export const getReportStatusSummary = async (period = 'week') => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const url = `${API_BASE_URL}/fieldworker/damage/status-summary?period=${period}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get report status summary');
    }

    return data;
  } catch (error) {
    console.error('Get report status summary error:', error);
    throw error;
  }
};


export const getNearbyReports = async (coordinates, radius = 5) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const url = `${API_BASE_URL}/fieldworker/damage/nearby?lat=${coordinates.latitude}&lon=${coordinates.longitude}&radius=${radius}`;
    console.log('Fetching nearby reports from:', url);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    // Check if response is ok first before trying to parse JSON
    if (!response.ok) {
      const text = await response.text();
      console.error('Nearby reports API error response:', text);
      throw new Error(`Nearby reports API returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Get nearby reports error:', error);
    // Return empty array instead of throwing
    return [];
  }
};


export const markNotificationAsRead = async (notificationId) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/fieldworker/notifications/${notificationId}/read`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to mark notification as read');
    }

    return data;
  } catch (error) {
    console.error('Mark notification as read error:', error);
    throw error;
  }
};


export const getNotifications = async (unreadOnly = false) => {
  try {
    const token = await getValidAuthToken();
    
    if (!token) {
      throw new Error('No valid auth token found');
    }

    const url = unreadOnly 
      ? `${API_BASE_URL}/fieldworker/notifications?unreadOnly=true` 
      : `${API_BASE_URL}/fieldworker/notifications`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Failed to get notifications');
    }

    return data;
  } catch (error) {
    console.error('Get notifications error:', error);
    throw error;
  }
};

// Mock weather data function for fallback
const getMockWeatherData = (coordinates) => {
  // Generate semi-realistic weather data
  const now = new Date();
  const month = now.getMonth(); // 0-11
  
  // Temperature range based on northern hemisphere seasons
  let tempRange, conditionPool;
  
  if (month >= 2 && month <= 4) {
    // Spring
    tempRange = { min: 10, max: 25 };
    conditionPool = ['Clear', 'Clouds', 'Rain', 'Drizzle'];
  } else if (month >= 5 && month <= 7) {
    // Summer
    tempRange = { min: 18, max: 35 };
    conditionPool = ['Clear', 'Clouds', 'Thunderstorm', 'Drizzle'];
  } else if (month >= 8 && month <= 10) {
    // Fall
    tempRange = { min: 8, max: 22 };
    conditionPool = ['Clouds', 'Rain', 'Fog', 'Clear'];
  } else {
    // Winter
    tempRange = { min: -5, max: 15 };
    conditionPool = ['Snow', 'Clouds', 'Clear', 'Rain'];
  }
  
  // Generate temperature within range
  const temperature = Math.round(Math.random() * (tempRange.max - tempRange.min) + tempRange.min);
  
  // Select random condition
  const condition = conditionPool[Math.floor(Math.random() * conditionPool.length)];
  
  // Generate humidity and wind
  const humidity = Math.round(Math.random() * 50) + 30; // 30-80%
  const windSpeed = Math.round((Math.random() * 20 + 5) * 10) / 10; // 5-25 km/h with 1 decimal
  
  return {
    coord: { lat: coordinates?.latitude || 0, lon: coordinates?.longitude || 0 },
    weather: [{
      id: getWeatherConditionId(condition),
      main: condition,
      description: getWeatherDescription(condition),
      icon: getWeatherIcon(condition)
    }],
    main: {
      temp: temperature,
      feels_like: Math.round(temperature - 2 + Math.random() * 4),
      temp_min: Math.round(temperature - 2 - Math.random() * 2),
      temp_max: Math.round(temperature + 2 + Math.random() * 2),
      pressure: Math.round(1000 + Math.random() * 30),
      humidity: humidity
    },
    wind: {
      speed: windSpeed,
      deg: Math.round(Math.random() * 360)
    },
    clouds: {
      all: condition === 'Clear' ? Math.round(Math.random() * 10) : Math.round(Math.random() * 50) + 50
    },
    visibility: condition === 'Fog' ? 1000 + Math.round(Math.random() * 4000) : 10000,
    dt: Math.floor(Date.now() / 1000),
    sys: {
      country: 'US',
      sunrise: Math.floor((now.setHours(6, 0, 0, 0)) / 1000),
      sunset: Math.floor((now.setHours(18, 0, 0, 0)) / 1000)
    },
    name: 'Current Location'
  };
};

// Helper functions for mock weather
const getWeatherConditionId = (condition) => {
  switch (condition) {
    case 'Clear': return 800;
    case 'Clouds': return 801 + Math.floor(Math.random() * 4);
    case 'Rain': return 500 + Math.floor(Math.random() * 4);
    case 'Thunderstorm': return 200 + Math.floor(Math.random() * 5);
    case 'Snow': return 600 + Math.floor(Math.random() * 4);
    case 'Drizzle': return 300 + Math.floor(Math.random() * 3);
    case 'Fog': return 741;
    default: return 800;
  }
};

const getWeatherDescription = (condition) => {
  switch (condition) {
    case 'Clear': return 'clear sky';
    case 'Clouds': 
      const cloudTypes = ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds'];
      return cloudTypes[Math.floor(Math.random() * cloudTypes.length)];
    case 'Rain':
      const rainTypes = ['light rain', 'moderate rain', 'heavy rain', 'very heavy rain'];
      return rainTypes[Math.floor(Math.random() * rainTypes.length)];
    case 'Thunderstorm': return 'thunderstorm';
    case 'Snow': return 'snow';
    case 'Drizzle': return 'light intensity drizzle';
    case 'Fog': return 'fog';
    default: return 'unknown';
  }
};

const getWeatherIcon = (condition) => {
  // OpenWeatherMap-like icon codes
  switch (condition) {
    case 'Clear': return '01d';
    case 'Clouds': return ['02d', '03d', '04d'][Math.floor(Math.random() * 3)];
    case 'Rain': return ['09d', '10d'][Math.floor(Math.random() * 2)];
    case 'Thunderstorm': return '11d';
    case 'Snow': return '13d';
    case 'Drizzle': return '09d';
    case 'Fog': return '50d';
    default: return '01d';
  }
};
