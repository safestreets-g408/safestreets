// API configuration
// Choose the appropriate URL based on your environment:

// For iOS simulator - use localhost 
// export const API_BASE_URL = 'http://localhost:5030/api';

// For iOS simulator with explicit loopback
export const API_BASE_URL = 'http://192.168.238.1:5030/api';

// For Android emulator - use 10.0.2.2 (special Android emulator IP that maps to host's localhost)
// export const API_BASE_URL = 'http://10.0.2.2:5030/api';

// For physical devices - use your computer's local network IP
// export const API_BASE_URL = 'http://192.168.X.X:5030/api'; // Replace X.X with your actual IP

// App configuration
export const APP_VERSION = '1.0.0';
export const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB

// Theme configuration
export const COLORS = {
  primary: '#003366',
  secondary: '#2196F3',
  accent: '#03DAC6',
  background: '#F8F9FA',
  surface: '#FFFFFF',
  error: '#B00020',
  text: '#212121',
  placeholder: '#9E9E9E',
  disabled: '#BDBDBD',
};

// Storage keys
export const STORAGE_KEYS = {
  authToken: 'fieldWorkerToken',
  userData: 'fieldWorkerData',
  settings: 'appSettings'
};