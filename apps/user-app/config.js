// API configuration
export const API_BASE_URL = 'http://192.168.13.215:5030/api'; // Update this IP to match your server

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