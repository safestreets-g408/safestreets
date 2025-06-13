// Layout
export const DRAWER_WIDTH = 280;

// API
export const API_BASE_URL = 'http://localhost:5030/api';
export const API_ENDPOINTS = {
  DAMAGE_REPORTS: '/damage/reports',
  REPAIRS: '/repairs',
  ANALYTICS: '/analytics',
  AUTH: '/auth',
  USERS: '/users',
  FIELD_WORKERS: '/field/workers',
  IMAGES: '/images',
};

// Auth
export const TOKEN_KEY = 'auth_token';
export const USER_KEY = 'user_data';

// Date formats
export const DATE_FORMAT = 'yyyy-MM-dd';
export const DATE_TIME_FORMAT = 'yyyy-MM-dd HH:mm:ss';

// Pagination
export const DEFAULT_PAGE_SIZE = 10;
export const PAGE_SIZE_OPTIONS = [5, 10, 25, 50];

// Chart colors
export const CHART_COLORS = {
  primary: '#2196f3',
  secondary: '#f50057',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
  info: '#00bcd4'
}; 