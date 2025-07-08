import axios from 'axios';
import { TOKEN_KEY, API_BASE_URL } from '../config/constants';

// Create a central axios instance to use across the app
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Redirect function - will be set by auth provider
let redirectToLogin = () => {
  console.error('Redirect function not set');
  // Fallback if redirect function wasn't set
  window.location.href = '/login';
};

// Function to set redirect handler
export const setAuthRedirect = (redirectFn) => {
  redirectToLogin = redirectFn;
};

// Add auth token to all requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Check if error is due to authentication issue
    if (error.response?.status === 401) {
      console.warn('Authentication error detected - clearing auth and redirecting to login');
      
      // Clear authentication data
      localStorage.removeItem(TOKEN_KEY);
      
      // Redirect to login page
      redirectToLogin();
      
      // Return custom error for handling in components
      return Promise.reject({
        isAuthError: true,
        message: 'Your session has expired. Please log in again.',
        originalError: error
      });
    }
    
    return Promise.reject(error);
  }
);

// Function to validate token
export const validateToken = async () => {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    
    if (!token) {
      return false;
    }
    
    // Make a request to a validation endpoint
    const response = await api.get('/admin/auth/validate');
    return response.status === 200;
  } catch (error) {
    console.error('Token validation failed', error);
    return false;
  }
};

export default api;
