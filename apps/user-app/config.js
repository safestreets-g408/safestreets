// API configuration
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';

const MANUAL_OVERRIDE_IP = '192.168.23.177'; 
const API_PORT = '5030';
// We'll use the API_BASE_URL from line 111

// Function to validate URL and check if it's reachable
export const testApiConnection = async (url, timeout = 5000) => {
  try {
    console.log('Testing API connection to:', url);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await fetch(`${url}/health`, {
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
  }
};

// Function to get the appropriate base URL
export const getBaseUrl = async () => {
  try {
    // Try to get the stored custom URL first
    const storedUrl = await AsyncStorage.getItem('custom_api_url');
    if (storedUrl) {
      console.log('Found custom API URL in storage:', storedUrl);
      
      // Test if the stored URL is reachable
      const isReachable = await testApiConnection(storedUrl);
      if (isReachable) {
        console.log('Custom API URL is reachable');
        return storedUrl;
      } else {
        console.log('Custom API URL is not reachable, falling back to default');
      }
    }
  } catch (error) {
    console.error('Error reading or testing custom API URL:', error);
  }

  // Get the Expo host URL when running in Expo Go
  const debuggerHost = Constants.expoConfig?.hostUri || Constants.manifest?.debuggerHost;
  const expoHost = debuggerHost ? debuggerHost.split(':')[0] : null;
  
  let baseUrl;

  if (__DEV__) {
    // Array of possible URLs to try
    const possibleUrls = [];
    
    if (Platform.OS === 'android') {
      if (expoHost) {
        possibleUrls.push(`http://${expoHost}:${API_PORT}/api`);  // Use the Expo debug IP
      } 
      possibleUrls.push(`http://${MANUAL_OVERRIDE_IP}:${API_PORT}/api`); // Fallback to manual IP
      possibleUrls.push(`http://10.0.2.2:${API_PORT}/api`); // Android emulator localhost
    } else {
      if (expoHost) {
        possibleUrls.push(`http://${expoHost}:${API_PORT}/api`);  // Expo host
      }
      possibleUrls.push(`http://${MANUAL_OVERRIDE_IP}:${API_PORT}/api`); // Manual override IP
      possibleUrls.push(`http://localhost:${API_PORT}/api`); // Local development
      possibleUrls.push(`http://127.0.0.1:${API_PORT}/api`); // Another localhost option
    }
    
    // Try each URL until we find one that's reachable
    for (const url of possibleUrls) {
      try {
        console.log('Trying API URL:', url);
        const isReachable = await testApiConnection(url);
        if (isReachable) {
          console.log('Found reachable API URL:', url);
          baseUrl = url;
          break;
        }
      } catch (e) {
        console.log('Failed testing URL:', url, e);
      }
    }
    
    // If none of the URLs worked, use the default
    if (!baseUrl) {
      console.warn('No reachable API URL found, using default');
      if (Platform.OS === 'android') {
        baseUrl = `http://${MANUAL_OVERRIDE_IP}:${API_PORT}/api`;
      } else {
        baseUrl = `http://localhost:${API_PORT}/api`;
      }
    }
  } else {
    // Production URL
    baseUrl = 'https://api.safestreets-prod.com/api';
  }

  console.log('Using API base URL:', baseUrl);
  return baseUrl;
};

// This is the default URL configuration
// It will be updated at runtime by getBaseUrl()
export let API_BASE_URL = `http://${MANUAL_OVERRIDE_IP}:${API_PORT}/api`;

// Set a custom API URL (can be called from settings screen)
export const setCustomApiUrl = async (url) => {
  try {
    await AsyncStorage.setItem('custom_api_url', url);
    API_BASE_URL = url;
    console.log('Custom API URL saved:', url);
    return true;
  } catch (error) {
    console.log('Error saving custom API URL:', error);
    return false;
  }
};

// Initialize the API URL
(async () => {
  try {
    API_BASE_URL = await getBaseUrl();
  } catch (error) {
    console.log('Error initializing API URL:', error);
  }
})();

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