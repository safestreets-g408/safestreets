// API configuration
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';

const MANUAL_OVERRIDE_IP = '192.168.23.177'; 
const API_PORT = '5030';

// Function to get the appropriate base URL
export const getBaseUrl = async () => {
  try {
    const storedUrl = await AsyncStorage.getItem('custom_api_url');
    if (storedUrl) {
      console.log('Using custom API URL from storage:', storedUrl);
      return storedUrl;
    }
  } catch (error) {
    console.log('Error reading custom API URL from storage:', error);
  }

  // Get the Expo host URL when running in Expo Go
  const debuggerHost = Constants.expoConfig?.hostUri || Constants.manifest?.debuggerHost;
  const expoHost = debuggerHost ? debuggerHost.split(':')[0] : null;
  
  let baseUrl;

  if (__DEV__) {
    if (Platform.OS === 'android') {
      if (expoHost) {
        baseUrl = `http://${expoHost}:${API_PORT}/api`;  // Use the Expo debug IP
      } else {
        baseUrl = `http://${MANUAL_OVERRIDE_IP}:${API_PORT}/api`; // Fallback to manual IP
      }
    } else if (expoHost) {
      // For Expo Go - use the same IP that Expo server is running on
      baseUrl = `http://${expoHost}:${API_PORT}/api`;
    } else if (MANUAL_OVERRIDE_IP) {
      // Use manually specified IP if available
      baseUrl = `http://${MANUAL_OVERRIDE_IP}:${API_PORT}/api`;
    } else {
      // Fallback to localhost
      baseUrl = `http://localhost:${API_PORT}/api`;
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