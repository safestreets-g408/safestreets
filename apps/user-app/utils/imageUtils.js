import { API_BASE_URL } from '../config';
import { getValidAuthToken, getAuthToken } from './auth';
import AsyncStorage from '@react-native-async-storage/async-storage';

// In-memory cache for tokens to avoid excessive AsyncStorage calls
let cachedToken = null;
let tokenExpiryTime = null;

const TOKEN_CACHE_DURATION = 55 * 60 * 1000; // 55 minutes in milliseconds

const getToken = async () => {
  // Check if we have a valid cached token
  if (cachedToken && tokenExpiryTime && Date.now() < tokenExpiryTime) {
    console.log('Using cached token (expires in', Math.round((tokenExpiryTime - Date.now())/1000/60), 'minutes)');
    return cachedToken;
  }
  
  try {
    // Try to get from AsyncStorage first
    const token = await AsyncStorage.getItem('fieldWorkerToken');
    if (token) {
      console.log('Token retrieved from storage (length:', token.length, ')');
      cachedToken = token;
      tokenExpiryTime = Date.now() + TOKEN_CACHE_DURATION;
      return token;
    }
    
    // Otherwise get a fresh token
    console.log('No token in storage, getting fresh token');
    const freshToken = await getValidAuthToken();
    if (freshToken) {
      console.log('Fresh token obtained (length:', freshToken.length, ')');
      cachedToken = freshToken;
      tokenExpiryTime = Date.now() + TOKEN_CACHE_DURATION;
      return freshToken;
    }
    
    console.warn('No token available from any source');
    return ''; // Return empty string if no token available
  } catch (e) {
    console.error('Error getting auth token for images:', e);
    return '';
  }
};

export const getReportImageUrl = async (report, type = 'thumbnail') => {
  if (!report) return 'https://via.placeholder.com/300';
  
  try {
    // Get authentication token
    const token = await getToken();
    const tokenParam = token ? `?token=${token}` : '';
    
    // Use date to prevent caching
    const timestamp = Date.now();
    const cacheBuster = tokenParam ? `&_t=${timestamp}` : `?_t=${timestamp}`;
    
    // Return direct URL if available and valid
    if (report.imageUrl && report.imageUrl.startsWith('http')) {
      return report.imageUrl;
    }
    
    // Handle reports with image IDs
    if (report._id) {
      return `${API_BASE_URL}/fieldworker/damage/report/${report._id}/image/${type}${tokenParam}${cacheBuster}`;
    }
    
    // Reports with just id (no underscore)
    if (report.id) {
      return `${API_BASE_URL}/fieldworker/damage/report/${report.id}/image/${type}${tokenParam}${cacheBuster}`;
    }
  } catch (error) {
    console.error('Error generating image URL:', error);
    // Continue execution to handle fallbacks
  }
  
  // For reports with images array
  if (report.images && report.images.length > 0) {
    // If the image object contains a URL
    const firstImage = report.images[0];
    if (typeof firstImage === 'string') {
      return firstImage;
    } else if (firstImage.url) {
      return firstImage.url;
    }
    
    // Otherwise construct URL from the first image ID and report ID
    const reportId = report._id || report.id;
    if (reportId) {
      return `${API_BASE_URL}/fieldworker/damage/report/${reportId}/image/${type}${tokenParam}`;
    }
  }
  
  // Default placeholder
  return 'https://via.placeholder.com/300?text=No+Image';
};



const getTokenSync = () => {
  try {
    // Use cached token if available and not expired
    if (cachedToken && tokenExpiryTime && Date.now() < tokenExpiryTime) {
      return cachedToken;
    }
    
    // If we don't have a valid cached token, try to force initialize one
    // This is a fallback and might not work reliably
    console.warn('No cached token available for sync image loading');
    
    // In extreme cases, return empty as a fallback
    return '';
  } catch (e) {
    console.error('Error getting auth token for images sync:', e);
    return '';
  }
};

/**
 * Synchronous version that tries to use a cached token if available
 * Use only when async version cannot be used
 */
export const getReportImageUrlSync = (report, type = 'thumbnail') => {
  if (!report) {
    console.warn('getReportImageUrlSync: No report provided');
    return 'https://via.placeholder.com/300?text=Missing+Report';
  }
  
  try {
    // Try to get a token
    const token = getTokenSync();
    
    // Log report ID for debugging
    const reportId = report._id || report.id;
    console.log(`Getting image URL for report ${reportId}, type: ${type}`);
    
    // For security, don't log the full token
    const tokenIndicator = token ? 
      `${token.substring(0, 10)}...${token.substring(token.length - 5)}` : 
      'none';
    console.log(`Using token: ${tokenIndicator}, length: ${token?.length || 0}`);
    
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : '';
    
    // Use date to prevent caching
    const timestamp = Date.now();
    const cacheBuster = tokenParam ? `&_t=${timestamp}` : `?_t=${timestamp}`;
    
    // Return direct URL if available and valid
    if (report.imageUrl && report.imageUrl.startsWith('http')) {
      console.log('Using direct imageUrl from report');
      return report.imageUrl;
    }
    
    // Handle reports with image IDs
    if (report._id) {
      const url = `${API_BASE_URL}/fieldworker/damage/report/${report._id}/image/${type}${tokenParam}${cacheBuster}`;
      console.log('Constructed URL from _id:', url.substring(0, url.indexOf('?')));
      return url;
    }
    
    // Reports with just id (no underscore)
    if (report.id) {
      const url = `${API_BASE_URL}/fieldworker/damage/report/${report.id}/image/${type}${tokenParam}${cacheBuster}`;
      console.log('Constructed URL from id:', url.substring(0, url.indexOf('?')));
      return url;
    }
    
    // For reports with images array
    if (report.images && report.images.length > 0) {
      console.log('Report has images array with', report.images.length, 'items');
      // If the image object contains a URL
      const firstImage = report.images[0];
      if (typeof firstImage === 'string') {
        if (firstImage.startsWith('http')) {
          console.log('Using string URL from images array');
          return firstImage;
        } else {
          console.log('Image string does not start with http:', firstImage.substring(0, 20));
        }
      } else if (firstImage && firstImage.url) {
        if (firstImage.url.startsWith('http')) {
          console.log('Using nested URL from images array');
          return firstImage.url;
        } else {
          console.log('Nested URL does not start with http:', firstImage.url.substring(0, 20));
        }
      }
      
      // Otherwise construct URL from the first image ID and report ID
      const reportId = report._id || report.id;
      if (reportId) {
        const url = `${API_BASE_URL}/fieldworker/damage/report/${reportId}/image/${type}${tokenParam}${cacheBuster}`;
        console.log('Constructed URL from reportId with images array:', url.substring(0, url.indexOf('?')));
        return url;
      }
    } else {
      console.log('Report has no images array');
    }
  } catch (error) {
    console.error('Error generating image URL (sync):', error);
    // Continue to fallback
  }
  
  // Default placeholder
  return 'https://via.placeholder.com/300?text=No+Image+Available';
};

/**
 * Initialize token for images - call this as early as possible in app startup
 */
export const preloadImageToken = async () => {
  try {
    // Clear existing cache to force fresh token
    cachedToken = null;
    tokenExpiryTime = null;
    
    // Get fresh token from AuthStorage
    const token = await getAuthToken();
    if (token) {
      cachedToken = token;
      tokenExpiryTime = Date.now() + TOKEN_CACHE_DURATION;
      console.log('Image token preloaded successfully (length:', token.length, ')');
      return token;
    } 
    
    // If no token from storage, try to get a fresh one
    console.log('No token from storage, trying getValidAuthToken');
    const freshToken = await getValidAuthToken();
    if (freshToken) {
      cachedToken = freshToken;
      tokenExpiryTime = Date.now() + TOKEN_CACHE_DURATION;
      console.log('Fresh image token obtained (length:', freshToken.length, ')');
      return freshToken;
    }
    
    console.warn('No token available for image loading');
    return null;
  } catch (error) {
    console.error('Failed to preload image token:', error);
    return null;
  }
};
