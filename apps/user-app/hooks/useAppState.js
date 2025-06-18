import { useState, useEffect, useRef } from 'react';
import { AppState } from 'react-native';

/**
 * Custom hook for handling app state changes and global app state
 */
export const useAppState = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [appError, setAppError] = useState(null);
  const appStateRef = useRef(AppState.currentState);
  
  // Safely try to use Auth context, but don't require it
  let auth = {};
  try {
    const { useAuth } = require('../context/AuthContext');
    auth = useAuth();
  } catch (e) {
    console.warn('Auth context not available, continuing without authentication features');
  }

  // Track app state changes (foreground/background)
  useEffect(() => {
    const handleAppStateChange = (nextAppState) => {
      if (appStateRef.current.match(/inactive|background/) && nextAppState === 'active') {
        // App has come to the foreground
        if (auth.checkAuthStatus) {
          auth.checkAuthStatus();
        }
      }
      appStateRef.current = nextAppState;
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    
    // Initial loading state
    setTimeout(() => setIsLoading(false), 1000);

    return () => {
      subscription.remove();
    };
  }, [auth.checkAuthStatus]);

  /**
   * Global error handler
   */
  const handleError = (error, stackTrace) => {
    console.error('Error in app:', error);
    console.log('Stack trace:', stackTrace);
    setAppError(error);
  };

  return {
    isLoading,
    appError,
    handleError
  };
};
