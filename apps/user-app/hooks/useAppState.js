import { useState, useEffect, useRef } from 'react';
import { AppState } from 'react-native';
import { useAuth } from '../context/AuthContext';

/**
 * Custom hook for handling app state changes and global app state
 */
export const useAppState = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [appError, setAppError] = useState(null);
  const appStateRef = useRef(AppState.currentState);
  const { checkAuthStatus } = useAuth();

  // Track app state changes (foreground/background)
  useEffect(() => {
    const handleAppStateChange = (nextAppState) => {
      if (appStateRef.current.match(/inactive|background/) && nextAppState === 'active') {
        // App has come to the foreground
        checkAuthStatus();
      }
      appStateRef.current = nextAppState;
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    
    // Initial loading state
    setTimeout(() => setIsLoading(false), 1000);

    return () => {
      subscription.remove();
    };
  }, [checkAuthStatus]);

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
