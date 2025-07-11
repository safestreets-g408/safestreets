import { useEffect, useRef, useState } from 'react';
import { AppState, Platform } from 'react-native';
import { useNotifications } from '../context/NotificationContext';
import * as Notifications from 'expo-notifications';
import { createNotificationChannels } from '../utils/notifications';

// Hook to handle notification permissions and background/foreground behavior
export const useNotificationHandler = () => {
  const [appState, setAppState] = useState(AppState.currentState);
  const { activeNotifications } = useNotifications();
  const [showToast, setShowToast] = useState(false);
  const [currentToast, setCurrentToast] = useState(null);
  const appStateRef = useRef(AppState.currentState);
  const toastTimeoutRef = useRef(null);

  // Handle notification channels
  useEffect(() => {
    // Set up notification channels for Android
    if (Platform.OS === 'android') {
      createNotificationChannels();
    }
  }, []);

  // Handle app state changes to manage notification behavior
  useEffect(() => {
    const handleAppStateChange = (nextAppState) => {
      if (
        appStateRef.current.match(/inactive|background/) &&
        nextAppState === 'active'
      ) {
        // App has come to the foreground - update notification badge
        Notifications.setBadgeCountAsync(0);
      }
      
      appStateRef.current = nextAppState;
      setAppState(nextAppState);
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    return () => subscription.remove();
  }, []);

  // Show toast when new notification arrives in foreground
  useEffect(() => {
    if (activeNotifications.length > 0) {
      const lastNotification = activeNotifications[0];
      
      if (appState === 'active' && lastNotification) {
        // Clear any existing timeout
        if (toastTimeoutRef.current) {
          clearTimeout(toastTimeoutRef.current);
        }
        
        // Show toast for latest notification
        setCurrentToast(lastNotification);
        setShowToast(true);
        
        // Auto-hide toast after 4 seconds
        toastTimeoutRef.current = setTimeout(() => {
          setShowToast(false);
        }, 4000);
      }
    }
    
    return () => {
      if (toastTimeoutRef.current) {
        clearTimeout(toastTimeoutRef.current);
      }
    };
  }, [activeNotifications, appState]);

  const hideToast = () => {
    if (toastTimeoutRef.current) {
      clearTimeout(toastTimeoutRef.current);
    }
    setShowToast(false);
  };

  return {
    showToast,
    currentToast,
    hideToast,
  };
};

export default useNotificationHandler;
