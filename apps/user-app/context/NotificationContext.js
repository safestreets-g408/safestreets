import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { Platform, Alert } from 'react-native';
import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useAuth } from './AuthContext';
import { useSocket } from './SocketContext';
import axios from 'axios';
import { API_BASE_URL } from '../config';

// Configure notifications behavior
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

const NotificationContext = createContext();

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [expoPushToken, setExpoPushToken] = useState('');
  const [notification, setNotification] = useState(null);
  const [activeNotifications, setActiveNotifications] = useState([]);
  const notificationListener = useRef();
  const responseListener = useRef();
  const { isAuthenticated, fieldWorker } = useAuth();
  const { socket, connected } = useSocket();

  useEffect(() => {
    registerForPushNotifications();

    // Set up listeners
    notificationListener.current = Notifications.addNotificationReceivedListener(notification => {
      setNotification(notification);
      // Add to active notifications
      const notifData = notification.request.content;
      setActiveNotifications(prev => [...prev, {
        id: notification.request.identifier,
        title: notifData.title,
        body: notifData.body,
        data: notifData.data,
        timestamp: new Date(),
      }]);
    });

    responseListener.current = Notifications.addNotificationResponseReceivedListener(response => {
      const { data } = response.notification.request.content;
      handleNotificationResponse(data);
    });

    // Clean up listeners on unmount
    return () => {
      Notifications.removeNotificationSubscription(notificationListener.current);
      Notifications.removeNotificationSubscription(responseListener.current);
    };
  }, []);

  // Register and get token when authenticated
  useEffect(() => {
    if (isAuthenticated && fieldWorker && expoPushToken) {
      registerTokenWithServer(expoPushToken);
    }
  }, [isAuthenticated, fieldWorker, expoPushToken]);

  // Listen for socket notifications when connected
  useEffect(() => {
    if (!socket || !connected) return;
    
    socket.on('notification', handleIncomingNotification);
    
    return () => {
      socket.off('notification');
    };
  }, [socket, connected]);

  const registerForPushNotifications = async () => {
    if (!Device.isDevice) {
      Alert.alert('Notification Error', 'Notifications require a physical device to work properly');
      return;
    }

    try {
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;
      
      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }
      
      if (finalStatus !== 'granted') {
        Alert.alert('Warning', 'You need to enable notifications to receive alerts about new tasks and reports');
        return;
      }
      
      const token = await Notifications.getExpoPushTokenAsync({
        projectId: Constants.expoConfig.extra.eas.projectId,
      });
      
      setExpoPushToken(token.data);
      await AsyncStorage.setItem('pushToken', token.data);
      
      // Configure for Android
      if (Platform.OS === 'android') {
        Notifications.setNotificationChannelAsync('default', {
          name: 'default',
          importance: Notifications.AndroidImportance.MAX,
          vibrationPattern: [0, 250, 250, 250],
          lightColor: '#FF231F7C',
        });
      }
    } catch (error) {
      console.error('Failed to get push token:', error);
    }
  };

  const registerTokenWithServer = async (token) => {
    try {
      await axios.post(`${API_BASE_URL}/field-worker/register-device`, {
        pushToken: token,
        deviceType: Platform.OS,
      });
      console.log('Successfully registered push token with server');
    } catch (error) {
      console.error('Error registering push token with server:', error);
    }
  };

  const handleIncomingNotification = (data) => {
    // Process notification data from socket
    schedulePushNotification(data.title, data.body, data.data);
  };

  const handleNotificationResponse = (data) => {
    // Process user's interaction with notification
    try {
      // We need to use navigation ref from App.js since we don't have access to navigation prop here
      const navigationRef = require('../components/navigation/RootNavigation').navigationRef;
      
      if (!navigationRef || !navigationRef.isReady()) {
        console.warn('Navigation ref not ready');
        return;
      }
      
      if (data?.screenName) {
        // Navigate using provided screen name
        navigationRef.navigate(data.screenName, data.params || {});
      } else if (data?.type === 'task') {
        // Navigate to task details
        navigationRef.navigate('Tasks', { 
          screen: 'TaskDetails', 
          params: { id: data.taskId } 
        });
      } else if (data?.type === 'report') {
        // Navigate to report details
        navigationRef.navigate('Reports', { 
          screen: 'ReportDetails', 
          params: { id: data.reportId } 
        });
      }
    } catch (error) {
      console.error('Error navigating from notification:', error);
    }
  };

  const schedulePushNotification = async (title, body, data = {}) => {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: 'default',
        badge: 1,
      },
      trigger: null, // Deliver immediately
    });
  };

  const dismissNotification = async (notificationId) => {
    await Notifications.dismissNotificationAsync(notificationId);
    setActiveNotifications(prev => 
      prev.filter(notification => notification.id !== notificationId)
    );
  };

  const dismissAllNotifications = async () => {
    await Notifications.dismissAllNotificationsAsync();
    setActiveNotifications([]);
  };

  const value = {
    expoPushToken,
    notification,
    activeNotifications,
    schedulePushNotification,
    dismissNotification,
    dismissAllNotifications,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};
