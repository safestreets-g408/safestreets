import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Notification categories for different types
export const NOTIFICATION_TYPES = {
  TASK: 'task',
  REPORT: 'report',
  ALERT: 'alert',
  MESSAGE: 'message',
  SYSTEM: 'system',
};

// Function to get badge count from AsyncStorage
export const getBadgeCount = async () => {
  try {
    const count = await AsyncStorage.getItem('badgeCount');
    return count ? parseInt(count, 10) : 0;
  } catch (error) {
    console.error('Error getting badge count:', error);
    return 0;
  }
};

// Function to set badge count in AsyncStorage
export const setBadgeCount = async (count) => {
  try {
    await AsyncStorage.setItem('badgeCount', count.toString());
    if (Platform.OS === 'ios') {
      await Notifications.setBadgeCountAsync(count);
    }
  } catch (error) {
    console.error('Error setting badge count:', error);
  }
};

// Function to increment badge count
export const incrementBadgeCount = async () => {
  const currentCount = await getBadgeCount();
  await setBadgeCount(currentCount + 1);
  return currentCount + 1;
};

// Function to reset badge count
export const resetBadgeCount = async () => {
  await setBadgeCount(0);
};

// Function to parse notification data for uniform handling
export const parseNotificationData = (notification) => {
  // Extract useful information from notification object
  if (!notification) return null;
  
  const data = notification.request?.content?.data || notification.data || {};
  
  return {
    id: notification.request?.identifier || data.id,
    title: notification.request?.content?.title || data.title,
    body: notification.request?.content?.body || data.body,
    type: data.type || NOTIFICATION_TYPES.SYSTEM,
    data: data,
    timestamp: new Date(data.timestamp || Date.now()),
    read: false,
  };
};

// Function to create notification channels for Android
export const createNotificationChannels = () => {
  if (Platform.OS === 'android') {
    // Main default channel
    Notifications.setNotificationChannelAsync('default', {
      name: 'Default Notifications',
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#FF231F7C',
    });
    
    // Task assignments channel
    Notifications.setNotificationChannelAsync('tasks', {
      name: 'Task Assignments',
      description: 'Notifications for new task assignments and updates',
      importance: Notifications.AndroidImportance.HIGH,
      vibrationPattern: [0, 250, 0, 250],
      lightColor: '#3478F6',
    });
    
    // Reports channel
    Notifications.setNotificationChannelAsync('reports', {
      name: 'Reports',
      description: 'Notifications about damage reports and assessments',
      importance: Notifications.AndroidImportance.HIGH,
      vibrationPattern: [0, 250, 0, 250],
      lightColor: '#34C759',
    });
    
    // Alerts channel
    Notifications.setNotificationChannelAsync('alerts', {
      name: 'Urgent Alerts',
      description: 'Critical notifications requiring immediate attention',
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 500, 200, 500],
      lightColor: '#FF3B30',
      sound: 'alert_sound',
    });
    
    // Messages channel
    Notifications.setNotificationChannelAsync('messages', {
      name: 'Messages',
      description: 'Chat and communication messages',
      importance: Notifications.AndroidImportance.DEFAULT,
      vibrationPattern: [0, 100, 100, 100],
      lightColor: '#007AFF',
    });
  }
};

// Get channel ID based on notification type
export const getChannelId = (type) => {
  switch (type) {
    case NOTIFICATION_TYPES.TASK:
      return 'tasks';
    case NOTIFICATION_TYPES.REPORT:
      return 'reports';
    case NOTIFICATION_TYPES.ALERT:
      return 'alerts';
    case NOTIFICATION_TYPES.MESSAGE:
      return 'messages';
    default:
      return 'default';
  }
};

// Schedule a notification with the appropriate channel
export const scheduleNotificationWithChannel = async (title, body, data = {}, options = {}) => {
  const type = data.type || NOTIFICATION_TYPES.SYSTEM;
  const channelId = getChannelId(type);
  
  const notificationContent = {
    title,
    body,
    data: {
      ...data,
      timestamp: Date.now(),
    },
    badge: await incrementBadgeCount(),
  };
  
  // Set sound based on notification type
  if (type === NOTIFICATION_TYPES.ALERT) {
    notificationContent.sound = 'alert_sound';
  } else {
    notificationContent.sound = 'default';
  }
  
  // For Android, specify the channel
  if (Platform.OS === 'android') {
    notificationContent.channelId = channelId;
  }
  
  return await Notifications.scheduleNotificationAsync({
    content: notificationContent,
    trigger: options.trigger || null, // null means deliver immediately
  });
};
