import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { useTheme, Badge, Divider, IconButton } from 'react-native-paper';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Swipeable } from 'react-native-gesture-handler';
import * as Animatable from 'react-native-animatable';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { format, isToday, isYesterday } from 'date-fns';
import { useNotifications } from '../../context/NotificationContext';
import { NOTIFICATION_TYPES } from '../../utils/notifications';

const NotificationList = ({ onClose }) => {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const { activeNotifications, dismissNotification, dismissAllNotifications } = useNotifications();
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    setNotifications([...activeNotifications].sort((a, b) => 
      new Date(b.timestamp) - new Date(a.timestamp)
    ));
  }, [activeNotifications]);

  const getFormattedDate = (timestamp) => {
    const date = new Date(timestamp);
    if (isToday(date)) return `Today, ${format(date, 'h:mm a')}`;
    if (isYesterday(date)) return `Yesterday, ${format(date, 'h:mm a')}`;
    return format(date, 'MMM d, h:mm a');
  };

  const getNotificationIcon = (type) => {
    switch (type) {
      case NOTIFICATION_TYPES.TASK:
        return 'clipboard-check-outline';
      case NOTIFICATION_TYPES.REPORT:
        return 'file-document-outline';
      case NOTIFICATION_TYPES.ALERT:
        return 'alert-circle-outline';
      case NOTIFICATION_TYPES.MESSAGE:
        return 'message-outline';
      default:
        return 'bell-outline';
    }
  };

  const getNotificationColor = (type) => {
    switch (type) {
      case NOTIFICATION_TYPES.TASK:
        return theme.colors.primary;
      case NOTIFICATION_TYPES.REPORT:
        return theme.colors.secondary;
      case NOTIFICATION_TYPES.ALERT:
        return theme.colors.error;
      case NOTIFICATION_TYPES.MESSAGE:
        return theme.colors.notification;
      default:
        return theme.colors.primary;
    }
  };

  const handleNotificationPress = (notification) => {
    // Handle navigation based on notification type
    if (notification.data?.screenName) {
      // Navigate to the specified screen
      console.log(`Navigate to: ${notification.data.screenName}`);
      // Navigation would happen here
    }
  };

  const renderRightActions = (progress, dragX, notification) => {
    const trans = dragX.interpolate({
      inputRange: [-100, 0],
      outputRange: [0, 100],
      extrapolate: 'clamp',
    });

    return (
      <View style={styles.rightAction}>
        <Animated.View style={{ transform: [{ translateX: trans }] }}>
          <TouchableOpacity 
            onPress={() => dismissNotification(notification.id)}
            style={[styles.actionButton, { backgroundColor: theme.colors.error }]}
          >
            <MaterialCommunityIcons name="delete-outline" size={24} color="white" />
            <Text style={styles.actionText}>Delete</Text>
          </TouchableOpacity>
        </Animated.View>
      </View>
    );
  };

  const renderNotificationItem = ({ item }) => {
    const type = item.data?.type || NOTIFICATION_TYPES.SYSTEM;
    const iconName = getNotificationIcon(type);
    const iconColor = getNotificationColor(type);

    return (
      <Swipeable
        renderRightActions={(progress, dragX) => renderRightActions(progress, dragX, item)}
      >
        <TouchableOpacity
          style={[styles.notificationItem, { backgroundColor: theme.colors.background }]}
          onPress={() => handleNotificationPress(item)}
        >
          <View style={styles.iconContainer}>
            <MaterialCommunityIcons name={iconName} size={24} color={iconColor} />
          </View>
          
          <View style={styles.contentContainer}>
            <Text style={[styles.title, { color: theme.colors.onSurface }]}>{item.title}</Text>
            <Text style={[styles.body, { color: theme.colors.onSurfaceVariant }]} numberOfLines={2}>
              {item.body}
            </Text>
            <Text style={[styles.timestamp, { color: theme.colors.outline }]}>
              {getFormattedDate(item.timestamp)}
            </Text>
          </View>
        </TouchableOpacity>
        <Divider />
      </Swipeable>
    );
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <Text style={[styles.headerTitle, { color: theme.colors.onSurface }]}>
          Notifications
        </Text>
        <View style={styles.headerActions}>
          {notifications.length > 0 && (
            <IconButton
              icon="trash-can-outline"
              size={24}
              onPress={dismissAllNotifications}
              iconColor={theme.colors.error}
            />
          )}
          <IconButton
            icon="close"
            size={24}
            onPress={onClose}
            iconColor={theme.colors.onSurface}
          />
        </View>
      </View>

      {notifications.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Animatable.View 
            animation="pulse" 
            iterationCount="infinite" 
            duration={2000}
          >
            <MaterialCommunityIcons 
              name="bell-off-outline" 
              size={48} 
              color={theme.colors.outline} 
            />
          </Animatable.View>
          <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
            No notifications yet
          </Text>
        </View>
      ) : (
        <FlatList
          data={notifications}
          keyExtractor={(item) => item.id}
          renderItem={renderNotificationItem}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  list: {
    flexGrow: 1,
  },
  notificationItem: {
    flexDirection: 'row',
    padding: 16,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  contentContainer: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  body: {
    fontSize: 14,
    marginBottom: 4,
  },
  timestamp: {
    fontSize: 12,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 16,
  },
  rightAction: {
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  actionButton: {
    justifyContent: 'center',
    alignItems: 'center',
    width: 80,
    height: '100%',
  },
  actionText: {
    color: 'white',
    fontSize: 12,
    marginTop: 4,
  },
});

export default NotificationList;
