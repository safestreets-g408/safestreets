import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { useTheme } from 'react-native-paper';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { NOTIFICATION_TYPES } from '../../utils/notifications';

const NotificationToast = ({ notification, onPress, onDismiss, autoHide = true }) => {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const slideAnimation = useRef(new Animated.Value(-100)).current;
  const opacityAnimation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const showToast = () => {
      Animated.parallel([
        Animated.timing(slideAnimation, {
          toValue: 0,
          duration: 300,
          useNativeDriver: true,
        }),
        Animated.timing(opacityAnimation, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        })
      ]).start();
    };

    const hideToast = () => {
      Animated.parallel([
        Animated.timing(slideAnimation, {
          toValue: -100,
          duration: 300,
          useNativeDriver: true,
        }),
        Animated.timing(opacityAnimation, {
          toValue: 0,
          duration: 300,
          useNativeDriver: true,
        })
      ]).start(() => {
        if (onDismiss) onDismiss();
      });
    };

    showToast();

    // Auto hide after 4 seconds if enabled
    let timeout;
    if (autoHide) {
      timeout = setTimeout(hideToast, 4000);
    }

    return () => {
      if (timeout) clearTimeout(timeout);
    };
  }, [notification, onDismiss, autoHide, slideAnimation, opacityAnimation]);

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

  // Get notification data
  const { title, body, data } = notification;
  const type = data?.type || NOTIFICATION_TYPES.SYSTEM;
  const iconName = getNotificationIcon(type);
  const iconColor = getNotificationColor(type);

  return (
    <Animated.View 
      style={[
        styles.container,
        { 
          backgroundColor: theme.colors.surface,
          shadowColor: theme.colors.shadow,
          top: insets.top + 10,
          transform: [{ translateY: slideAnimation }],
          opacity: opacityAnimation,
        }
      ]}
    >
      <TouchableOpacity
        style={styles.content}
        onPress={onPress}
        activeOpacity={0.8}
      >
        <View style={styles.iconContainer}>
          <MaterialCommunityIcons name={iconName} size={24} color={iconColor} />
        </View>
        
        <View style={styles.textContainer}>
          <Text 
            style={[styles.title, { color: theme.colors.onSurface }]}
            numberOfLines={1}
          >
            {title}
          </Text>
          <Text 
            style={[styles.body, { color: theme.colors.onSurfaceVariant }]}
            numberOfLines={2}
          >
            {body}
          </Text>
        </View>
        
        <TouchableOpacity style={styles.closeButton} onPress={onDismiss}>
          <MaterialCommunityIcons 
            name="close" 
            size={16} 
            color={theme.colors.onSurfaceVariant} 
          />
        </TouchableOpacity>
      </TouchableOpacity>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    left: 16,
    right: 16,
    borderRadius: 8,
    elevation: 4,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    zIndex: 1000,
  },
  content: {
    flexDirection: 'row',
    padding: 12,
    alignItems: 'center',
  },
  iconContainer: {
    marginRight: 12,
  },
  textContainer: {
    flex: 1,
  },
  title: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 2,
  },
  body: {
    fontSize: 12,
  },
  closeButton: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
});

export default NotificationToast;
