import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Card, Avatar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const NotificationCard = ({ 
  notification, 
  onPress, 
  onMarkAsRead,
  index = 0 
}) => {
  const getNotificationIcon = (type) => {
    switch (type) {
      case 'report_update':
        return 'file-document-edit';
      case 'task_assigned':
        return 'briefcase-plus';
      case 'task_completed':
        return 'check-circle';
      case 'system':
        return 'bell';
      default:
        return 'information';
    }
  };

  const getNotificationColor = (type) => {
    switch (type) {
      case 'report_update':
        return theme.colors.info;
      case 'task_assigned':
        return theme.colors.warning;
      case 'task_completed':
        return theme.colors.success;
      case 'system':
        return theme.colors.primary;
      default:
        return theme.colors.primary;
    }
  };

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffInMinutes = Math.floor((now - time) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  return (
    <Animatable.View
      animation="fadeInUp"
      duration={600}
      delay={index * 100}
    >
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        <Card style={[
          styles.notificationCard,
          !notification.read && styles.unreadCard
        ]}>
          <View style={styles.cardContent}>
            <View style={styles.leftContent}>
              <Avatar.Icon
                size={48}
                icon={getNotificationIcon(notification.type)}
                style={[
                  styles.notificationIcon,
                  { backgroundColor: getNotificationColor(notification.type) }
                ]}
                color={theme.colors.surface}
              />
              <View style={styles.textContent}>
                <Text style={[
                  styles.notificationTitle,
                  !notification.read && styles.unreadTitle
                ]}>
                  {notification.title}
                </Text>
                <Text style={styles.notificationMessage} numberOfLines={2}>
                  {notification.message}
                </Text>
                <Text style={styles.notificationTime}>
                  {formatTimeAgo(notification.timestamp)}
                </Text>
              </View>
            </View>
            
            <View style={styles.rightContent}>
              {!notification.read && (
                <View style={styles.unreadIndicator} />
              )}
              <TouchableOpacity
                onPress={() => onMarkAsRead(notification.id)}
                style={styles.markReadButton}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <MaterialCommunityIcons
                  name={notification.read ? "check-circle" : "check-circle-outline"}
                  size={20}
                  color={notification.read ? theme.colors.success : theme.colors.onSurfaceVariant}
                />
              </TouchableOpacity>
            </View>
          </View>
        </Card>
      </TouchableOpacity>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  notificationCard: {
    marginHorizontal: 16,
    marginVertical: 6,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius.large,
    ...theme.shadows.small,
  },
  unreadCard: {
    borderLeftWidth: 4,
    borderLeftColor: theme.colors.primary,
    backgroundColor: theme.colors.primaryContainer,
  },
  cardContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: 16,
  },
  leftContent: {
    flexDirection: 'row',
    flex: 1,
    alignItems: 'flex-start',
  },
  notificationIcon: {
    marginRight: 12,
  },
  textContent: {
    flex: 1,
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.onSurface,
    marginBottom: 4,
    lineHeight: 20,
  },
  unreadTitle: {
    fontWeight: '700',
    color: theme.colors.primary,
  },
  notificationMessage: {
    fontSize: 14,
    color: theme.colors.onSurfaceVariant,
    lineHeight: 18,
    marginBottom: 6,
  },
  notificationTime: {
    fontSize: 12,
    color: theme.colors.onSurfaceVariant,
    fontWeight: '500',
  },
  rightContent: {
    alignItems: 'center',
    marginLeft: 8,
  },
  unreadIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.primary,
    marginBottom: 8,
  },
  markReadButton: {
    padding: 4,
  },
});

export default NotificationCard;
