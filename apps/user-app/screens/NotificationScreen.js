import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme, Divider, IconButton } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { format, isToday, isYesterday } from 'date-fns';
import { useNotifications } from '../context/NotificationContext';
import { NOTIFICATION_TYPES } from '../utils/notifications';

const NotificationScreen = ({ navigation }) => {
  const theme = useTheme();
  const { activeNotifications, dismissNotification, dismissAllNotifications } = useNotifications();
  const [notifications, setNotifications] = useState([]);
  const [groupedNotifications, setGroupedNotifications] = useState({});

  useEffect(() => {
    const sorted = [...activeNotifications].sort((a, b) => 
      new Date(b.timestamp) - new Date(a.timestamp)
    );
    
    setNotifications(sorted);
    
    // Group notifications by date
    const grouped = sorted.reduce((groups, notification) => {
      const date = new Date(notification.timestamp);
      let key;
      
      if (isToday(date)) {
        key = 'Today';
      } else if (isYesterday(date)) {
        key = 'Yesterday';
      } else {
        key = format(date, 'EEEE, MMMM d');
      }
      
      if (!groups[key]) {
        groups[key] = [];
      }
      
      groups[key].push(notification);
      return groups;
    }, {});
    
    setGroupedNotifications(grouped);
  }, [activeNotifications]);

  const getFormattedTime = (timestamp) => {
    return format(new Date(timestamp), 'h:mm a');
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
    // Handle navigation based on notification type and data
    if (notification.data?.screenName) {
      navigation.navigate(notification.data.screenName, notification.data.params || {});
    } else if (notification.data?.type === NOTIFICATION_TYPES.TASK) {
      navigation.navigate('TaskDetails', { id: notification.data.taskId });
    } else if (notification.data?.type === NOTIFICATION_TYPES.REPORT) {
      navigation.navigate('ReportDetails', { id: notification.data.reportId });
    }
  };

  const renderNotificationItem = ({ item }) => {
    const type = item.data?.type || NOTIFICATION_TYPES.SYSTEM;
    const iconName = getNotificationIcon(type);
    const iconColor = getNotificationColor(type);

    return (
      <TouchableOpacity
        style={[styles.notificationItem, { backgroundColor: theme.colors.background }]}
        onPress={() => handleNotificationPress(item)}
      >
        <View style={[styles.iconContainer, { backgroundColor: `${iconColor}10` }]}>
          <MaterialCommunityIcons name={iconName} size={24} color={iconColor} />
        </View>
        
        <View style={styles.contentContainer}>
          <Text style={[styles.title, { color: theme.colors.onSurface }]}>{item.title}</Text>
          <Text style={[styles.body, { color: theme.colors.onSurfaceVariant }]} numberOfLines={2}>
            {item.body}
          </Text>
          <Text style={[styles.timestamp, { color: theme.colors.outline }]}>
            {getFormattedTime(item.timestamp)}
          </Text>
        </View>
        
        <IconButton
          icon="close"
          size={20}
          onPress={() => dismissNotification(item.id)}
          iconColor={theme.colors.outline}
        />
      </TouchableOpacity>
    );
  };

  const renderSectionHeader = (title) => (
    <View style={[styles.sectionHeader, { backgroundColor: theme.colors.surfaceVariant }]}>
      <Text style={[styles.sectionTitle, { color: theme.colors.onSurfaceVariant }]}>
        {title}
      </Text>
    </View>
  );

  // Flatten grouped notifications for FlatList with section headers
  const renderSections = () => {
    const sections = [];
    
    Object.entries(groupedNotifications).forEach(([title, items]) => {
      sections.push({
        id: `header-${title}`,
        isHeader: true,
        title,
      });
      
      items.forEach((item) => {
        sections.push({
          ...item,
          isHeader: false,
        });
      });
    });
    
    return sections;
  };

  const renderItem = ({ item }) => {
    if (item.isHeader) {
      return renderSectionHeader(item.title);
    }
    return renderNotificationItem({ item });
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={styles.header}>
        <Text style={[styles.headerTitle, { color: theme.colors.onSurface }]}>
          Notifications
        </Text>
        
        {notifications.length > 0 && (
          <IconButton
            icon="trash-can-outline"
            size={24}
            onPress={dismissAllNotifications}
            iconColor={theme.colors.error}
          />
        )}
      </View>

      {notifications.length === 0 ? (
        <View style={styles.emptyContainer}>
          <MaterialCommunityIcons 
            name="bell-off-outline" 
            size={60} 
            color={theme.colors.outline} 
          />
          <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
            No notifications yet
          </Text>
          <Text style={[styles.emptySubtext, { color: theme.colors.outline }]}>
            You'll see notifications about tasks, reports, and alerts here
          </Text>
        </View>
      ) : (
        <FlatList
          data={renderSections()}
          keyExtractor={(item) => item.id || `notification-${item.timestamp}`}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          ItemSeparatorComponent={() => <Divider />}
        />
      )}
    </SafeAreaView>
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
    paddingVertical: 12,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: 'bold',
  },
  list: {
    flexGrow: 1,
  },
  notificationItem: {
    flexDirection: 'row',
    padding: 16,
    alignItems: 'center',
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  contentContainer: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
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
    fontSize: 18,
    fontWeight: '600',
  },
  emptySubtext: {
    marginTop: 8,
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  sectionHeader: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default NotificationScreen;
