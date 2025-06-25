import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Card, Avatar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const RecentActivityCard = ({ 
  activities = [], 
  onViewAll,
  index = 0 
}) => {
  const getActivityIcon = (type) => {
    switch (type) {
      case 'report_submitted':
        return 'file-document-plus';
      case 'report_updated':
        return 'file-document-edit';
      case 'task_completed':
        return 'check-circle';
      case 'task_assigned':
        return 'briefcase-plus';
      case 'comment_added':
        return 'comment-plus';
      default:
        return 'information';
    }
  };

  const getActivityColor = (type) => {
    switch (type) {
      case 'report_submitted':
        return theme.colors.info;
      case 'report_updated':
        return theme.colors.warning;
      case 'task_completed':
        return theme.colors.success;
      case 'task_assigned':
        return theme.colors.secondary;
      case 'comment_added':
        return theme.colors.tertiary;
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

  if (!activities || activities.length === 0) {
    return (
      <Animatable.View
        animation="fadeInUp"
        duration={600}
        delay={index * 100}
      >
        <Card style={styles.activityCard}>
          <View style={styles.header}>
            <View style={styles.titleContainer}>
              <MaterialCommunityIcons
                name="history"
                size={24}
                color={theme.colors.primary}
              />
              <Text style={styles.cardTitle}>Recent Activity</Text>
            </View>
          </View>
          
          <View style={styles.emptyState}>
            <MaterialCommunityIcons
              name="clock-outline"
              size={48}
              color={theme.colors.onSurfaceVariant}
              style={styles.emptyIcon}
            />
            <Text style={styles.emptyText}>No recent activity</Text>
            <Text style={styles.emptySubtext}>
              Your recent actions will appear here
            </Text>
          </View>
        </Card>
      </Animatable.View>
    );
  }

  return (
    <Animatable.View
      animation="fadeInUp"
      duration={600}
      delay={index * 100}
    >
      <Card style={styles.activityCard}>
        <View style={styles.header}>
          <View style={styles.titleContainer}>
            <MaterialCommunityIcons
              name="history"
              size={24}
              color={theme.colors.primary}
            />
            <Text style={styles.cardTitle}>Recent Activity</Text>
          </View>
          
          {activities.length > 3 && (
            <TouchableOpacity onPress={onViewAll} style={styles.viewAllButton}>
              <Text style={styles.viewAllText}>View All</Text>
              <MaterialCommunityIcons
                name="chevron-right"
                size={16}
                color={theme.colors.primary}
              />
            </TouchableOpacity>
          )}
        </View>
        
        <View style={styles.activitiesList}>
          {activities.slice(0, 3).map((activity, activityIndex) => (
            <View key={activity.id || activityIndex} style={styles.activityItem}>
              <Avatar.Icon
                size={36}
                icon={getActivityIcon(activity.type)}
                style={[
                  styles.activityIcon,
                  { backgroundColor: getActivityColor(activity.type) }
                ]}
                color={theme.colors.surface}
              />
              
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle} numberOfLines={1}>
                  {activity.title}
                </Text>
                <Text style={styles.activityDescription} numberOfLines={2}>
                  {activity.description}
                </Text>
                <Text style={styles.activityTime}>
                  {formatTimeAgo(activity.timestamp)}
                </Text>
              </View>
            </View>
          ))}
        </View>
      </Card>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  activityCard: {
    marginHorizontal: 16,
    marginVertical: 8,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius.large,
    ...theme.shadows.small,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 12,
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: theme.colors.onSurface,
    ...theme.typography.titleMedium,
  },
  viewAllButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  viewAllText: {
    fontSize: 14,
    color: theme.colors.primary,
    fontWeight: '600',
  },
  activitiesList: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  activityIcon: {
    marginRight: 12,
  },
  activityContent: {
    flex: 1,
  },
  activityTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: theme.colors.onSurface,
    marginBottom: 4,
  },
  activityDescription: {
    fontSize: 13,
    color: theme.colors.onSurfaceVariant,
    lineHeight: 16,
    marginBottom: 4,
  },
  activityTime: {
    fontSize: 12,
    color: theme.colors.onSurfaceVariant,
    fontWeight: '500',
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
    paddingTop: 20,
  },
  emptyIcon: {
    opacity: 0.3,
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.onSurfaceVariant,
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: theme.colors.onSurfaceVariant,
    textAlign: 'center',
    opacity: 0.7,
  },
});

export default RecentActivityCard;
