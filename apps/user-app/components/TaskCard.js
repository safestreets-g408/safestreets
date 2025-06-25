import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { Card, Chip, Avatar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const TaskCard = ({ 
  task, 
  onPress,
  onStatusUpdate,
  index = 0 
}) => {
  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high':
      case 'urgent':
        return theme.colors.error;
      case 'medium':
        return theme.colors.warning;
      case 'low':
        return theme.colors.success;
      default:
        return theme.colors.primary;
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return theme.colors.success;
      case 'in_progress':
        return theme.colors.info;
      case 'pending':
        return theme.colors.warning;
      case 'cancelled':
        return theme.colors.error;
      default:
        return theme.colors.outline;
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return 'check-circle';
      case 'in_progress':
        return 'progress-wrench';
      case 'pending':
        return 'clock-outline';
      case 'cancelled':
        return 'close-circle';
      default:
        return 'help-circle';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return 'Due today';
    if (diffDays === 2) return 'Due tomorrow';
    if (diffDays <= 7) return `Due in ${diffDays} days`;
    return date.toLocaleDateString();
  };

  const isOverdue = (dueDate) => {
    if (!dueDate) return false;
    return new Date(dueDate) < new Date() && task.status !== 'completed';
  };

  return (
    <Animatable.View
      animation="fadeInUp"
      duration={600}
      delay={index * 100}
    >
      <TouchableOpacity onPress={onPress} activeOpacity={0.8}>
        <Card style={[
          styles.taskCard,
          isOverdue(task.dueDate) && styles.overdueCard
        ]}>
          <LinearGradient
            colors={[
              theme.colors.surface,
              task.priority === 'high' 
                ? theme.colors.errorContainer 
                : theme.colors.surfaceVariant
            ]}
            style={styles.cardGradient}
          >
            <View style={styles.cardContent}>
              {/* Header */}
              <View style={styles.cardHeader}>
                <View style={styles.titleContainer}>
                  <Text style={styles.taskTitle} numberOfLines={2}>
                    {task.title}
                  </Text>
                  <View style={styles.badgeContainer}>
                    <Chip
                      style={[
                        styles.priorityChip,
                        { backgroundColor: getPriorityColor(task.priority) }
                      ]}
                      textStyle={styles.chipText}
                      compact
                    >
                      {task.priority?.toUpperCase() || 'NORMAL'}
                    </Chip>
                    <Chip
                      style={[
                        styles.statusChip,
                        { backgroundColor: getStatusColor(task.status) }
                      ]}
                      textStyle={styles.chipText}
                      icon={() => (
                        <MaterialCommunityIcons
                          name={getStatusIcon(task.status)}
                          size={14}
                          color={theme.colors.surface}
                        />
                      )}
                      compact
                    >
                      {task.status?.replace('_', ' ').toUpperCase() || 'PENDING'}
                    </Chip>
                  </View>
                </View>
              </View>

              {/* Description */}
              {task.description && (
                <Text style={styles.taskDescription} numberOfLines={3}>
                  {task.description}
                </Text>
              )}

              {/* Location */}
              {task.location && (
                <View style={styles.locationContainer}>
                  <MaterialCommunityIcons
                    name="map-marker"
                    size={16}
                    color={theme.colors.primary}
                  />
                  <Text style={styles.locationText} numberOfLines={1}>
                    {task.location}
                  </Text>
                </View>
              )}

              {/* Task Image */}
              {task.imageUrl && (
                <View style={styles.imageContainer}>
                  <Image
                    source={{ uri: task.imageUrl }}
                    style={styles.taskImage}
                    resizeMode="cover"
                  />
                  <LinearGradient
                    colors={[
                      'transparent',
                      'rgba(0,0,0,0.3)'
                    ]}
                    style={styles.imageOverlay}
                  />
                </View>
              )}

              {/* Footer */}
              <View style={styles.cardFooter}>
                <View style={styles.dateContainer}>
                  <MaterialCommunityIcons
                    name={isOverdue(task.dueDate) ? "alert-circle" : "calendar"}
                    size={16}
                    color={isOverdue(task.dueDate) ? theme.colors.error : theme.colors.onSurfaceVariant}
                  />
                  <Text style={[
                    styles.dueDateText,
                    isOverdue(task.dueDate) && styles.overdueText
                  ]}>
                    {task.dueDate ? formatDate(task.dueDate) : 'No due date'}
                  </Text>
                </View>

                {/* Progress Indicator */}
                {task.progress !== undefined && (
                  <View style={styles.progressContainer}>
                    <Text style={styles.progressText}>
                      {Math.round(task.progress)}%
                    </Text>
                    <View style={styles.progressBar}>
                      <View style={[
                        styles.progressFill,
                        { 
                          width: `${task.progress}%`,
                          backgroundColor: getStatusColor(task.status)
                        }
                      ]} />
                    </View>
                  </View>
                )}

                {/* Assigned Worker */}
                {task.assignedTo && (
                  <View style={styles.assignedContainer}>
                    <Avatar.Text
                      size={24}
                      label={task.assignedTo.charAt(0).toUpperCase()}
                      style={styles.assignedAvatar}
                      labelStyle={styles.assignedAvatarLabel}
                    />
                    <Text style={styles.assignedText} numberOfLines={1}>
                      {task.assignedTo}
                    </Text>
                  </View>
                )}
              </View>

              {/* Quick Actions */}
              {task.status !== 'completed' && (
                <View style={styles.actionsContainer}>
                  {task.status === 'pending' && (
                    <TouchableOpacity
                      style={[styles.actionButton, styles.startButton]}
                      onPress={() => onStatusUpdate && onStatusUpdate(task.id, 'in_progress')}
                    >
                      <MaterialCommunityIcons
                        name="play"
                        size={16}
                        color={theme.colors.surface}
                      />
                      <Text style={styles.actionButtonText}>Start</Text>
                    </TouchableOpacity>
                  )}

                  {task.status === 'in_progress' && (
                    <TouchableOpacity
                      style={[styles.actionButton, styles.completeButton]}
                      onPress={() => onStatusUpdate && onStatusUpdate(task.id, 'completed')}
                    >
                      <MaterialCommunityIcons
                        name="check"
                        size={16}
                        color={theme.colors.surface}
                      />
                      <Text style={styles.actionButtonText}>Complete</Text>
                    </TouchableOpacity>
                  )}

                  <TouchableOpacity
                    style={[styles.actionButton, styles.viewButton]}
                    onPress={onPress}
                  >
                    <MaterialCommunityIcons
                      name="eye"
                      size={16}
                      color={theme.colors.surface}
                    />
                    <Text style={styles.actionButtonText}>View</Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>
          </LinearGradient>
        </Card>
      </TouchableOpacity>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  taskCard: {
    marginHorizontal: 16,
    marginVertical: 8,
    borderRadius: theme.borderRadius.large,
    overflow: 'hidden',
    ...theme.shadows.medium,
  },
  overdueCard: {
    borderLeftWidth: 4,
    borderLeftColor: theme.colors.error,
  },
  cardGradient: {
    borderRadius: theme.borderRadius.large,
  },
  cardContent: {
    padding: 20,
  },
  cardHeader: {
    marginBottom: 12,
  },
  titleContainer: {
    gap: 8,
  },
  taskTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: theme.colors.onSurface,
    lineHeight: 24,
    ...theme.typography.titleMedium,
  },
  badgeContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  priorityChip: {
    borderRadius: theme.borderRadius.small,
  },
  statusChip: {
    borderRadius: theme.borderRadius.small,
  },
  chipText: {
    fontSize: 10,
    fontWeight: '600',
    color: theme.colors.surface,
  },
  taskDescription: {
    fontSize: 14,
    color: theme.colors.onSurfaceVariant,
    lineHeight: 20,
    marginBottom: 12,
  },
  locationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 12,
  },
  locationText: {
    fontSize: 14,
    color: theme.colors.onSurfaceVariant,
    flex: 1,
  },
  imageContainer: {
    position: 'relative',
    borderRadius: theme.borderRadius.medium,
    overflow: 'hidden',
    marginBottom: 16,
  },
  taskImage: {
    width: '100%',
    height: 120,
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 40,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    flexWrap: 'wrap',
    gap: 8,
  },
  dateContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dueDateText: {
    fontSize: 13,
    color: theme.colors.onSurfaceVariant,
    fontWeight: '500',
  },
  overdueText: {
    color: theme.colors.error,
    fontWeight: '600',
  },
  progressContainer: {
    alignItems: 'center',
    gap: 4,
  },
  progressText: {
    fontSize: 12,
    color: theme.colors.onSurfaceVariant,
    fontWeight: '600',
  },
  progressBar: {
    width: 60,
    height: 4,
    backgroundColor: theme.colors.surfaceVariant,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  assignedContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  assignedAvatar: {
    backgroundColor: theme.colors.primary,
  },
  assignedAvatarLabel: {
    fontSize: 10,
    fontWeight: '600',
  },
  assignedText: {
    fontSize: 12,
    color: theme.colors.onSurfaceVariant,
    fontWeight: '500',
    maxWidth: 80,
  },
  actionsContainer: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: theme.borderRadius.small,
    flex: 1,
    justifyContent: 'center',
  },
  startButton: {
    backgroundColor: theme.colors.info,
  },
  completeButton: {
    backgroundColor: theme.colors.success,
  },
  viewButton: {
    backgroundColor: theme.colors.primary,
  },
  actionButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: theme.colors.surface,
  },
});

export default TaskCard;
