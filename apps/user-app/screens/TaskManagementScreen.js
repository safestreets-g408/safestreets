import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  FlatList,
  TouchableOpacity,
  Alert,
  RefreshControl,
  StatusBar,
  Dimensions,
  Platform
} from 'react-native';
import {
  useTheme,
  Button,
  Chip,
  ActivityIndicator,
  Searchbar,
  FAB,
  Snackbar,
  IconButton
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ModernCard, ConsistentHeader } from '../components/ui';
import { SafeAreaView } from 'react-native-safe-area-context';
import { getFieldWorkerTasks, updateTaskStatus as updateTaskStatusAPI, transformReportToTask } from '../utils/taskAPI';
import { useFocusEffect } from '@react-navigation/native';

const { width: screenWidth } = Dimensions.get('window');

const TaskManagementScreen = ({ navigation }) => {
  const theme = useTheme();
  
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  useEffect(() => {
    loadTasks();
  }, []);

  // Reload tasks when screen comes into focus (e.g., returning from camera)
  useFocusEffect(
    useCallback(() => {
      loadTasks();
    }, [])
  );

  const loadTasks = async () => {
    try {
      setLoading(true);
      
      // Fetch tasks from API
      const reports = await getFieldWorkerTasks();
      
      // Transform damage reports to task format
      const transformedTasks = reports.map(transformReportToTask);
      
      setTasks(transformedTasks);
    } catch (error) {
      console.error('Error loading tasks:', error);
      Alert.alert('Error', 'Failed to load tasks. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadTasks();
    setRefreshing(false);
  };

  const updateTaskStatus = async (taskId, newStatus) => {
    try {
      // If completing a task, show confirmation and navigate to after image capture
      if (newStatus === 'completed') {
        const task = tasks.find(t => t.id === taskId);
        if (!task) return;
        
        Alert.alert(
          'Complete Task',
          'To complete this task, you need to capture an after image showing the repaired area.',
          [
            {
              text: 'Cancel',
              style: 'cancel'
            },
            {
              text: 'Capture Image',
              onPress: () => {
                navigation.navigate('AfterImageCamera', {
                  reportId: task.id,
                  taskTitle: task.title,
                  onComplete: () => completeTaskWithAfterImage(task.id)
                });
              }
            }
          ]
        );
        return;
      }

      // If putting task on hold, show confirmation
      if (newStatus === 'on_hold') {
        const task = tasks.find(t => t.id === taskId);
        if (!task) return;
        
        Alert.alert(
          'Put Task On Hold',
          'Are you sure you want to put this task on hold? You can resume it later.',
          [
            {
              text: 'Cancel',
              style: 'cancel'
            },
            {
              text: 'Put On Hold',
              onPress: async () => {
                try {
                  await updateTaskStatusAPI(taskId, 'on_hold');
                  setTasks(prevTasks => 
                    prevTasks.map(task => 
                      task.id === taskId ? { ...task, status: 'on_hold' } : task
                    )
                  );
                  setSnackbarMessage('Task put on hold');
                  setSnackbarVisible(true);
                } catch (error) {
                  console.error('Error updating task status:', error);
                  Alert.alert('Error', error.message || 'Failed to update task status');
                }
              }
            }
          ]
        );
        return;
      }
      
      // Update status via API
      const updatedReport = await updateTaskStatusAPI(taskId, newStatus);
      
      // Update local state
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.id === taskId ? { ...task, status: newStatus } : task
        )
      );
      
      setSnackbarMessage(`Task status updated to ${newStatus.replace('_', ' ')}`);
      setSnackbarVisible(true);
    } catch (error) {
      console.error('Error updating task status:', error);
      Alert.alert('Error', error.message || 'Failed to update task status');
    }
  };

  const completeTaskWithAfterImage = async (taskId) => {
    try {
      // Update status to completed via API
      const updatedReport = await updateTaskStatusAPI(taskId, 'completed');
      
      // Update local state
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.id === taskId ? { ...task, status: 'completed', hasAfterImage: true } : task
        )
      );
      
      setSnackbarMessage('Task completed successfully!');
      setSnackbarVisible(true);
    } catch (error) {
      console.error('Error completing task:', error);
      Alert.alert('Error', error.message || 'Failed to complete task');
    }
  };

  const getStatusConfig = (status) => {
    const configs = {
      pending: {
        color: theme.colors.warning,
        backgroundColor: theme.colors.warning + '20',
        icon: 'clock-outline',
        label: 'Pending'
      },
      in_progress: {
        color: theme.colors.primary,
        backgroundColor: theme.colors.primary + '20',
        icon: 'progress-clock',
        label: 'In Progress'
      },
      completed: {
        color: theme.colors.success,
        backgroundColor: theme.colors.success + '20',
        icon: 'check-circle',
        label: 'Completed'
      },
      on_hold: {
        color: theme.colors.error,
        backgroundColor: theme.colors.error + '20',
        icon: 'pause-circle',
        label: 'On Hold'
      }
    };
    return configs[status] || configs.pending;
  };

  const getPriorityConfig = (priority) => {
    const configs = {
      low: { color: theme.colors.success, label: 'Low' },
      medium: { color: theme.colors.warning, label: 'Medium' },
      high: { color: theme.colors.error, label: 'High' },
      urgent: { color: theme.colors.error, label: 'Urgent' }
    };
    return configs[priority?.toLowerCase()] || configs.low;
  };

  const filteredTasks = tasks.filter(task => {
    const matchesSearch = task.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         task.location.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = selectedFilter === 'all' || task.status === selectedFilter;
    return matchesSearch && matchesFilter;
  });

  const getTaskCounts = () => {
    return {
      all: tasks.length,
      pending: tasks.filter(t => t.status === 'pending').length,
      in_progress: tasks.filter(t => t.status === 'in_progress').length,
      completed: tasks.filter(t => t.status === 'completed').length,
      on_hold: tasks.filter(t => t.status === 'on_hold').length
    };
  };

  const taskCounts = getTaskCounts();

  const renderTaskItem = ({ item, index }) => {
    const statusConfig = getStatusConfig(item.status);
    const priorityConfig = getPriorityConfig(item.priority);
    
    return (
      <Animatable.View 
        animation="fadeInUp" 
        duration={600} 
        delay={index * 100}
        style={styles.taskItemContainer}
      >
        <ModernCard 
          style={styles.taskCard}
          onPress={() => navigation.navigate('ViewReport', { reportId: item.id })}
          elevation={2}
        >
          <View style={styles.taskHeader}>
            <View style={styles.taskTitleContainer}>
              <Text style={[styles.taskTitle, { color: theme.colors.text }]}>
                {item.title}
              </Text>
              <View style={styles.taskMeta}>
                <MaterialCommunityIcons 
                  name="map-marker" 
                  size={14} 
                  color={theme.colors.textSecondary} 
                />
                <Text style={[styles.taskLocation, { color: theme.colors.textSecondary }]}>
                  {item.location}
                </Text>
              </View>
            </View>
            
            <View style={styles.statusContainer}>
              <View style={[styles.statusBadge, { backgroundColor: statusConfig.backgroundColor }]}>
                <MaterialCommunityIcons 
                  name={statusConfig.icon} 
                  size={16} 
                  color={statusConfig.color} 
                />
                <Text style={[styles.statusText, { color: statusConfig.color }]}>
                  {statusConfig.label}
                </Text>
              </View>
              
              <View style={styles.viewDetailsIndicator}>
                <MaterialCommunityIcons 
                  name="chevron-right" 
                  size={16} 
                  color={theme.colors.primary} 
                />
              </View>
            </View>
          </View>

          <Text style={[styles.taskDescription, { color: theme.colors.textSecondary }]}>
            {item.description.length > 80 
              ? `${item.description.substring(0, 80)}...` 
              : item.description
            }
            {item.description.length > 80 && (
              <Text style={[styles.readMoreText, { color: theme.colors.primary }]}>
                {' '}Tap to read more
              </Text>
            )}
          </Text>

          <View style={styles.taskFooter}>
            <View style={styles.taskInfo}>
              <View style={styles.infoItem}>
                <MaterialCommunityIcons 
                  name="flag" 
                  size={14} 
                  color={priorityConfig.color} 
                />
                <Text style={[styles.infoText, { color: priorityConfig.color }]}>
                  {priorityConfig.label}
                </Text>
              </View>
              
              <View style={styles.infoItem}>
                <MaterialCommunityIcons 
                  name="clock-outline" 
                  size={14} 
                  color={theme.colors.textSecondary} 
                />
                <Text style={[styles.infoText, { color: theme.colors.textSecondary }]}>
                  {item.estimatedDuration}
                </Text>
              </View>
              
              <View style={styles.infoItem}>
                <MaterialCommunityIcons 
                  name="calendar" 
                  size={14} 
                  color={theme.colors.textSecondary} 
                />
                <Text style={[styles.infoText, { color: theme.colors.textSecondary }]}>
                  {new Date(item.dueDate).toLocaleDateString()}
                </Text>
              </View>

              {item.status === 'completed' && item.hasAfterImage && (
                <View style={styles.infoItem}>
                  <MaterialCommunityIcons 
                    name="camera-check" 
                    size={14} 
                    color={theme.colors.success} 
                  />
                  <Text style={[styles.infoText, { color: theme.colors.success }]}>
                    After Image
                  </Text>
                </View>
              )}
            </View>

            <View style={styles.taskActions}>
              {item.status === 'pending' && (
                <>
                  <Button
                    mode="contained"
                    compact
                    onPress={() => updateTaskStatus(item.id, 'in_progress')}
                    style={[styles.actionButton, { backgroundColor: theme.colors.primary }]}
                    labelStyle={styles.actionButtonLabel}
                  >
                    Start Work
                  </Button>
                  <Button
                    mode="outlined"
                    compact
                    onPress={() => updateTaskStatus(item.id, 'on_hold')}
                    style={[styles.actionButton, { borderColor: theme.colors.error }]}
                    labelStyle={[styles.actionButtonLabel, { color: theme.colors.error }]}
                  >
                    Put on Hold
                  </Button>
                </>
              )}
              
              {item.status === 'in_progress' && (
                <>
                  <Button
                    mode="contained"
                    compact
                    onPress={() => updateTaskStatus(item.id, 'completed')}
                    style={[styles.actionButton, { backgroundColor: theme.colors.success }]}
                    labelStyle={styles.actionButtonLabel}
                  >
                    Mark Complete
                  </Button>
                  <Button
                    mode="outlined"
                    compact
                    onPress={() => updateTaskStatus(item.id, 'on_hold')}
                    style={[styles.actionButton, { borderColor: theme.colors.error }]}
                    labelStyle={[styles.actionButtonLabel, { color: theme.colors.error }]}
                  >
                    Put on Hold
                  </Button>
                </>
              )}

              {item.status === 'on_hold' && (
                <Button
                  mode="contained"
                  compact
                  onPress={() => updateTaskStatus(item.id, 'in_progress')}
                  style={[styles.actionButton, { backgroundColor: theme.colors.primary }]}
                  labelStyle={styles.actionButtonLabel}
                >
                  Resume Work
                </Button>
              )}

              {item.status === 'completed' && !item.hasAfterImage && (
                <Button
                  mode="outlined"
                  compact
                  onPress={() => navigation.navigate('AfterImageCamera', {
                    reportId: item.id,
                    taskTitle: item.title,
                    onComplete: () => {
                      // Just refresh the tasks to show the updated after image status
                      loadTasks();
                    }
                  })}
                  style={[styles.actionButton, { borderColor: theme.colors.warning }]}
                  labelStyle={{ color: theme.colors.warning }}
                >
                  Upload After Image
                </Button>
              )}

              {item.status === 'completed' && item.hasAfterImage && (
                <Chip
                  icon="check-circle"
                  style={[styles.completedChip, { backgroundColor: theme.colors.success + '20' }]}
                  textStyle={{ color: theme.colors.success }}
                >
                  Task Resolved
                </Chip>
              )}
            </View>
          </View>
        </ModernCard>
      </Animatable.View>
    );
  };

  const FilterChip = ({ status, label, count }) => (
    <TouchableOpacity onPress={() => setSelectedFilter(status)}>
      <Chip
        mode={selectedFilter === status ? 'flat' : 'outlined'}
        selected={selectedFilter === status}
        style={[
          styles.filterChip,
          selectedFilter === status && { backgroundColor: theme.colors.primary }
        ]}
        textStyle={[
          styles.filterChipText,
          selectedFilter === status && { color: '#ffffff' }
        ]}
      >
        {label} ({count})
      </Chip>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>
          Loading tasks...
        </Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />
      
      {/* Header - with iOS optimizations */}
      <ConsistentHeader
        title="Task Management"
        subtitle={`${taskCounts.all} tasks assigned`}
        useGradient={true}
        elevated={true}
        back={{
          visible: true,
          onPress: () => navigation.goBack()
        }}
        blurEffect={Platform.OS === 'ios'}
        actions={[
          {
            icon: 'refresh',
            onPress: onRefresh
          }
        ]}
      />

      {/* Search and Filters */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search tasks..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchBar}
          inputStyle={styles.searchInput}
        />
        
        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false}
          style={styles.filtersContainer}
          contentContainerStyle={styles.filtersContent}
        >
          <FilterChip status="all" label="All" count={taskCounts.all} />
          <FilterChip status="pending" label="Pending" count={taskCounts.pending} />
          <FilterChip status="in_progress" label="In Progress" count={taskCounts.in_progress} />
          <FilterChip status="completed" label="Completed" count={taskCounts.completed} />
          <FilterChip status="on_hold" label="On Hold" count={taskCounts.on_hold} />
        </ScrollView>
      </View>

      {/* Task List */}
      <FlatList
        data={filteredTasks}
        renderItem={renderTaskItem}
        keyExtractor={(item) => item.id.toString()}
        style={styles.taskList}
        contentContainerStyle={styles.taskListContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <MaterialCommunityIcons 
              name="clipboard-check-outline" 
              size={64} 
              color={theme.colors.textSecondary} 
            />
            <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>
              No tasks found
            </Text>
            <Text style={[styles.emptySubText, { color: theme.colors.textSecondary }]}>
              {selectedFilter === 'all' 
                ? 'You have no assigned tasks at the moment'
                : `No ${selectedFilter.replace('_', ' ')} tasks found`
              }
            </Text>
          </View>
        }
      />

      {/* Floating Action Button */}
      <FAB
        icon="plus"
        style={[styles.fab, { backgroundColor: theme.colors.primary }]}
        onPress={() => navigation.navigate('Camera')}
        label="New Report"
      />

      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => setSnackbarVisible(false)}
        duration={3000}
        style={{ backgroundColor: theme.colors.success }}
      >
        {snackbarMessage}
      </Snackbar>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
  },
  header: {
    paddingTop: StatusBar.currentHeight || 44,
    paddingBottom: 16,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  headerTitle: {
    flex: 1,
    marginLeft: 8,
  },
  headerTitleText: {
    fontSize: 20,
    fontWeight: '700',
    color: '#ffffff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 2,
  },
  searchContainer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  searchBar: {
    marginBottom: 12,
    elevation: 2,
  },
  searchInput: {
    fontSize: 16,
  },
  filtersContainer: {
    marginHorizontal: -16,
    paddingHorizontal: 16,
  },
  filtersContent: {
    paddingRight: 16,
  },
  filterChip: {
    marginRight: 8,
  },
  filterChipText: {
    fontSize: 12,
  },
  taskList: {
    flex: 1,
  },
  taskListContent: {
    paddingHorizontal: 16,
    paddingBottom: 80,
  },
  taskItemContainer: {
    marginBottom: 12,
  },
  taskCard: {
    padding: 16,
  },
  taskHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  taskTitleContainer: {
    flex: 1,
    marginRight: 12,
  },
  taskTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  taskMeta: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  taskLocation: {
    fontSize: 12,
    marginLeft: 4,
  },
  statusContainer: {
    alignItems: 'flex-end',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginBottom: 4,
  },
  viewDetailsIndicator: {
    opacity: 0.6,
  },
  statusText: {
    marginLeft: 4,
    fontSize: 12,
    fontWeight: '500',
  },
  taskDescription: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 12,
  },
  readMoreText: {
    fontSize: 12,
    fontWeight: '500',
  },
  taskFooter: {
    flexDirection: 'column',
    gap: 12,
  },
  taskInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
    marginBottom: 4,
  },
  infoText: {
    marginLeft: 4,
    fontSize: 12,
  },
  taskActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
    gap: 8,
    marginTop: 8,
  },
  actionButton: {
    marginLeft: 0,
    minWidth: 100,
  },
  completedChip: {
    alignSelf: 'flex-start',
  },
  actionButtonLabel: {
    fontSize: 12,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 60,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubText: {
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
});

export default TaskManagementScreen;
