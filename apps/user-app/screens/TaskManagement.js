// screens/TaskManagement.js
import React, { useState, useEffect, useCallback } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  FlatList, 
  TouchableOpacity, 
  RefreshControl,
  Image,
  Alert,
  SafeAreaView,
  ScrollView,
  StatusBar
} from 'react-native';
import { 
  Card, 
  Title, 
  Paragraph, 
  Chip, 
  ActivityIndicator, 
  Button, 
  TextInput, 
  Divider,
  Avatar,
  Badge,
  FAB,
  Portal,
  Surface
} from 'react-native-paper';
import { MaterialIcons, Ionicons, FontAwesome5 } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';

const TaskManagement = ({ navigation }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTask, setSelectedTask] = useState(null);
  const [completionNotes, setCompletionNotes] = useState('');
  const [afterRepairImage, setAfterRepairImage] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [filterStatus, setFilterStatus] = useState('all');

  // Mock data for UI demonstration
  const mockTasks = [
    {
      id: 1,
      title: 'Repair pothole on Main Street',
      description: 'Fill and patch the pothole near 123 Main St intersection',
      status: 'assigned',
      priority: 'high',
      dueDate: new Date(Date.now() + 172800000).toISOString(), // 2 days from now
      reportId: 1,
      beforeImage: 'https://via.placeholder.com/300',
      location: '123 Main St, San Francisco, CA',
      assignedAt: new Date().toISOString()
    },
    {
      id: 2,
      title: 'Fix broken streetlight',
      description: 'Replace bulb and check wiring for streetlight at Oak and Pine',
      status: 'in_progress',
      priority: 'medium',
      dueDate: new Date(Date.now() + 86400000).toISOString(), // 1 day from now
      reportId: 2,
      beforeImage: 'https://via.placeholder.com/300',
      location: 'Corner of Oak and Pine, San Francisco, CA',
      assignedAt: new Date(Date.now() - 86400000).toISOString()
    },
    {
      id: 3,
      title: 'Remove graffiti',
      description: 'Clean graffiti from public building wall using approved cleaning solution',
      status: 'completed',
      priority: 'low',
      dueDate: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
      reportId: 3,
      beforeImage: 'https://via.placeholder.com/300',
      afterImage: 'https://via.placeholder.com/300',
      location: '456 Park Ave, San Francisco, CA',
      assignedAt: new Date(Date.now() - 172800000).toISOString(),
      completedAt: new Date(Date.now() - 43200000).toISOString(),
      completionNotes: 'Removed all graffiti and applied anti-graffiti coating'
    }
  ];

  // Fetch tasks
  const fetchTasks = useCallback(async () => {
    try {
      // Simulate API call delay
      setTimeout(() => {
        setTasks(mockTasks);
        setLoading(false);
        setRefreshing(false);
      }, 1000);
    } catch (error) {
      console.error('Error fetching tasks:', error);
      Alert.alert('Error', 'Failed to load tasks');
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchTasks();
    }, [fetchTasks])
  );

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchTasks();
  }, [fetchTasks]);

  const handleTaskPress = (task) => {
    setSelectedTask(task);
    if (task.status === 'completed') {
      setCompletionNotes(task.completionNotes || '');
      setAfterRepairImage(task.afterImage || null);
    } else {
      setCompletionNotes('');
      setAfterRepairImage(null);
    }
  };

  const handleStatusUpdate = (taskId, newStatus) => {
    setTasks(tasks.map(task => 
      task.id === taskId ? { ...task, status: newStatus } : task
    ));
    
    if (newStatus === 'in_progress') {
      Alert.alert('Task Started', 'You have started working on this task');
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });
      
      if (!result.canceled) {
        setAfterRepairImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to capture image');
    }
  };

  const handleSubmitCompletion = async () => {
    if (!afterRepairImage) {
      Alert.alert('Missing Image', 'Please take an after-repair photo');
      return;
    }

    if (!completionNotes.trim()) {
      Alert.alert('Missing Notes', 'Please add completion notes');
      return;
    }

    setSubmitting(true);
    
    try {
      // Simulate API call
      setTimeout(() => {
        setTasks(tasks.map(task => 
          task.id === selectedTask.id 
            ? { 
                ...task, 
                status: 'completed', 
                completedAt: new Date().toISOString(),
                completionNotes: completionNotes,
                afterImage: afterRepairImage
              } 
            : task
        ));
        
        setSelectedTask(null);
        setSubmitting(false);
        Alert.alert('Success', 'Task marked as completed');
      }, 1500);
    } catch (error) {
      console.error('Error submitting task completion:', error);
      Alert.alert('Error', 'Failed to submit task completion');
      setSubmitting(false);
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return '#e74c3c'; // Red
      case 'medium':
        return '#f39c12'; // Orange
      case 'low':
        return '#3498db'; // Blue
      default:
        return '#95a5a6'; // Gray
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'assigned':
        return '#f39c12'; // Orange
      case 'in_progress':
        return '#3498db'; // Blue
      case 'completed':
        return '#2ecc71'; // Green
      default:
        return '#95a5a6'; // Gray
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getFilteredTasks = () => {
    if (filterStatus === 'all') return tasks;
    return tasks.filter(task => task.status === filterStatus);
  };

  const renderTaskItem = ({ item }) => (
    <TouchableOpacity onPress={() => handleTaskPress(item)}>
      <Surface style={styles.taskCard}>
        <LinearGradient
          colors={['#ffffff', '#f8f9fa']}
          style={styles.cardGradient}
        >
          <View style={styles.taskHeader}>
            <View style={styles.taskTitleContainer}>
              <Badge 
                style={[styles.priorityBadge, { backgroundColor: getPriorityColor(item.priority) }]}
                size={14}
              />
              <Title style={styles.taskTitle}>{item.title}</Title>
            </View>
            <Chip 
              style={[styles.statusChip, { backgroundColor: getStatusColor(item.status) }]}
              textStyle={styles.chipText}
            >
              {item.status.replace('_', ' ').toUpperCase()}
            </Chip>
          </View>
          
          <Paragraph style={styles.taskDescription} numberOfLines={2}>
            {item.description}
          </Paragraph>
          
          <View style={styles.taskDetails}>
            <View style={styles.taskDetail}>
              <MaterialIcons name="location-on" size={16} color="#555" />
              <Text style={styles.detailText} numberOfLines={1}>{item.location}</Text>
            </View>
            <View style={styles.taskDetail}>
              <MaterialIcons name="event" size={16} color="#555" />
              <Text style={styles.detailText}>Due: {formatDate(item.dueDate)}</Text>
            </View>
          </View>
          
          <View style={styles.taskFooter}>
            <View style={styles.taskImagePreview}>
              <Image source={{ uri: item.beforeImage }} style={styles.previewImage} />
              <View style={styles.imageOverlay} />
              <MaterialIcons name="image" size={20} color="#fff" style={styles.imageIcon} />
            </View>
            
            {item.status !== 'completed' && (
              <Button 
                mode="contained" 
                onPress={() => handleStatusUpdate(item.id, item.status === 'assigned' ? 'in_progress' : 'completed')}
                style={[styles.actionButton, { backgroundColor: getStatusColor(item.status === 'assigned' ? 'in_progress' : 'completed') }]}
                labelStyle={styles.actionButtonLabel}
                icon={item.status === 'assigned' ? 'play-arrow' : 'check'}
              >
                {item.status === 'assigned' ? 'Start Task' : 'Mark Complete'}
              </Button>
            )}
          </View>
        </LinearGradient>
      </Surface>
    </TouchableOpacity>
  );

  const renderTaskDetail = () => {
    if (!selectedTask) return null;
    
    return (
      <View style={styles.detailContainer}>
        <LinearGradient
          colors={['#3498db', '#2980b9']}
          style={styles.detailHeader}
        >
          <TouchableOpacity onPress={() => setSelectedTask(null)} style={styles.backButton}>
            <MaterialIcons name="arrow-back" size={24} color="#fff" />
          </TouchableOpacity>
          <Title style={styles.detailTitle}>{selectedTask.title}</Title>
        </LinearGradient>
        
        <ScrollView style={styles.detailScroll}>
          <Surface style={styles.detailCard}>
            <Paragraph style={styles.detailDescription}>{selectedTask.description}</Paragraph>
            
            <View style={styles.detailInfoRow}>
              <View style={styles.detailInfo}>
                <Text style={styles.detailLabel}>Priority:</Text>
                <Chip 
                  style={[styles.detailChip, { backgroundColor: getPriorityColor(selectedTask.priority) }]}
                  textStyle={styles.chipText}
                >
                  {selectedTask.priority.toUpperCase()}
                </Chip>
              </View>
              <View style={styles.detailInfo}>
                <Text style={styles.detailLabel}>Status:</Text>
                <Chip 
                  style={[styles.detailChip, { backgroundColor: getStatusColor(selectedTask.status) }]}
                  textStyle={styles.chipText}
                >
                  {selectedTask.status.replace('_', ' ').toUpperCase()}
                </Chip>
              </View>
            </View>
            
            <View style={styles.detailInfoRow}>
              <View style={styles.detailInfo}>
                <Text style={styles.detailLabel}>Assigned:</Text>
                <Text style={styles.detailValue}>{formatDate(selectedTask.assignedAt)}</Text>
              </View>
              <View style={styles.detailInfo}>
                <Text style={styles.detailLabel}>Due Date:</Text>
                <Text style={styles.detailValue}>{formatDate(selectedTask.dueDate)}</Text>
              </View>
            </View>
            
            <View style={styles.detailLocation}>
              <MaterialIcons name="location-on" size={20} color="#3498db" />
              <Text style={styles.locationText}>{selectedTask.location}</Text>
            </View>
          </Surface>
          
          <Surface style={styles.detailCard}>
            <Text style={styles.imagesTitle}>Before Image:</Text>
            <View style={styles.imageContainer}>
              <Image source={{ uri: selectedTask.beforeImage }} style={styles.detailImage} />
            </View>
          </Surface>
          
          {selectedTask.status === 'completed' ? (
            <Surface style={styles.detailCard}>
              <Text style={styles.imagesTitle}>After Image:</Text>
              <View style={styles.imageContainer}>
                <Image source={{ uri: selectedTask.afterImage }} style={styles.detailImage} />
              </View>
              
              <Text style={styles.notesTitle}>Completion Notes:</Text>
              <View style={styles.notesContainer}>
                <Paragraph style={styles.notesText}>{selectedTask.completionNotes}</Paragraph>
              </View>
              
              <View style={styles.completedContainer}>
                <LinearGradient
                  colors={['#2ecc71', '#27ae60']}
                  style={styles.completedBadge}
                >
                  <MaterialIcons name="check-circle" size={20} color="#fff" />
                  <Text style={styles.completedText}>
                    Completed on: {formatDate(selectedTask.completedAt)}
                  </Text>
                </LinearGradient>
              </View>
            </Surface>
          ) : (
            <Surface style={styles.detailCard}>
              <LinearGradient
                colors={['#3498db', '#2980b9']}
                style={styles.completionHeader}
              >
                <Text style={styles.completionTitle}>Complete this task</Text>
              </LinearGradient>
              
              <Text style={styles.imagesTitle}>After Repair Image:</Text>
              {afterRepairImage ? (
                <View style={styles.afterImageContainer}>
                  <View style={styles.imageContainer}>
                    <Image source={{ uri: afterRepairImage }} style={styles.detailImage} />
                  </View>
                  <Button 
                    mode="contained" 
                    icon="camera" 
                    onPress={pickImage}
                    style={styles.retakeButton}
                  >
                    Retake Photo
                  </Button>
                </View>
              ) : (
                <TouchableOpacity 
                  style={styles.cameraButton} 
                  onPress={pickImage}
                >
                  <LinearGradient
                    colors={['rgba(52, 152, 219, 0.1)', 'rgba(52, 152, 219, 0.2)']}
                    style={styles.cameraGradient}
                  >
                    <View style={styles.cameraIconContainer}>
                      <MaterialIcons name="camera-alt" size={40} color="#3498db" />
                      <Text style={styles.cameraText}>Take After-Repair Photo</Text>
                    </View>
                  </LinearGradient>
                </TouchableOpacity>
              )}
              
              <Text style={styles.notesTitle}>Completion Notes:</Text>
              <TextInput
                mode="outlined"
                multiline
                numberOfLines={4}
                value={completionNotes}
                onChangeText={setCompletionNotes}
                placeholder="Describe the repairs completed..."
                style={styles.notesInput}
                theme={{ colors: { primary: '#3498db' } }}
              />
              
              <Button 
                mode="contained" 
                onPress={handleSubmitCompletion}
                style={styles.submitButton}
                loading={submitting}
                disabled={submitting || !afterRepairImage || !completionNotes.trim()}
                icon="check"
              >
                Submit Completed Task
              </Button>
            </Surface>
          )}
        </ScrollView>
      </View>
    );
  };

  if (loading && !refreshing) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading tasks...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#3498db" />
      {selectedTask ? (
        renderTaskDetail()
      ) : (
        <>
          <LinearGradient
            colors={['#3498db', '#2980b9']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.header}
          >
            <View style={styles.headerContent}>
              <Title style={styles.headerTitle}>My Tasks</Title>
              <MaterialIcons name="assignment" size={28} color="#fff" />
            </View>
            <View style={styles.filterContainer}>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <TouchableOpacity 
                  style={[styles.filterButton, filterStatus === 'all' && styles.filterButtonActive]} 
                  onPress={() => setFilterStatus('all')}
                >
                  <Text style={[styles.filterText, filterStatus === 'all' && styles.filterTextActive]}>All</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.filterButton, filterStatus === 'assigned' && styles.filterButtonActive]} 
                  onPress={() => setFilterStatus('assigned')}
                >
                  <Text style={[styles.filterText, filterStatus === 'assigned' && styles.filterTextActive]}>Assigned</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.filterButton, filterStatus === 'in_progress' && styles.filterButtonActive]} 
                  onPress={() => setFilterStatus('in_progress')}
                >
                  <Text style={[styles.filterText, filterStatus === 'in_progress' && styles.filterTextActive]}>In Progress</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.filterButton, filterStatus === 'completed' && styles.filterButtonActive]} 
                  onPress={() => setFilterStatus('completed')}
                >
                  <Text style={[styles.filterText, filterStatus === 'completed' && styles.filterTextActive]}>Completed</Text>
                </TouchableOpacity>
              </ScrollView>
            </View>
          </LinearGradient>
          
          <FlatList
            data={getFilteredTasks()}
            renderItem={renderTaskItem}
            keyExtractor={item => item.id.toString()}
            contentContainerStyle={styles.taskList}
            refreshControl={
              <RefreshControl
                refreshing={refreshing}
                onRefresh={onRefresh}
                colors={['#3498db']}
              />
            }
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <LinearGradient
                  colors={['rgba(52, 152, 219, 0.1)', 'rgba(52, 152, 219, 0.05)']}
                  style={styles.emptyGradient}
                >
                  <MaterialIcons name="assignment" size={64} color="#ccc" />
                  <Text style={styles.emptyText}>No tasks found</Text>
                  <Text style={styles.emptySubText}>
                    {filterStatus !== 'all' 
                      ? `Try changing your filter or check back later` 
                      : `You don't have any tasks assigned yet`}
                  </Text>
                </LinearGradient>
              </View>
            }
          />
          
          <FAB
            style={styles.fab}
            icon="refresh"
            onPress={onRefresh}
            color="#fff"
          />
        </>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 16,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
  },
  filterContainer: {
    marginTop: 8,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  filterButtonActive: {
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  filterText: {
    color: '#fff',
    fontWeight: '500',
  },
  filterTextActive: {
    color: '#3498db',
    fontWeight: 'bold',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#555',
  },
  taskList: {
    padding: 16,
    paddingBottom: 80,
  },
  taskCard: {
    marginBottom: 16,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardGradient: {
    padding: 16,
  },
  taskHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  taskTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  priorityBadge: {
    marginRight: 8,
    elevation: 2,
  },
  taskTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    flex: 1,
    color: '#333',
  },
  chipText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  taskDescription: {
    color: '#555',
    marginBottom: 12,
    lineHeight: 20,
  },
  taskDetails: {
    marginBottom: 12,
  },
  taskDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  detailText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#555',
  },
  taskFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusChip: {
    height: 28,
    elevation: 2,
  },
  taskImagePreview: {
    width: 60,
    height: 60,
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  imageOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.2)',
  },
  imageIcon: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    marginLeft: -10,
    marginTop: -10,
  },
  actionButton: {
    borderRadius: 8,
    elevation: 2,
    paddingHorizontal: 12,
  },
  actionButtonLabel: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    marginTop: 40,
  },
  emptyGradient: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    borderRadius: 16,
    width: '100%',
  },
  emptyText: {
    marginTop: 16,
    fontSize: 20,
    color: '#555',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  emptySubText: {
    marginTop: 8,
    fontSize: 14,
    color: '#777',
    textAlign: 'center',
  },
  detailContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  detailHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    paddingTop: 20,
    paddingBottom: 20,
    elevation: 4,
  },
  backButton: {
    marginRight: 16,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 20,
    padding: 8,
  },
  detailTitle: {
    flex: 1,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  detailScroll: {
    flex: 1,
    padding: 16,
  },
  detailCard: {
    padding: 16,
    marginBottom: 16,
    borderRadius: 16,
    backgroundColor: '#fff',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  detailDescription: {
    fontSize: 16,
    color: '#333',
    marginBottom: 16,
    lineHeight: 24,
  },
  detailInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  detailInfo: {
    flex: 1,
  },
  detailLabel: {
    fontSize: 14,
    color: '#777',
    marginBottom: 4,
  },
  detailValue: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  detailChip: {
    alignSelf: 'flex-start',
    elevation: 1,
  },
  detailLocation: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 12,
    padding: 12,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#3498db',
  },
  locationText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  imagesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 8,
    marginBottom: 12,
    color: '#333',
  },
  imageContainer: {
    borderRadius: 12,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  detailImage: {
    height: 200,
    width: '100%',
  },
  notesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 12,
    color: '#333',
  },
  notesContainer: {
    backgroundColor: '#f9f9f9',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#3498db',
  },
  notesText: {
    fontSize: 15,
    color: '#333',
    lineHeight: 22,
  },
  completedContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
  },
  completedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  completedText: {
    fontSize: 14,
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  completionHeader: {
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    alignItems: 'center',
  },
  completionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  cameraButton: {
    height: 160,
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#3498db',
    borderStyle: 'dashed',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    backgroundColor: 'rgba(52, 152, 219, 0.05)',
  },
  cameraIconContainer: {
    alignItems: 'center',
  },
  cameraText: {
    marginTop: 8,
    color: '#3498db',
    fontSize: 16,
  },
  afterImageContainer: {
    marginBottom: 16,
  },
  retakeButton: {
    marginTop: -8,
    marginBottom: 16,
    alignSelf: 'flex-end',
  },
  notesInput: {
    marginBottom: 16,
    backgroundColor: 'white',
  },
  submitButton: {
    marginVertical: 8,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#2ecc71',
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#3498db',
  }
});

export default TaskManagement;
