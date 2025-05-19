// screens/ReportsScreen.js
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
  StatusBar
} from 'react-native';
import { Card, Title, Paragraph, Chip, ActivityIndicator, Button, Surface, Divider } from 'react-native-paper';
import { useFocusEffect } from '@react-navigation/native';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';

const ReportsScreen = ({ navigation }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Mock data for UI demonstration
  const mockReports = [
    {
      id: 1,
      status: 'pending',
      imageUrl: 'https://via.placeholder.com/300',
      description: 'Pothole on Main Street that needs immediate attention',
      createdAt: new Date().toISOString(),
      latitude: 37.7749,
      longitude: -122.4194
    },
    {
      id: 2,
      status: 'in_progress',
      imageUrl: 'https://via.placeholder.com/300',
      description: 'Broken streetlight at the corner of Oak and Pine',
      createdAt: new Date(Date.now() - 86400000).toISOString(),
      latitude: 37.7750,
      longitude: -122.4180
    },
    {
      id: 3,
      status: 'resolved',
      imageUrl: 'https://via.placeholder.com/300',
      description: 'Graffiti on public building wall',
      createdAt: new Date(Date.now() - 172800000).toISOString(),
      latitude: 37.7752,
      longitude: -122.4175
    }
  ];

  // Fetch reports (mock implementation)
  const fetchReports = async () => {
    try {
      // Simulate API call delay
      setTimeout(() => {
        setReports(mockReports);
        setLoading(false);
        setRefreshing(false);
      }, 1000);
    } catch (error) {
      console.error('Error fetching reports:', error);
      Alert.alert('Error', 'Network error while fetching reports');
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Pull to refresh functionality
  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchReports();
  }, []);

  // Fetch reports when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      fetchReports();
    }, [])
  );

  // Format the date for better readability
  const formatDate = (dateString) => {
    const options = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  // Get status color based on report status
  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return '#f39c12';
      case 'in_progress': 
        return '#3498db';
      case 'resolved':
        return '#2ecc71';
      case 'rejected':
        return '#e74c3c';
      default:
        return '#95a5a6';
    }
  };

  // Get status icon based on report status
  const getStatusIcon = (status) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return 'clock-outline';
      case 'in_progress': 
        return 'progress-wrench';
      case 'resolved':
        return 'check-circle';
      case 'rejected':
        return 'close-circle';
      default:
        return 'help-circle';
    }
  };

  // Handle report selection
  const handleReportPress = (report) => {
    navigation.navigate('ViewReport', { reportId: report.id });
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading your reports...</Text>
      </View>
    );
  }

  if (reports.length === 0) {
    return (
      <View style={styles.centerContainer}>
        <Surface style={styles.emptySurface}>
          <Ionicons name="document-text-outline" size={64} color="#95a5a6" />
          <Text style={styles.emptyText}>No reports submitted yet</Text>
          <Text style={styles.emptySubText}>
            Submit a new report to track road issues in your area
          </Text>
          <Button 
            mode="contained"
            icon="camera"
            style={styles.emptyButton}
            onPress={() => navigation.navigate('Camera')}
          >
            Submit a Report
          </Button>
        </Surface>
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#f5f5f5" barStyle="dark-content" />
      
      <FlatList
        data={reports}
        keyExtractor={(item) => item.id.toString()}
        refreshControl={
          <RefreshControl 
            refreshing={refreshing} 
            onRefresh={onRefresh}
            colors={['#3498db']} 
          />
        }
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => (
          <TouchableOpacity 
            onPress={() => handleReportPress(item)}
            activeOpacity={0.7}
          >
            <Card style={styles.card}>
              <Card.Content>
                <View style={styles.cardHeader}>
                  <View style={styles.idContainer}>
                    <Text style={styles.idLabel}>ID</Text>
                    <Text style={styles.idNumber}>{item.id}</Text>
                  </View>
                  <Chip 
                    style={[
                      styles.statusChip, 
                      { backgroundColor: getStatusColor(item.status) }
                    ]}
                    icon={() => (
                      <Ionicons 
                        name={getStatusIcon(item.status)} 
                        size={14} 
                        color="white" 
                      />
                    )}
                  >
                    <Text style={styles.statusText}>
                      {item.status.replace('_', ' ')}
                    </Text>
                  </Chip>
                </View>
                
                <Divider style={styles.divider} />
                
                <View style={styles.cardContent}>
                  <Image 
                    source={{ uri: item.imageUrl }} 
                    style={styles.thumbnail} 
                  />
                  <View style={styles.details}>
                    <Paragraph numberOfLines={2} style={styles.description}>
                      {item.description || 'No description provided'}
                    </Paragraph>
                    <View style={styles.metaContainer}>
                      <View style={styles.metaItem}>
                        <Ionicons name="calendar-outline" size={14} color="#7f8c8d" />
                        <Text style={styles.metaText}>{formatDate(item.createdAt)}</Text>
                      </View>
                      <View style={styles.metaItem}>
                        <Ionicons name="location-outline" size={14} color="#7f8c8d" />
                        <Text style={styles.metaText}>
                          {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
                        </Text>
                      </View>
                    </View>
                  </View>
                </View>
              </Card.Content>
              <Card.Actions style={styles.cardActions}>
                <Button 
                  mode="text" 
                  compact 
                  onPress={() => handleReportPress(item)}
                  color="#3498db"
                >
                  View Details
                </Button>
              </Card.Actions>
            </Card>
          </TouchableOpacity>
        )}
      />
      
      <TouchableOpacity 
        style={styles.fab}
        onPress={() => navigation.navigate('Camera')}
      >
        <Ionicons name="camera" size={24} color="white" />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  listContent: {
    padding: 16,
    paddingBottom: 80, // Extra padding for FAB
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#3498db',
  },
  card: {
    marginBottom: 16,
    elevation: 3,
    borderRadius: 12,
    overflow: 'hidden',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  idContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  idLabel: {
    fontSize: 14,
    color: '#7f8c8d',
    marginRight: 4,
  },
  idNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  statusChip: {
    height: 32,
    paddingHorizontal: 4,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  divider: {
    marginVertical: 12,
  },
  cardContent: {
    flexDirection: 'row',
  },
  thumbnail: {
    width: 90,
    height: 90,
    borderRadius: 8,
  },
  details: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'space-between',
  },
  description: {
    fontSize: 15,
    marginBottom: 8,
    color: '#34495e',
    lineHeight: 20,
  },
  metaContainer: {
    marginTop: 4,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  metaText: {
    fontSize: 12,
    color: '#7f8c8d',
    marginLeft: 4,
  },
  cardActions: {
    justifyContent: 'flex-end',
    paddingTop: 0,
  },
  emptySurface: {
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
    elevation: 2,
    width: '100%',
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#34495e',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubText: {
    fontSize: 14,
    color: '#7f8c8d',
    textAlign: 'center',
    marginBottom: 24,
  },
  emptyButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 20,
    backgroundColor: '#3498db',
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 6,
  },
});

export default ReportsScreen;