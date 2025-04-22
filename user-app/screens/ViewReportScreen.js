import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Image, 
  ScrollView, 
  TouchableOpacity,
  Alert,
  Linking,
  StatusBar
} from 'react-native';
import { Card, Title, Paragraph, Chip, Button, ActivityIndicator, Divider, Surface, Avatar } from 'react-native-paper';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';

const ViewReportScreen = ({ route, navigation }) => {
  const { reportId } = route.params || { reportId: 1 }; 
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  // Mock data for UI demonstration
  const mockReport = {
    id: reportId,
    status: 'pending',
    imageUrl: 'https://via.placeholder.com/300',
    description: 'Pothole on Main Street that needs immediate attention',
    createdAt: new Date().toISOString(),
    latitude: 37.7749,
    longitude: -122.4194,
    address: '123 Main St, San Francisco, CA',
    updatedAt: new Date().toISOString(),
    comments: [
      { id: 1, text: 'Scheduled for repair next week', timestamp: new Date(Date.now() - 86400000).toISOString() },
      { id: 2, text: 'Materials ordered', timestamp: new Date(Date.now() - 43200000).toISOString() }
    ]
  };

  // Fetch report details
  useEffect(() => {
    const fetchReportDetails = async () => {
      try {
        // Simulate API call delay
        setTimeout(() => {
          setReport(mockReport);
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error('Error fetching report details:', error);
        Alert.alert('Error', 'Failed to load report details');
        setLoading(false);
      }
    };

    fetchReportDetails();
  }, [reportId]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return '#f39c12'; // Orange
      case 'in_progress':
        return '#3498db'; // Blue
      case 'resolved':
        return '#2ecc71'; // Green
      default:
        return '#95a5a6'; // Gray
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return 'clock-outline';
      case 'in_progress': 
        return 'progress-wrench';
      case 'resolved':
        return 'check-circle';
      default:
        return 'help-circle';
    }
  };

  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  const openMap = () => {
    if (report && report.latitude && report.longitude) {
      const url = `https://maps.google.com/?q=${report.latitude},${report.longitude}`;
      Linking.openURL(url).catch(err => {
        Alert.alert('Error', 'Could not open maps application');
      });
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading report details...</Text>
      </View>
    );
  }

  if (!report) {
    return (
      <View style={styles.errorContainer}>
        <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
        <Ionicons name="alert-circle-outline" size={60} color="#e74c3c" />
        <Text style={styles.errorText}>Report not found</Text>
        <Button 
          mode="contained" 
          onPress={() => navigation.goBack()}
          style={styles.backButton}
          icon="arrow-left"
        >
          Go Back
        </Button>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
      
      <Surface style={styles.headerSurface}>
        <View style={styles.headerContent}>
          <Title style={styles.title}>Report #{report.id}</Title>
          <Chip 
            style={[styles.statusChip, { backgroundColor: getStatusColor(report.status) }]}
            textStyle={styles.statusText}
            icon={() => <Ionicons name={getStatusIcon(report.status)} size={16} color="white" />}
          >
            {report.status.replace('_', ' ').toUpperCase()}
          </Chip>
        </View>
        <Text style={styles.dateText}>
          <Ionicons name="calendar-outline" size={14} color="#7f8c8d" /> Reported on {formatDate(report.createdAt)}
        </Text>
      </Surface>
      
      <Card style={styles.imageCard} elevation={3}>
        <Card.Cover 
          source={{ uri: report.imageUrl }} 
          style={styles.image} 
          resizeMode="cover"
        />
        <Card.Actions style={styles.imageActions}>
          <Button 
            icon="image-filter" 
            mode="text" 
            onPress={() => Alert.alert('View Full Image', 'This would open the full-size image')}>
            View Full Size
          </Button>
        </Card.Actions>
      </Card>
      
      <Card style={styles.detailsCard} elevation={2}>
        <Card.Content>
          <View style={styles.sectionHeader}>
            <Ionicons name="document-text-outline" size={20} color="#3498db" />
            <Title style={styles.sectionTitle}>Description</Title>
          </View>
          <Paragraph style={styles.description}>{report.description}</Paragraph>
          
          <Divider style={styles.divider} />
          
          <View style={styles.sectionHeader}>
            <Ionicons name="location-outline" size={20} color="#3498db" />
            <Title style={styles.sectionTitle}>Location</Title>
          </View>
          <Paragraph style={styles.locationText}>{report.address}</Paragraph>
          <Button 
            mode="outlined" 
            icon="map-marker" 
            onPress={openMap} 
            style={styles.mapButton}
          >
            View on Map
          </Button>
        </Card.Content>
      </Card>
      
      {report.comments && report.comments.length > 0 && (
        <Card style={styles.commentsCard} elevation={2}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Ionicons name="chatbubble-ellipses-outline" size={20} color="#3498db" />
              <Title style={styles.sectionTitle}>Updates</Title>
            </View>
            
            {report.comments.map(comment => (
              <Surface key={comment.id} style={styles.commentContainer}>
                <Text style={styles.commentText}>{comment.text}</Text>
                <View style={styles.commentFooter}>
                  <Ionicons name="time-outline" size={12} color="#7f8c8d" />
                  <Text style={styles.commentDate}>{formatDate(comment.timestamp)}</Text>
                </View>
              </Surface>
            ))}
          </Card.Content>
        </Card>
      )}
      
      <View style={styles.buttonContainer}>
        <Button 
          mode="contained" 
          onPress={() => navigation.goBack()}
          style={styles.button}
          icon="arrow-left"
          contentStyle={styles.buttonContent}
        >
          Back to Reports
        </Button>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#7f8c8d',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  errorText: {
    fontSize: 18,
    color: '#e74c3c',
    marginBottom: 20,
    marginTop: 10,
  },
  headerSurface: {
    margin: 16,
    padding: 16,
    borderRadius: 8,
    elevation: 4,
    backgroundColor: 'white',
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    flex: 1,
  },
  statusChip: {
    height: 32,
  },
  statusText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  dateText: {
    color: '#7f8c8d',
    fontSize: 14,
  },
  imageCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 8,
    overflow: 'hidden',
  },
  image: {
    height: 220,
  },
  imageActions: {
    justifyContent: 'flex-end',
  },
  detailsCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 8,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  description: {
    fontSize: 16,
    lineHeight: 24,
    color: '#2c3e50',
  },
  divider: {
    marginVertical: 16,
    height: 1,
    backgroundColor: '#ecf0f1',
  },
  locationText: {
    fontSize: 16,
    color: '#2c3e50',
    marginBottom: 10,
  },
  mapButton: {
    marginTop: 5,
    borderColor: '#3498db',
  },
  commentsCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 8,
  },
  commentContainer: {
    padding: 12,
    borderRadius: 8,
    marginVertical: 6,
    backgroundColor: '#f8f9fa',
    elevation: 1,
  },
  commentText: {
    fontSize: 15,
    color: '#2c3e50',
  },
  commentFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  commentDate: {
    fontSize: 12,
    color: '#7f8c8d',
    marginLeft: 4,
  },
  buttonContainer: {
    margin: 16,
    marginTop: 8,
  },
  button: {
    paddingVertical: 8,
    backgroundColor: '#3498db',
    borderRadius: 8,
    elevation: 2,
  },
  buttonContent: {
    height: 48,
  },
  backButton: {
    backgroundColor: '#3498db',
    paddingHorizontal: 24,
    paddingVertical: 8,
  }
});

export default ViewReportScreen;
