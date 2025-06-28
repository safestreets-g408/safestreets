import React, { useState, useEffect, useCallback } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Image, 
  ScrollView, 
  TouchableOpacity,
  Alert,
  Linking,
  StatusBar,
  Platform,
  ImageBackground
} from 'react-native';
import { Card, Title, Paragraph, Chip, Button, ActivityIndicator, 
  Divider, Surface, Avatar, Badge, IconButton, Menu, Portal, Modal } from 'react-native-paper';
import { MaterialIcons, Ionicons, FontAwesome5, MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { getReportImageUrlSync, preloadImageToken } from '../utils/imageUtils';
import { API_BASE_URL } from '../config';
import { getReportById } from '../utils/reportAPI';
import { useAuth } from '../context/AuthContext';

const ViewReportScreen = ({ route = {}, navigation }) => {
  const { reportId } = route.params || {}; 
  const { fieldWorker } = useAuth();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [imageModalVisible, setImageModalVisible] = useState(false);
  const [menuVisible, setMenuVisible] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);

  // Fetch report details
  const fetchReportDetails = useCallback(async () => {
    if (!reportId) {
      Alert.alert('Error', 'No report ID provided');
      setLoading(false);
      return;
    }

    try {
      console.log('Fetching report details for ID:', reportId);
      
      // Ensure we have a fresh token before making the API call
      await preloadImageToken();
      
      // Fetch report data with better error handling
      const reportData = await getReportById(reportId);
      
      if (!reportData) {
        throw new Error('Failed to fetch report details - empty response');
      }
      
      console.log('Report data fetched successfully:', 
        JSON.stringify({
          id: reportData._id || reportData.id,
          damageType: reportData.damageType,
          status: reportData.status,
          hasImages: Boolean(reportData.images && reportData.images.length)
        })
      );
      
      // Ensure ID consistency
      if (!reportData.id && reportData._id) {
        reportData.id = reportData._id;
      } else if (!reportData._id && reportData.id) {
        reportData._id = reportData.id;
      }
      
      setReport(reportData);
      
      // Generate image URL with better error handling
      let url = null;
      
      // First check if report has direct image URLs
      if (reportData.imageUrl && reportData.imageUrl.startsWith('http')) {
        url = reportData.imageUrl;
        console.log('Using direct imageUrl from report:', url);
      } 
      // Then check if report has images array
      else if (reportData.images && reportData.images.length > 0) {
        const firstImage = reportData.images[0];
        if (typeof firstImage === 'string' && firstImage.startsWith('http')) {
          url = firstImage;
          console.log('Using URL from images array:', url);
        } else if (firstImage.url && firstImage.url.startsWith('http')) {
          url = firstImage.url;
          console.log('Using nested URL from images array:', url);
        }
      }
      
      // If no direct URLs found, try to construct based on ID and type
      if (!url || url.includes('placeholder')) {
        console.log('No direct image URLs found, trying to construct URL with types');
        
        // Try different image types in order of preference
        const imageTypes = ['before', 'main', 'thumbnail', 'default'];
        
        for (const type of imageTypes) {
          const constructedUrl = getReportImageUrlSync(reportData, type);
          console.log(`Trying image type '${type}':`, constructedUrl);
          
          // If we have a valid URL that's not a placeholder, use it
          if (constructedUrl && !constructedUrl.includes('placeholder')) {
            url = constructedUrl;
            console.log('Using constructed URL with type:', type);
            break;
          }
        }
      }
      
      // Set the image URL or use placeholder if nothing worked
      if (!url || url.includes('placeholder')) {
        console.log('No valid image URL found, using placeholder');
        url = 'https://via.placeholder.com/300?text=No+Image+Available';
      }
      
      setImageUrl(url);
      console.log('Final image URL set to:', url);
    } catch (error) {
      console.error('Error fetching report details:', error);
      Alert.alert('Error', 'Failed to load report details. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [reportId]);

  useEffect(() => {
    // Preload image token first, then fetch report details
    const loadData = async () => {
      try {
        await preloadImageToken();
        console.log('Image token preloaded in ViewReportScreen');
        await fetchReportDetails();
      } catch (err) {
        console.error('Error in ViewReportScreen setup:', err);
        fetchReportDetails(); // Still try to fetch report details even if token loading fails
      }
    };
    
    loadData();
  }, [fetchReportDetails]);

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
        <Surface style={styles.loadingCard}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.loadingText}>Loading report details...</Text>
          <Text style={styles.loadingSubText}>Please wait while we fetch the report information</Text>
        </Surface>
      </View>
    );
  }

  if (!report) {
    return (
      <View style={styles.errorContainer}>
        <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
        <Surface style={styles.errorCard}>
          <View style={styles.errorIconContainer}>
            <Ionicons name="alert-circle-outline" size={60} color="#e74c3c" />
          </View>
          <Text style={styles.errorText}>Report Not Found</Text>
          <Text style={styles.errorSubText}>We couldn't find the report you're looking for.</Text>
          <Button 
            mode="contained" 
            onPress={() => navigation.goBack()}
            style={styles.backButton}
            labelStyle={styles.backButtonLabel}
            icon="arrow-left"
            uppercase={false}
          >
            Back to Reports
          </Button>
        </Surface>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
      
      {/* Modern Header with Action Buttons */}
      <Surface style={styles.headerSurface}>
        <View style={styles.headerContent}>
          <View style={styles.backButtonContainer}>
            <IconButton
              icon="arrow-left"
              size={24}
              color="#0f172a"
              style={{ backgroundColor: 'rgba(0,0,0,0.03)', borderRadius: 12 }}
              onPress={() => navigation.goBack()}
            />
          </View>
          <Title style={styles.title}>Report Details</Title>
          <Menu
            visible={menuVisible}
            onDismiss={() => setMenuVisible(false)}
            anchor={
              <IconButton
                icon="dots-vertical"
                size={24}
                color="#0f172a"
                style={{ backgroundColor: 'rgba(0,0,0,0.03)', borderRadius: 12 }}
                onPress={() => setMenuVisible(true)}
              />
            }
            contentStyle={styles.menuContent}
          >
            <Menu.Item 
              onPress={() => {
                setMenuVisible(false);
                Alert.alert('Share Report', 'Sharing functionality would be implemented here');
              }} 
              title="Share Report" 
              icon="share-variant"
            />
            <Menu.Item 
              onPress={() => {
                setMenuVisible(false);
                Alert.alert('Download Report', 'Download functionality would be implemented here');
              }} 
              title="Download PDF" 
              icon="file-pdf-box"
            />
            <Menu.Item 
              onPress={() => {
                setMenuVisible(false);
                Alert.alert('Report Issue', 'Report issue functionality would be implemented here');
              }} 
              title="Report an Issue" 
              icon="flag"
            />
          </Menu>
        </View>
      </Surface>

      <ScrollView style={styles.scrollContent}>
        {/* Image Hero Section with Overlay */}
        <Card style={styles.imageCard} elevation={4}>
          <ImageBackground 
            source={{ 
              uri: imageUrl || 'https://via.placeholder.com/600?text=Report+Image',
              headers: {
                'Cache-Control': 'no-store, no-cache',
                'Pragma': 'no-cache'
              }
            }} 
            style={styles.imageBackground}
            imageStyle={{ opacity: 0.95 }}
            resizeMode="cover"
            defaultSource={require('../assets/icon.png')}
            onError={(e) => {
              console.error('Image loading error:', e.nativeEvent?.error);
              // Force reload on error
              if (imageUrl && !imageUrl.includes('placeholder')) {
                setTimeout(() => {
                  setImageUrl(imageUrl + '&reload=' + Date.now());
                }, 1000);
              }
              console.log('Image failed to load:', e.nativeEvent.error);
              // If image fails, try to generate a new URL with a different param
              if (report) {
                const newUrl = getReportImageUrlSync(report, 'main');
                if (newUrl !== imageUrl) {
                  setImageUrl(newUrl);
                }
              }
            }}
          >
            <LinearGradient
              colors={['transparent', 'rgba(0,0,0,0.8)']}
              style={styles.imageGradient}
            >
              <View style={styles.imageOverlayContent}>
                <View style={styles.reportIdContainer}>
                  <Text style={styles.reportIdLabel}>REPORT ID</Text>
                  <Text style={styles.reportIdValue}>#{report.id || report._id || 'Unknown'}</Text>
                </View>
                
                <TouchableOpacity 
                  style={styles.viewImageButton}
                  onPress={() => setImageModalVisible(true)}
                >
                  <MaterialIcons name="fullscreen" size={20} color="white" />
                </TouchableOpacity>
              </View>
            </LinearGradient>
          </ImageBackground>
        </Card>
        
        {/* Status and Date Info Bar */}
        <Surface style={styles.statusSurface}>
          <View style={styles.statusContainer}>
            <View style={[styles.statusIndicator, { backgroundColor: getStatusColor(report.status) }]} />
            <View>
              <Text style={styles.statusLabel}>STATUS</Text>
              <Text style={styles.statusValue}>{report.status.replace('_', ' ').toUpperCase()}</Text>
            </View>
          </View>
          <View style={styles.dateContainer}>
            <Text style={styles.dateLabel}>REPORTED</Text>
            <Text style={styles.dateValue}>{formatDate(report.createdAt)}</Text>
          </View>
        </Surface>
        
        {/* Description Card */}
        <Card style={styles.detailsCard} elevation={3}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <MaterialCommunityIcons name="text-box-outline" size={22} color="#3498db" />
              <Title style={styles.sectionTitle}>Description</Title>
            </View>
            <Paragraph style={styles.description}>
              {report.description || 'No detailed description available for this report.'}
            </Paragraph>
          </Card.Content>
        </Card>
        
        {/* Location Card */}
        <Card style={styles.detailsCard} elevation={3}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <MaterialCommunityIcons name="map-marker-radius" size={22} color="#3498db" />
              <Title style={styles.sectionTitle}>Location</Title>
            </View>
            <Paragraph style={styles.locationText}>
              {report.address || report.location || 'Location information not available'}
            </Paragraph>
            
            {/* Map Preview Placeholder */}
            <View style={styles.mapPreviewContainer}>
              <View style={styles.mapPlaceholder}>
                <MaterialCommunityIcons name="map-marker-radius" size={40} color="#3498db" />
                <Text style={styles.mapPlaceholderText}>Map Preview</Text>
              </View>
              <LinearGradient
                colors={['transparent', 'rgba(0,0,0,0.6)']}
                style={styles.mapPreviewGradient}
              >
                <TouchableOpacity style={styles.mapButton} onPress={openMap}>
                  <MaterialIcons name="directions" size={18} color="white" />
                  <Text style={styles.mapButtonText}>Get Directions</Text>
                </TouchableOpacity>
              </LinearGradient>
            </View>
          </Card.Content>
        </Card>
        
        {/* Activity Timeline */}
        {report.comments && report.comments.length > 0 && (
          <Card style={styles.commentsCard} elevation={3}>
            <Card.Content>
              <View style={styles.sectionHeader}>
                <MaterialCommunityIcons name="timeline-clock-outline" size={22} color="#3498db" />
                <Title style={styles.sectionTitle}>Activity Timeline</Title>
              </View>
              
              <View style={styles.timelineContainer}>
                {report.comments.map((comment, index) => (
                  <View key={comment.id} style={styles.timelineItem}>
                    <View style={styles.timelineLine} />
                    <View style={[styles.timelineDot, { backgroundColor: index === 0 ? '#3498db' : '#bdc3c7' }]}>
                      <MaterialCommunityIcons 
                        name={index === 0 ? "comment-text-outline" : "clock-outline"} 
                        size={16} 
                        color="white" 
                      />
                    </View>
                    <View style={styles.timelineContent}>
                      <Surface style={styles.timelineCard}>
                        <Text style={styles.commentText}>{comment.text}</Text>
                        <View style={styles.commentFooter}>
                          <MaterialCommunityIcons name="clock-outline" size={14} color="#7f8c8d" />
                          <Text style={styles.commentDate}>{formatDate(comment.timestamp)}</Text>
                        </View>
                      </Surface>
                    </View>
                  </View>
                ))}
              </View>
            </Card.Content>
          </Card>
        )}
        
        <View style={styles.buttonContainer}>
          <Button 
            mode="contained" 
            onPress={() => navigation.goBack()}
            style={styles.button}
            labelStyle={styles.buttonLabel}
            icon="arrow-left"
            uppercase={false}
          >
            Back to Reports
          </Button>
        </View>
      </ScrollView>
      
      {/* Image Fullscreen Modal */}
      <Portal>
        <Modal
          visible={imageModalVisible}
          onDismiss={() => setImageModalVisible(false)}
          contentContainerStyle={styles.imageModal}
        >
          <IconButton
            icon="close"
            size={24}
            color="white"
            style={styles.closeModalButton}
            onPress={() => setImageModalVisible(false)}
          />
          <Image
            source={{ 
              uri: imageUrl || 'https://via.placeholder.com/300?text=No+Image+Available',
              headers: {
                'Cache-Control': 'no-store, no-cache',
                'Pragma': 'no-cache'
              }
            }}
            style={styles.fullScreenImage}
            resizeMode="contain"
            defaultSource={require('../assets/icon.png')}
            onError={(e) => {
              console.log('Modal image failed to load:', e.nativeEvent.error);
              Alert.alert('Image Error', 'Could not load the full image. The image may be unavailable.');
            }}
          />
          <Text style={styles.imageHintText}>Report ID: {report && (report.id || report._id || 'Unknown')}</Text>
        </Modal>
      </Portal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9fb',
  },
  scrollContent: {
    flex: 1,
    backgroundColor: '#f9f9fb',
    paddingBottom: 8,
  },
  // Loading State Styles
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f9f9fb',
  },
  loadingCard: {
    padding: 28,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 3,
    width: '85%',
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.03)',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
    letterSpacing: 0.2,
  },
  loadingSubText: {
    marginTop: 10,
    fontSize: 14,
    textAlign: 'center',
    color: '#64748b',
    paddingHorizontal: 20,
    lineHeight: 20,
  },
  // Error State Styles
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#f9f9fb',
  },
  errorCard: {
    padding: 28,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 3,
    width: '88%',
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.03)',
  },
  errorIconContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  errorText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#ef4444',
    marginBottom: 10,
    letterSpacing: 0.2,
  },
  errorSubText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#64748b',
    marginBottom: 26,
    paddingHorizontal: 10,
    lineHeight: 22,
  },
  // Header Styles
  headerSurface: {
    backgroundColor: '#fff',
    paddingTop: Platform.OS === 'ios' ? 50 : 10,
    paddingBottom: 14,
    paddingHorizontal: 16,
    elevation: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    zIndex: 100,
    borderBottomColor: 'rgba(0,0,0,0.05)',
    borderBottomWidth: 1,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 6,
  },
  backButtonContainer: {
    marginRight: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    flex: 1,
    textAlign: 'center',
    color: '#1a1a2e',
    letterSpacing: 0.2,
  },
  menuContent: {
    borderRadius: 8,
    marginTop: 40,
  },
  // Image Styles
  imageCard: {
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    backgroundColor: 'transparent',
  },
  imageBackground: {
    height: 280,
    width: '100%',
    justifyContent: 'flex-end',
  },
  imageGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: '60%',
    justifyContent: 'flex-end',
    padding: 20,
  },
  imageOverlayContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
  },
  reportIdContainer: {
    padding: 8,
  },
  reportIdLabel: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 11,
    fontWeight: 'bold',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    marginBottom: 2,
  },
  reportIdValue: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  viewImageButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.35)',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  // Status Styles
  statusSurface: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginHorizontal: 16,
    marginTop: 12,
    marginBottom: 16,
    padding: 18,
    borderRadius: 16,
    backgroundColor: 'white',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusIndicator: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.15,
    shadowRadius: 3,
  },
  statusLabel: {
    fontSize: 10,
    color: '#94a3b8',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 3,
  },
  statusValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#334155',
  },
  dateContainer: {
    alignItems: 'flex-end',
  },
  dateLabel: {
    fontSize: 10,
    color: '#94a3b8',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 3,
  },
  dateValue: {
    fontSize: 15,
    color: '#334155',
    fontWeight: '500',
  },
  // Card Styles
  detailsCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 16,
    backgroundColor: 'white',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.02)',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: '700',
    marginLeft: 10,
    color: '#1e293b',
    letterSpacing: 0.2,
  },
  description: {
    fontSize: 15,
    lineHeight: 24,
    color: '#475569',
    paddingHorizontal: 4,
    letterSpacing: 0.1,
  },
  locationText: {
    fontSize: 15,
    color: '#475569',
    marginBottom: 16,
    paddingHorizontal: 4,
    letterSpacing: 0.1,
  },
  // Map Styles
  mapPreviewContainer: {
    height: 180,
    borderRadius: 12,
    overflow: 'hidden',
    marginTop: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.04)',
  },
  mapPreviewImage: {
    width: '100%',
    height: '100%',
  },
  mapPlaceholder: {
    width: '100%',
    height: '100%',
    backgroundColor: '#f1f5f9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  mapPlaceholderText: {
    color: '#94a3b8',
    fontSize: 15,
    marginTop: 8,
    fontWeight: '500',
  },
  mapPreviewGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: '50%',
    justifyContent: 'flex-end',
    alignItems: 'center',
    padding: 14,
  },
  mapButton: {
    flexDirection: 'row',
    backgroundColor: '#0284c7',
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 25,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 3,
    elevation: 2,
  },
  mapButtonText: {
    color: 'white',
    fontWeight: '600',
    marginLeft: 6,
    fontSize: 14,
  },
  // Comments / Timeline Styles
  commentsCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 16,
    backgroundColor: 'white',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.02)',
  },
  timelineContainer: {
    marginTop: 12,
    paddingLeft: 12,
  },
  timelineItem: {
    flexDirection: 'row',
    marginBottom: 24,
    position: 'relative',
  },
  timelineLine: {
    position: 'absolute',
    left: 16,
    top: 26,
    bottom: -14,
    width: 2,
    backgroundColor: '#e2e8f0',
  },
  timelineDot: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#0ea5e9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
    zIndex: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
    borderWidth: 2,
    borderColor: 'white',
  },
  timelineContent: {
    flex: 1,
    paddingBottom: 6,
  },
  timelineCard: {
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#f8fafc',
    elevation: 1,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  commentText: {
    fontSize: 15,
    color: '#334155',
    lineHeight: 22,
    letterSpacing: 0.1,
  },
  commentFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
  },
  commentDate: {
    fontSize: 12,
    color: '#64748b',
    marginLeft: 6,
    fontWeight: '500',
  },
  // Button Styles
  buttonContainer: {
    marginHorizontal: 16,
    marginTop: 8,
    marginBottom: 28,
  },
  button: {
    paddingVertical: 8,
    backgroundColor: '#0284c7',
    borderRadius: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  buttonLabel: {
    fontSize: 15,
    fontWeight: '600',
    padding: 4,
    letterSpacing: 0.2,
  },
  backButton: {
    backgroundColor: '#0284c7',
    borderRadius: 12,
    paddingVertical: 8,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  backButtonLabel: {
    fontSize: 15,
    letterSpacing: 0.5,
    fontWeight: '600',
  },
  // Modal Styles
  imageModal: {
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    flex: 1,
    margin: 0,
    padding: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeModalButton: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 50 : 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 24,
    zIndex: 10,
    padding: 4,
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  fullScreenImage: {
    width: '100%',
    height: '85%',
  },
  imageHintText: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 13,
    position: 'absolute',
    bottom: 30,
    alignSelf: 'center',
    padding: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
    letterSpacing: 0.5,
    overflow: 'hidden',
  }
});

export default ViewReportScreen;
