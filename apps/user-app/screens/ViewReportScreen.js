import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  StatusBar,
  Dimensions,
  Alert,
  Share,
  RefreshControl,
  Platform
} from 'react-native';
import {
  useTheme,
  Button,
  Chip,
  IconButton,
  ActivityIndicator,
  Divider,
  Snackbar,
  Surface
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ModernCard, ConsistentHeader } from '../components/ui';
import { getReportImageUrlSync } from '../utils/imageUtils';
import { getReportDetails, getReportById } from '../utils/reportAPI';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: screenWidth } = Dimensions.get('window');

const ViewReportScreen = ({ route, navigation }) => {
  const theme = useTheme();
  const { reportId } = route.params || {};
  
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  useEffect(() => {
    if (reportId) {
      loadReport();
    } else {
      console.error('No reportId provided to ViewReportScreen');
      setSnackbarMessage('No report ID provided');
      setSnackbarVisible(true);
      setLoading(false);
    }
  }, [reportId]);

  const loadReport = async () => {
    try {
      setLoading(true);
      console.log('Loading report with ID:', reportId);
      
      // Try the field worker endpoint first (for assigned reports)
      try {
        const response = await getReportDetails(reportId);
        console.log('Report API response (field worker):', response);
        
        if (response) {
          setReport(response);
          console.log('Report data loaded successfully from field worker endpoint:', response);
          return;
        }
      } catch (fieldWorkerError) {
        console.log('Field worker endpoint failed, trying alternative endpoint:', fieldWorkerError.message);
        
        // Try the alternative getReportById endpoint
        try {
          const response = await getReportById(reportId);
          console.log('Report API response (alternative):', response);
          
          if (response) {
            setReport(response);
            console.log('Report data loaded successfully from alternative endpoint:', response);
            return;
          }
        } catch (alternativeError) {
          console.log('Alternative endpoint also failed:', alternativeError.message);
          
          // If both API endpoints fail, try to get from cache
          try {
            // Get the current list of reports and find this one
            const reports = JSON.parse(await AsyncStorage.getItem('cachedReports') || '[]');
            const cachedReport = reports.find(r => 
              (r._id === reportId) || (r.id === reportId) || (r.reportId === reportId)
            );
            
            if (cachedReport) {
              setReport(cachedReport);
              console.log('Using cached report data:', cachedReport);
              return;
            }
          } catch (cacheError) {
            console.log('Cache lookup failed:', cacheError.message);
          }
          
          // If no cached data, throw the most relevant error
          throw fieldWorkerError;
        }
      }
      
      throw new Error('No report data received from any source');
    } catch (error) {
      console.error('Error loading report:', error);
      setSnackbarMessage(`Failed to load report: ${error.message}`);
      setSnackbarVisible(true);
      
      // Show error state
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    loadReport().finally(() => setRefreshing(false));
  };

  const handleShare = async () => {
    try {
      const title = report.title || `${report.damageType || 'Damage'} Report`;
      const location = report.location || 'Unknown location';
      const status = report.repairStatus || report.status || 'pending';
      
      await Share.share({
        message: `Report: ${title}\nLocation: ${location}\nStatus: ${status}`,
        title: 'Damage Report'
      });
    } catch (error) {
      console.error('Error sharing report:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed': 
      case 'resolved':
        return theme.colors.success;
      case 'in_progress': 
      case 'in progress':
        return theme.colors.warning;
      case 'pending': 
        return theme.colors.error;
      case 'on_hold':
      case 'on hold':
        return '#FF9800'; // Orange
      case 'cancelled':
        return '#9E9E9E'; // Gray
      default: 
        return theme.colors.secondary;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': 
      case 'critical':
        return theme.colors.error;
      case 'medium': 
      case 'normal':
        return theme.colors.warning;
      case 'low': 
        return theme.colors.success;
      default: 
        return theme.colors.secondary;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>
          Loading report...
        </Text>
      </View>
    );
  }

  if (!report) {
    return (
      <View style={[styles.container, styles.centered]}>
        <MaterialCommunityIcons 
          name="file-document-outline" 
          size={64} 
          color={theme.colors.outline} 
        />
        <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>
          Report not found
        </Text>
        <Button 
          mode="contained" 
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          Go Back
        </Button>
      </View>
    );
  }

  return (
    <SafeAreaView 
      style={[styles.container, { backgroundColor: theme.colors.background }]} 
      edges={['right', 'left']} // Don't include top edge as ConsistentHeader handles that
    >
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Report Title and Status */}
        <Animatable.View animation="fadeInUp" delay={100}>
          <ModernCard style={styles.titleCard}>
            <View style={styles.titleSection}>
              <Text style={[styles.reportTitle, { color: theme.colors.text }]}>
                {report.title || `${report.damageType || 'Damage'} Report`}
              </Text>
              <View style={styles.statusRow}>
                <Chip
                  mode="flat"
                  style={[styles.statusChip, { backgroundColor: getStatusColor(report.repairStatus || report.status) + '20' }]}
                  textStyle={{ color: getStatusColor(report.repairStatus || report.status), fontWeight: '600' }}
                >
                  {(report.repairStatus || report.status || 'pending').replace('_', ' ').toUpperCase()}
                </Chip>
                <Chip
                  mode="flat"
                  style={[styles.priorityChip, { backgroundColor: getPriorityColor(report.priority) + '20' }]}
                  textStyle={{ color: getPriorityColor(report.priority), fontWeight: '600' }}
                >
                  {report.priority || 'Medium'} Priority
                </Chip>
              </View>
            </View>
          </ModernCard>
        </Animatable.View>

        {/* Images */}
        {(report.images?.length > 0 || report._id || report.id) && (
          <Animatable.View animation="fadeInUp" delay={200}>
            <ModernCard style={styles.imagesCard}>
              <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
                Images
              </Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.imagesScroll}>
                {report.images && report.images.length > 0 ? (
                  // If report has images array, display them
                  report.images.map((image, index) => {
                    let imageUrl;
                    if (typeof image === 'string' && image.startsWith('http')) {
                      imageUrl = image;
                    } else if (image?.url && image.url.startsWith('http')) {
                      imageUrl = image.url;
                    } else {
                      // Use utility function to construct URL
                      imageUrl = getReportImageUrlSync(report, index === 0 ? 'main' : 'thumbnail');
                    }
                    
                    return (
                      <Surface key={index} style={styles.imageContainer} elevation={2}>
                        <Image
                          source={{ uri: imageUrl }}
                          style={styles.reportImage}
                          resizeMode="cover"
                          onError={(e) => console.log('Image load error:', e.nativeEvent.error, 'URL:', imageUrl)}
                          onLoad={() => console.log('Image loaded successfully:', imageUrl)}
                        />
                        <View style={styles.imageOverlay}>
                          <Text style={styles.imageLabel}>Image {index + 1}</Text>
                        </View>
                      </Surface>
                    );
                  })
                ) : (
                  // If no images array but we have a report ID, try to get default images
                  ['main', 'before', 'after'].map((type, index) => {
                    const imageUrl = getReportImageUrlSync(report, type);
                    return (
                      <Surface key={index} style={styles.imageContainer} elevation={2}>
                        <Image
                          source={{ uri: imageUrl }}
                          style={styles.reportImage}
                          resizeMode="cover"
                          onError={(e) => console.log(`${type} image load error:`, e.nativeEvent.error)}
                          onLoad={() => console.log(`${type} image loaded successfully`)}
                        />
                        <View style={styles.imageOverlay}>
                          <Text style={styles.imageLabel}>{type.charAt(0).toUpperCase() + type.slice(1)}</Text>
                        </View>
                      </Surface>
                    );
                  })
                )}
              </ScrollView>
            </ModernCard>
          </Animatable.View>
        )}

        {/* Description */}
        <Animatable.View animation="fadeInUp" delay={300}>
          <ModernCard style={styles.descriptionCard}>
            <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
              Description
            </Text>
            <Text style={[styles.description, { color: theme.colors.textSecondary }]}>
              {report.description || 'No description provided'}
            </Text>
          </ModernCard>
        </Animatable.View>

        {/* Details */}
        <Animatable.View animation="fadeInUp" delay={400}>
          <ModernCard style={styles.detailsCard}>
            <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
              Details
            </Text>
            
            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="map-marker" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Location</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.location || 'Unknown location'}
                </Text>
              </View>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="account" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Reported By</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.reporter || report.reportedBy || 'Unknown'}
                </Text>
              </View>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="hammer-wrench" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Damage Type</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.damageType || report.category || 'Unknown'}
                </Text>
              </View>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="alert-circle" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Severity</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.severity || 'Not specified'}
                </Text>
              </View>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="flag" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Priority</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.priority || 'Not set'}
                </Text>
              </View>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="calendar" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Reported At</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {formatDate(report.createdAt || report.reportedAt)}
                </Text>
              </View>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="clipboard-list" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Action Required</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.action || 'To be determined'}
                </Text>
              </View>
            </View>

            {/* Show assignment info if assigned */}
            {report.assignedTo && (
              <>
                <Divider style={styles.divider} />
                <View style={styles.detailRow}>
                  <MaterialCommunityIcons name="account-hard-hat" size={20} color={theme.colors.primary} />
                  <View style={styles.detailContent}>
                    <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Assigned To</Text>
                    <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                      {typeof report.assignedTo === 'object' 
                        ? report.assignedTo.name || report.assignedTo.username 
                        : report.assignedTo}
                    </Text>
                  </View>
                </View>
              </>
            )}

            {/* Show resolution date if resolved */}
            {report.resolvedAt && (
              <>
                <Divider style={styles.divider} />
                <View style={styles.detailRow}>
                  <MaterialCommunityIcons name="check-circle" size={20} color={theme.colors.primary} />
                  <View style={styles.detailContent}>
                    <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Resolved At</Text>
                    <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                      {formatDate(report.resolvedAt)}
                    </Text>
                  </View>
                </View>
              </>
            )}

            {/* Show report ID */}
            <Divider style={styles.divider} />
            <View style={styles.detailRow}>
              <MaterialCommunityIcons name="identifier" size={20} color={theme.colors.primary} />
              <View style={styles.detailContent}>
                <Text style={[styles.detailLabel, { color: theme.colors.textSecondary }]}>Report ID</Text>
                <Text style={[styles.detailValue, { color: theme.colors.text }]}>
                  {report.reportId || report._id || report.id}
                </Text>
              </View>
            </View>
          </ModernCard>
        </Animatable.View>

        {/* Notes */}
        {report.notes && report.notes.length > 0 && (
          <Animatable.View animation="fadeInUp" delay={500}>
            <ModernCard style={styles.notesCard}>
              <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
                Progress Notes
              </Text>
              {report.notes.map((note, index) => (
                <View key={note.id}>
                  <View style={styles.noteItem}>
                    <View style={styles.noteHeader}>
                      <Text style={[styles.noteAuthor, { color: theme.colors.primary }]}>
                        {note.author}
                      </Text>
                      <Text style={[styles.noteTimestamp, { color: theme.colors.textSecondary }]}>
                        {formatDate(note.timestamp)}
                      </Text>
                    </View>
                    <Text style={[styles.noteContent, { color: theme.colors.text }]}>
                      {note.content}
                    </Text>
                  </View>
                  {index < report.notes.length - 1 && <Divider style={styles.noteDivider} />}
                </View>
              ))}
            </ModernCard>
          </Animatable.View>
        )}

        {/* Action Buttons */}
        <Animatable.View animation="fadeInUp" delay={600}>
          <View style={styles.actionButtons}>
            <Button
              mode="contained"
              icon="pencil"
              onPress={() => {
                // TODO: Navigate to edit screen
                setSnackbarMessage('Edit functionality coming soon');
                setSnackbarVisible(true);
              }}
              style={[styles.actionButton, { backgroundColor: theme.colors.primary }]}
              contentStyle={styles.buttonContent}
            >
              Edit Report
            </Button>
            <Button
              mode="outlined"
              icon="map"
              onPress={() => {
                // TODO: Navigate to map view
                setSnackbarMessage('Map view coming soon');
                setSnackbarVisible(true);
              }}
              style={[styles.actionButton, { borderColor: theme.colors.primary }]}
              contentStyle={styles.buttonContent}
              textColor={theme.colors.primary}
            >
              View on Map
            </Button>
          </View>
        </Animatable.View>

        <View style={styles.bottomSpacing} />
      </ScrollView>

      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => setSnackbarVisible(false)}
        duration={3000}
        style={{ backgroundColor: theme.colors.surface }}
      >
        <Text style={{ color: theme.colors.text }}>{snackbarMessage}</Text>
      </Snackbar>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 18,
    textAlign: 'center',
  },
  backButton: {
    marginTop: 24,
  },
  header: {
    paddingTop: 50,
    paddingBottom: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 16,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  titleCard: {
    marginBottom: 16,
  },
  titleSection: {
    padding: 20,
  },
  reportTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 12,
    lineHeight: 32,
  },
  statusRow: {
    flexDirection: 'row',
    gap: 8,
  },
  statusChip: {
    borderRadius: 16,
  },
  priorityChip: {
    borderRadius: 16,
  },
  imagesCard: {
    marginBottom: 16,
  },
  imagesScroll: {
    paddingHorizontal: 4,
  },
  imageContainer: {
    marginRight: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  reportImage: {
    width: 200,
    height: 150,
    borderRadius: 12,
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 6,
  },
  imageLabel: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  descriptionCard: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  description: {
    fontSize: 16,
    lineHeight: 24,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  detailsCard: {
    marginBottom: 16,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  detailContent: {
    flex: 1,
    marginLeft: 12,
  },
  detailLabel: {
    fontSize: 14,
    marginBottom: 4,
  },
  detailValue: {
    fontSize: 16,
    fontWeight: '500',
  },
  divider: {
    marginHorizontal: 20,
  },
  notesCard: {
    marginBottom: 16,
  },
  noteItem: {
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  noteHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  noteAuthor: {
    fontSize: 14,
    fontWeight: '600',
  },
  noteTimestamp: {
    fontSize: 12,
  },
  noteContent: {
    fontSize: 14,
    lineHeight: 20,
  },
  noteDivider: {
    marginHorizontal: 20,
  },
  actionButtons: {
    gap: 12,
    marginTop: 8,
  },
  actionButton: {
    borderRadius: 12,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  bottomSpacing: {
    height: 32,
  },
});

export default ViewReportScreen;
