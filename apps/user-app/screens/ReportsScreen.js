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
  StatusBar,
  Platform,
  Dimensions
} from 'react-native';
import { 
  Card, 
  Title, 
  Paragraph, 
  Chip, 
  ActivityIndicator, 
  Button, 
  Surface, 
  Divider, 
  Searchbar, 
  Menu,
  useTheme,
  IconButton,
  FAB
} from 'react-native-paper';
import { useFocusEffect } from '@react-navigation/native';
import { MaterialIcons, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { useAuth } from '../context/AuthContext';
import { useThemeContext } from '../context/ThemeContext';
import { getUserReports, getFilteredUserReports } from '../utils/reportAPI';
import { getReportImageUrlSync } from '../utils/imageUtils';
import { ModernCard, EmptyState, ConsistentHeader } from '../components/ui';
import { API_BASE_URL } from '../config';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: screenWidth } = Dimensions.get('window');

const ReportsScreen = ({ navigation }) => {
  const { fieldWorker } = useAuth();
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [totalReports, setTotalReports] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('createdAt');
  const [sortOrder, setSortOrder] = useState('desc');
  const [filterVisible, setFilterVisible] = useState(false);
  const [statusFilter, setStatusFilter] = useState('');

  // Fetch reports from API
  const fetchReports = async (pageNum = 1, refresh = false) => {
    try {
      // Reset state if this is a refresh
      if (refresh) {
        setPage(1);
        pageNum = 1;
      }
      
      // Show loading indicators
      if (refresh) setRefreshing(true);
      else if (pageNum === 1) setLoading(true);
      else setLoadingMore(true);

      // Build filters object
      const filters = {
        page: pageNum,
        limit: 10,
        sortBy,
        sortOrder
      };

      // Add search query if present
      if (searchQuery.trim()) {
        filters.search = searchQuery.trim();
      }

      // Add status filter if selected
      if (statusFilter) {
        filters.status = statusFilter;
      }

      // Fetch reports with filters
      const response = await getFilteredUserReports(filters);

      const newReports = response.reports || [];
      const pagination = response.pagination || {};

      // Update state with fetched data
      if (refresh || pageNum === 1) {
        setReports(newReports);
        // Cache the reports for ViewReportScreen to use
        try {
          await AsyncStorage.setItem('cachedReports', JSON.stringify(newReports));
        } catch (cacheError) {
          console.log('Failed to cache reports:', cacheError);
        }
      } else {
        const allReports = [...reports, ...newReports];
        setReports(allReports);
        // Cache all reports
        try {
          await AsyncStorage.setItem('cachedReports', JSON.stringify(allReports));
        } catch (cacheError) {
          console.log('Failed to cache reports:', cacheError);
        }
      }

      // Update pagination info
      setHasMore(pagination.hasMore || false);
      setTotalReports(pagination.total || 0);
      setPage(pageNum);
    } catch (error) {
      console.error('Error fetching reports:', error);
      Alert.alert('Error', 'Network error while fetching reports. Please try again.');
    } finally {
      setLoading(false);
      setRefreshing(false);
      setLoadingMore(false);
    }
  };

  // Pull to refresh functionality
  const onRefresh = useCallback(() => {
    fetchReports(1, true);
  }, [sortBy, sortOrder, searchQuery, statusFilter]);

  // Load more reports when reaching the end of the list
  const handleLoadMore = () => {
    if (!loading && !loadingMore && hasMore) {
      fetchReports(page + 1);
    }
  };

  // Handle search query changes
  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  // Apply search after typing stops
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (page === 1) {
        fetchReports(1, true);
      } else {
        setPage(1);
        fetchReports(1);
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, sortBy, sortOrder, statusFilter]);

  // Fetch reports when screen comes into focus or fieldWorker changes
  useFocusEffect(
    useCallback(() => {
      if (fieldWorker) {
        fetchReports(1, true);
      }
      
      // Clean up function
      return () => {
        // Any cleanup needed
      };
    }, [fieldWorker])
  );

  // Change sorting order
  const handleSortChange = (newSortBy) => {
    if (sortBy === newSortBy) {
      // Toggle sort order if clicking the same field
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Default to descending when changing sort field
      setSortBy(newSortBy);
      setSortOrder('desc');
    }
  };

  // Handle status filter
  const handleStatusFilter = (status) => {
    setStatusFilter(status);
    setFilterVisible(false);
  };

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
        return '#FF9800'; // Orange for pending
      case 'in_progress': 
        return '#03A9F4'; // Blue for in progress
      case 'resolved':
        return '#4CAF50'; // Green for resolved
      case 'rejected':
        return '#F44336'; // Red for rejected
      default:
        return '#78909C'; // Gray for unknown status
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
    let reportId = null;
    
    // First try the standard ID fields
    if (report._id) {
      reportId = report._id;
    } else if (report.id) {
      reportId = report.id;
    } else if (report.reportId) {
      reportId = report.reportId;
    }
    
    // Ensure we have a valid ID before navigating
    if (!reportId) {
      console.error('Invalid report object, no ID found:', report);
      Alert.alert('Error', 'Could not view report details. Invalid report data.');
      return;
    }
    
    // Log the ID we're using for easier debugging
    console.log('Navigating to ViewReport with ID:', reportId);
    
    // Navigate with the proper ID
    navigation.navigate('ViewReport', { reportId });
  };

  // Render loading footer when loading more items
  const renderFooter = () => {
    if (!loadingMore) return null;
    return (
      <View style={styles.footerLoader}>
        <ActivityIndicator size="small" color={theme.colors.primary} />
        <Text style={styles.footerText}>Loading more reports...</Text>
      </View>
    );
  };

  // Get appropriate image URL using our utility function
  const getImageUrl = (report) => {
    if (!report) return 'https://via.placeholder.com/300';
    
    try {
      // Try different image types in order of preference
      const imageTypes = ['before', 'main', 'thumbnail', 'default'];
      
      for (const type of imageTypes) {
        const url = getReportImageUrlSync(report, type);
        // If we have a valid URL that's not a placeholder, use it
        if (url && !url.includes('placeholder')) {
          return url;
        }
      }
      
      // Check if there are images in the report object directly
      if (report.images && report.images.length > 0) {
        const firstImage = report.images[0];
        if (typeof firstImage === 'string' && firstImage.startsWith('http')) {
          return firstImage;
        } else if (firstImage.url && firstImage.url.startsWith('http')) {
          return firstImage.url;
        }
      }
      
      // Fallback to basic function
      return getReportImageUrlSync(report);
    } catch (error) {
      console.error('Error getting image URL:', error);
      return 'https://via.placeholder.com/300?text=Error';
    }
  };

  // Loading state
  if (loading && !refreshing) {
    return (
      <View style={[styles.container, styles.centered, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>
          Loading your reports...
        </Text>
      </View>
    );
  }

  // Empty state with search and filters
  if (!loading && reports.length === 0) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <StatusBar barStyle={isDarkMode ? "light-content" : "dark-content"} backgroundColor={theme.colors.background} />
        
        {/* Header */}
        <LinearGradient
          colors={[theme.colors.primary, theme.colors.primaryDark]}
          style={styles.header}
        >
          <View style={styles.headerContent}>
            <Text style={[styles.headerTitle, { color: theme.colors.onPrimary }]}>
              My Reports
            </Text>
            <Text style={[styles.headerSubtitle, { color: theme.colors.onPrimary + 'CC' }]}>
              Track and manage your submitted reports
            </Text>
          </View>
        </LinearGradient>
        
        {/* Search and filter bar */}
        <View style={styles.searchContainer}>
          <Searchbar
            placeholder="Search reports"
            onChangeText={handleSearch}
            value={searchQuery}
            style={[styles.searchBar, { backgroundColor: theme.colors.surface }]}
            icon="magnify"
            clearIcon="close-circle"
            theme={{ colors: { primary: theme.colors.primary } }}
          />
          <Menu
            visible={filterVisible}
            onDismiss={() => setFilterVisible(false)}
            anchor={
              <IconButton
                icon="filter-variant"
                iconColor={theme.colors.primary}
                size={24}
                onPress={() => setFilterVisible(true)}
                style={[styles.filterButton, { backgroundColor: theme.colors.surface }]}
              />
            }
            contentStyle={[styles.menuContent, { backgroundColor: theme.colors.surface }]}
          >
            <Menu.Item onPress={() => handleStatusFilter('')} title="All" leadingIcon="apps" />
            <Menu.Item onPress={() => handleStatusFilter('pending')} title="Pending" leadingIcon="clock-outline" />
            <Menu.Item onPress={() => handleStatusFilter('in_progress')} title="In Progress" leadingIcon="progress-wrench" />
            <Menu.Item onPress={() => handleStatusFilter('completed')} title="Completed" leadingIcon="check-circle" />
            <Menu.Item onPress={() => handleStatusFilter('cancelled')} title="Cancelled" leadingIcon="close-circle" />
          </Menu>
        </View>

        <View style={[styles.centered, { flex: 1, padding: 16 }]}>
          <EmptyState
            icon={searchQuery || statusFilter ? "magnify" : "file-document-outline"}
            title={searchQuery || statusFilter 
              ? "No reports match your filters" 
              : "No Reports Submitted Yet"}
            subtitle={searchQuery || statusFilter
              ? "Try changing your search criteria or filters"
              : "Help improve your community by submitting reports about road issues in your area"}
            actionText="Submit New Report"
            onAction={() => navigation.navigate('Camera')}
          />
        </View>
        
        <FAB
          icon="camera"
          style={[styles.fab, { backgroundColor: theme.colors.primary }]}
          onPress={() => navigation.navigate('Camera')}
        />
      </View>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />
      
      {/* Header - with iOS optimizations */}
      <ConsistentHeader
        title="My Reports"
        subtitle="Track and manage your submitted reports"
        useGradient={true}
        elevated={true}
        blurEffect={Platform.OS === 'ios'}
      />
      
      {/* Search and filter bar */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search reports"
          onChangeText={handleSearch}
          value={searchQuery}
          style={[styles.searchBar, { backgroundColor: theme.colors.surface }]}
          icon="magnify"
          clearIcon="close-circle"
          theme={{ colors: { primary: theme.colors.primary } }}
        />
        <Menu
          visible={filterVisible}
          onDismiss={() => setFilterVisible(false)}
          anchor={
            <IconButton
              icon="filter-variant"
              iconColor={theme.colors.primary}
              size={24}
              onPress={() => setFilterVisible(true)}
              style={[styles.filterButton, { backgroundColor: theme.colors.surface }]}
            />
          }
          contentStyle={[styles.menuContent, { backgroundColor: theme.colors.surface }]}
        >
          <Menu.Item onPress={() => handleStatusFilter('')} title="All" leadingIcon="apps" />
          <Menu.Item onPress={() => handleStatusFilter('pending')} title="Pending" leadingIcon="clock-outline" />
          <Menu.Item onPress={() => handleStatusFilter('in_progress')} title="In Progress" leadingIcon="progress-wrench" />
          <Menu.Item onPress={() => handleStatusFilter('completed')} title="Completed" leadingIcon="check-circle" />
          <Menu.Item onPress={() => handleStatusFilter('cancelled')} title="Cancelled" leadingIcon="close-circle" />
        </Menu>
      </View>
      
      {/* Results count */}
      <View style={styles.resultsContainer}>
        <Text style={[styles.resultsText, { color: theme.colors.textSecondary }]}>
          {totalReports} report{totalReports !== 1 ? 's' : ''} found
        </Text>
      </View>
      
      <FlatList
        data={reports}
        keyExtractor={(item) => (item._id || item.id).toString()}
        refreshControl={
          <RefreshControl 
            refreshing={refreshing} 
            onRefresh={onRefresh}
            colors={[theme.colors.primary]} 
            tintColor={theme.colors.primary}
          />
        }
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.3}
        ListFooterComponent={renderFooter}
        contentContainerStyle={styles.listContent}
        renderItem={({ item, index }) => (
          <Animatable.View 
            animation="fadeInUp" 
            delay={index * 100}
            style={styles.cardWrapper}
          >
            <ModernCard 
              onPress={() => handleReportPress(item)}
              style={styles.modernCard}
              interactive
            >
              {/* Card Top Section with Image and Status */}
              <View style={styles.cardTopSection}>
                <Image 
                  source={{ uri: getImageUrl(item) }} 
                  style={styles.cardImage} 
                  defaultSource={require('../assets/icon.png')}
                />
                <View style={styles.cardOverlay}>
                  <View style={[
                    styles.statusIndicator, 
                    { backgroundColor: getStatusColor(item.repairStatus || item.status) }
                  ]}>
                    <MaterialCommunityIcons 
                      name={getStatusIcon(item.repairStatus || item.status)} 
                      size={14} 
                      color="white" 
                      style={{marginRight: 4}}
                    />
                    <Text style={styles.statusText}>
                      {(item.repairStatus || item.status).replace('_', ' ').toUpperCase()}
                    </Text>
                  </View>
                  <View style={[styles.idIndicator, { backgroundColor: theme.colors.surface + 'E6' }]}>
                    <Text style={[styles.idNumber, { color: theme.colors.text }]}>
                      #{item.reportId || item._id || item.id}
                    </Text>
                  </View>
                </View>
              </View>
              
              <View style={styles.cardContent}>
                <View style={styles.tagRow}>
                  <Chip 
                    mode="flat"
                    style={[styles.tagChip, { backgroundColor: theme.colors.primary + '20' }]}
                    textStyle={[styles.tagChipText, { color: theme.colors.primary }]}
                    icon={() => <MaterialCommunityIcons name="tools" size={14} color={theme.colors.primary} />}
                  >
                    {item.damageType || 'Unknown'}
                  </Chip>
                  <Text style={[styles.dateIndicator, { color: theme.colors.textSecondary }]}>
                    {formatDate(item.createdAt).split(',')[0]}
                  </Text>
                </View>
                
                <Text numberOfLines={2} style={[styles.description, { color: theme.colors.text }]}>
                  {item.description || 'No description provided'}
                </Text>
                
                <View style={styles.locationRow}>
                  <MaterialCommunityIcons name="map-marker-outline" size={16} color={theme.colors.primary} />
                  <Text style={[styles.locationText, { color: theme.colors.textSecondary }]} numberOfLines={1}>
                    {item.location || 'Unknown location'}
                  </Text>
                </View>
                
                <View style={styles.cardFooter}>
                  <View style={styles.priorityIndicator}>
                    {item.priority > 7 ? (
                      <Chip 
                        mode="flat" 
                        compact 
                        style={[styles.priorityChip, { backgroundColor: theme.colors.error + '20' }]}
                        textStyle={{ color: theme.colors.error, fontWeight: '600' }}
                      >
                        High Priority
                      </Chip>
                    ) : item.priority > 4 ? (
                      <Chip 
                        mode="flat" 
                        compact 
                        style={[styles.priorityChip, { backgroundColor: theme.colors.warning + '20' }]}
                        textStyle={{ color: theme.colors.warning, fontWeight: '600' }}
                      >
                        Medium
                      </Chip>
                    ) : (
                      <Chip 
                        mode="flat" 
                        compact 
                        style={[styles.priorityChip, { backgroundColor: theme.colors.success + '20' }]}
                        textStyle={{ color: theme.colors.success, fontWeight: '600' }}
                      >
                        Low
                      </Chip>
                    )}
                  </View>
                  <Button 
                    mode="contained" 
                    compact 
                    onPress={() => handleReportPress(item)}
                    style={[styles.viewButton, { backgroundColor: theme.colors.primary }]}
                    labelStyle={styles.viewButtonLabel}
                    icon="arrow-right"
                    contentStyle={{flexDirection: 'row-reverse'}}
                  >
                    View
                  </Button>
                </View>
              </View>
            </ModernCard>
          </Animatable.View>
        )}
      />
      
      <FAB
        icon="camera"
        style={[styles.fab, { backgroundColor: theme.colors.primary }]}
        onPress={() => navigation.navigate('Camera')}
      />
    </SafeAreaView>
  );
};

// Helper functions
const getStatusColor = (status) => {
  switch (status?.toLowerCase()) {
    case 'completed': return '#10b981';
    case 'in_progress': return '#f59e0b';
    case 'pending': return '#dc2626';
    default: return '#6b7280';
  }
};

const getStatusIcon = (status) => {
  switch (status?.toLowerCase()) {
    case 'completed': return 'check-circle';
    case 'in_progress': return 'progress-clock';
    case 'pending': return 'clock-outline';
    default: return 'help-circle';
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
  header: {
    paddingTop: 50,
    paddingBottom: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  headerContent: {
    paddingHorizontal: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    opacity: 0.9,
  },
  searchContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    alignItems: 'center',
    gap: 8,
  },
  searchBar: {
    flex: 1,
    elevation: 2,
    borderRadius: 12,
  },
  filterButton: {
    borderRadius: 12,
    elevation: 2,
  },
  menuContent: {
    borderRadius: 12,
    elevation: 4,
  },
  resultsContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  resultsText: {
    fontSize: 14,
  },
  listContent: {
    padding: 16,
    paddingBottom: 100,
  },
  cardWrapper: {
    marginBottom: 16,
  },
  modernCard: {
    overflow: 'hidden',
  },
  cardTopSection: {
    position: 'relative',
    height: 160,
    width: '100%',
  },
  cardImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  cardOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.1)',
    justifyContent: 'space-between',
    padding: 12,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  idIndicator: {
    alignSelf: 'flex-end',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  idNumber: {
    fontSize: 12,
    fontWeight: '600',
  },
  cardContent: {
    padding: 16,
  },
  tagRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  tagChip: {
    height: 28,
    borderRadius: 14,
  },
  tagChipText: {
    fontSize: 12,
    fontWeight: '600',
  },
  dateIndicator: {
    fontSize: 12,
    fontWeight: '500',
  },
  description: {
    fontSize: 15,
    marginVertical: 10,
    lineHeight: 20,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 8,
  },
  locationText: {
    fontSize: 13,
    marginLeft: 6,
    flex: 1,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
  },
  priorityIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  priorityChip: {
    height: 24,
    borderRadius: 12,
  },
  viewButton: {
    borderRadius: 20,
    height: 36,
  },
  viewButtonLabel: {
    fontSize: 14,
  },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 20,
    borderRadius: 30,
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  footerLoader: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
  },
  footerText: {
    fontSize: 14,
    marginTop: 8,
  },
});

export default ReportsScreen;