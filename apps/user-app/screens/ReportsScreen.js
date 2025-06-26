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
  Platform
} from 'react-native';
import { Card, Title, Paragraph, Chip, ActivityIndicator, Button, Surface, Divider, Searchbar, Menu } from 'react-native-paper';
import { useFocusEffect } from '@react-navigation/native';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { getUserReports, getFilteredUserReports } from '../utils/reportAPI';
import { getReportImageUrlSync } from '../utils/imageUtils';
import { API_BASE_URL } from '../config';

const ReportsScreen = ({ navigation }) => {
  const { fieldWorker } = useAuth();
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
      } else {
        setReports(prev => [...prev, ...newReports]);
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
        <ActivityIndicator size="small" color="#003366" />
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
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#003366" />
        <Text style={styles.loadingText}>Loading your reports...</Text>
      </View>
    );
  }

  // Empty state with search and filters
  if (!loading && reports.length === 0) {
    return (
      <View style={styles.container}>
        <StatusBar backgroundColor="#003366" barStyle="light-content" />
        
        {/* Header */}
        <View style={styles.headerContainer}>
          <Text style={styles.headerTitle}>My Reports</Text>
          <Text style={styles.headerSubtitle}>
            Track and manage your submitted reports
          </Text>
        </View>
        
        {/* Search and filter bar */}
        <View style={styles.searchContainer}>
          <Searchbar
            placeholder="Search reports"
            onChangeText={handleSearch}
            value={searchQuery}
            style={styles.searchBar}
            icon="magnify"
            clearIcon="close-circle"
            theme={{ colors: { primary: '#003366' } }}
          />
          <Menu
            visible={filterVisible}
            onDismiss={() => setFilterVisible(false)}
            anchor={
              <TouchableOpacity 
                style={styles.filterButton} 
                onPress={() => setFilterVisible(true)}
              >
                <Ionicons name="filter" size={24} color="#003366" />
                {statusFilter ? <View style={styles.filterIndicator} /> : null}
              </TouchableOpacity>
            }
            contentStyle={styles.menuContent}
          >
            <Menu.Item onPress={() => handleStatusFilter('')} title="All" leadingIcon="apps" />
            <Menu.Item onPress={() => handleStatusFilter('pending')} title="Pending" leadingIcon="clock-outline" />
            <Menu.Item onPress={() => handleStatusFilter('in_progress')} title="In Progress" leadingIcon="progress-wrench" />
            <Menu.Item onPress={() => handleStatusFilter('completed')} title="Completed" leadingIcon="check-circle" />
            <Menu.Item onPress={() => handleStatusFilter('cancelled')} title="Cancelled" leadingIcon="close-circle" />
          </Menu>
        </View>

        <View style={styles.centerContainer}>
          <Surface style={styles.emptySurface} elevation={4}>
            {searchQuery || statusFilter ? (
              <Ionicons name="search" size={80} color="#78909C" />
            ) : (
              <Image
                source={require('../assets/icon.png')}
                style={styles.emptyStateImage}
                resizeMode="contain"
              />
            )}
            <Text style={styles.emptyText}>
              {searchQuery || statusFilter 
                ? "No reports match your filters" 
                : "No Reports Submitted Yet"}
            </Text>
            <Text style={styles.emptySubText}>
              {searchQuery || statusFilter
                ? "Try changing your search criteria or filters"
                : "Help improve your community by submitting reports about road issues in your area"}
            </Text>
            <Button 
              mode="contained"
              icon="camera"
              style={styles.emptyButton}
              onPress={() => navigation.navigate('Camera')}
              contentStyle={styles.emptyButtonContent}
              labelStyle={styles.emptyButtonLabel}
            >
              Submit New Report
            </Button>
          </Surface>
        </View>
        
        <TouchableOpacity 
          style={styles.fab}
          onPress={() => navigation.navigate('Camera')}
        >
          <Ionicons name="camera" size={24} color="white" />
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#003366" barStyle="light-content" />
      
      {/* Header and Search bar */}
      <View style={styles.headerContainer}>
        <Text style={styles.headerTitle}>My Reports</Text>
        <Text style={styles.headerSubtitle}>
          Track and manage your submitted reports
        </Text>
      </View>
      
      {/* Search and filter bar */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search reports"
          onChangeText={handleSearch}
          value={searchQuery}
          style={styles.searchBar}
          icon="magnify"
          clearIcon="close-circle"
          theme={{ colors: { primary: '#003366' } }}
        />
        <Menu
          visible={filterVisible}
          onDismiss={() => setFilterVisible(false)}
          anchor={
            <TouchableOpacity 
              style={styles.filterButton} 
              onPress={() => setFilterVisible(true)}
            >
              <Ionicons name="filter" size={24} color="#003366" />
              {statusFilter ? <View style={styles.filterIndicator} /> : null}
            </TouchableOpacity>
          }
          contentStyle={styles.menuContent}
        >
          <Menu.Item onPress={() => handleStatusFilter('')} title="All" leadingIcon="apps" />
          <Menu.Item onPress={() => handleStatusFilter('pending')} title="Pending" leadingIcon="clock-outline" />
          <Menu.Item onPress={() => handleStatusFilter('in_progress')} title="In Progress" leadingIcon="progress-wrench" />
          <Menu.Item onPress={() => handleStatusFilter('completed')} title="Completed" leadingIcon="check-circle" />
          <Menu.Item onPress={() => handleStatusFilter('cancelled')} title="Cancelled" leadingIcon="close-circle" />
        </Menu>
      </View>
      
      {/* Sort options */}
      <View style={styles.sortContainer}>
        <Text style={styles.sortLabel}>Sort by:</Text>
        <TouchableOpacity 
          style={[
            styles.sortOption, 
            sortBy === 'createdAt' && styles.sortOptionActive
          ]}
          onPress={() => handleSortChange('createdAt')}
        >
          <Text style={[
            styles.sortText, 
            sortBy === 'createdAt' && styles.sortTextActive
          ]}>Date</Text>
          {sortBy === 'createdAt' && (
            <Ionicons 
              name={sortOrder === 'asc' ? 'arrow-up' : 'arrow-down'} 
              size={14} 
              color={sortBy === 'createdAt' ? '#003366' : '#78909C'} 
            />
          )}
        </TouchableOpacity>
        <TouchableOpacity 
          style={[
            styles.sortOption, 
            sortBy === 'priority' && styles.sortOptionActive
          ]}
          onPress={() => handleSortChange('priority')}
        >
          <Text style={[
            styles.sortText, 
            sortBy === 'priority' && styles.sortTextActive
          ]}>Priority</Text>
          {sortBy === 'priority' && (
            <Ionicons 
              name={sortOrder === 'asc' ? 'arrow-up' : 'arrow-down'} 
              size={14} 
              color={sortBy === 'priority' ? '#003366' : '#78909C'} 
            />
          )}
        </TouchableOpacity>
        <TouchableOpacity 
          style={[
            styles.sortOption, 
            sortBy === 'repairStatus' && styles.sortOptionActive
          ]}
          onPress={() => handleSortChange('repairStatus')}
        >
          <Text style={[
            styles.sortText, 
            sortBy === 'repairStatus' && styles.sortTextActive
          ]}>Status</Text>
          {sortBy === 'repairStatus' && (
            <Ionicons 
              name={sortOrder === 'asc' ? 'arrow-up' : 'arrow-down'} 
              size={14} 
              color={sortBy === 'repairStatus' ? '#003366' : '#78909C'} 
            />
          )}
        </TouchableOpacity>
      </View>
      
      {/* Results count */}
      <View style={styles.resultsContainer}>
        <Text style={styles.resultsText}>
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
            colors={['#003366', '#0055a4']} 
            tintColor="#003366"
          />
        }
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.3}
        ListFooterComponent={renderFooter}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => (
          <TouchableOpacity 
            onPress={() => handleReportPress(item)}
            activeOpacity={0.7}
            style={styles.cardWrapper}
          >
            <Card style={styles.card} elevation={3}>
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
                    <Ionicons 
                      name={getStatusIcon(item.repairStatus || item.status)} 
                      size={14} 
                      color="white" 
                      style={{marginRight: 4}}
                    />
                    <Text style={styles.statusText}>
                      {(item.repairStatus || item.status).replace('_', ' ').toUpperCase()}
                    </Text>
                  </View>
                  <View style={styles.idIndicator}>
                    <Text style={styles.idNumber}>
                      #{item.reportId || item._id || item.id}
                    </Text>
                  </View>
                </View>
              </View>
              
              <Card.Content style={styles.cardContentPadded}>
                <View style={styles.tagRow}>
                  <Chip 
                    style={styles.tagChip}
                    textStyle={styles.tagChipText}
                    icon={() => <Ionicons name="construct-outline" size={14} color="#003366" />}
                  >
                    {item.damageType || 'Unknown'}
                  </Chip>
                  <Text style={styles.dateIndicator}>{formatDate(item.createdAt).split(',')[0]}</Text>
                </View>
                
                <Paragraph numberOfLines={2} style={styles.description}>
                  {item.description || 'No description provided'}
                </Paragraph>
                
                <View style={styles.locationRow}>
                  <Ionicons name="location-outline" size={16} color="#2980b9" />
                  <Text style={styles.locationText} numberOfLines={1}>
                    {item.location || 'Unknown location'}
                  </Text>
                </View>
                
                <View style={styles.cardFooter}>
                  <View style={styles.priorityIndicator}>
                    {item.priority > 7 ? (
                      <Chip compact style={styles.highPriorityChip}>High Priority</Chip>
                    ) : item.priority > 4 ? (
                      <Chip compact style={styles.mediumPriorityChip}>Medium Priority</Chip>
                    ) : (
                      <Chip compact style={styles.lowPriorityChip}>Low Priority</Chip>
                    )}
                  </View>
                  <Button 
                    mode="contained" 
                    compact 
                    onPress={() => handleReportPress(item)}
                    style={styles.viewButton}
                    labelStyle={styles.viewButtonLabel}
                    icon="arrow-right"
                    contentStyle={{flexDirection: 'row-reverse'}}
                  >
                    View
                  </Button>
                </View>
              </Card.Content>
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
    backgroundColor: '#f7f9fc',
  },
  headerContainer: {
    backgroundColor: '#003366',
    paddingTop: 60,
    paddingBottom: 16,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    elevation: 4,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 5,
    shadowOpacity: 0.3,
    marginBottom: 8,
    marginTop: Platform.OS === 'ios' ? -5 : 0,
  },
  headerTitle: {
    color: 'white',
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 4,
  },
  headerSubtitle: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 14,
    fontWeight: '400',
  },
  searchContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 8,
    alignItems: 'center',
  },
  searchBar: {
    flex: 1,
    elevation: 2,
    backgroundColor: 'white',
    borderRadius: 10,
    height: 44,
  },
  filterButton: {
    width: 44,
    height: 44,
    marginLeft: 8,
    borderRadius: 10,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 2,
  },
  filterIndicator: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#e74c3c',
    top: 10,
    right: 10,
  },
  menuContent: {
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: 'white',
    elevation: 4,
  },
  sortContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    alignItems: 'center',
    backgroundColor: 'white',
    marginHorizontal: 16,
    marginVertical: 8,
    borderRadius: 10,
    elevation: 2,
  },
  sortLabel: {
    color: '#34495e',
    fontSize: 14,
    fontWeight: '500',
    marginRight: 12,
  },
  sortOption: {
    flexDirection: 'row',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    alignItems: 'center',
    backgroundColor: '#f1f2f6',
  },
  sortOptionActive: {
    backgroundColor: 'rgba(0, 51, 102, 0.1)',
  },
  sortText: {
    fontSize: 13,
    color: '#78909C',
    marginRight: 4,
  },
  sortTextActive: {
    color: '#003366',
    fontWeight: '600',
  },
  resultsContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  resultsText: {
    fontSize: 14,
    color: '#78909C',
    fontStyle: 'italic',
  },
  listContent: {
    padding: 12,
    paddingBottom: 80, // Extra padding for FAB
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f7f9fc',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#003366',
    fontWeight: '500',
    letterSpacing: 0.3,
  },
  // Card styling
  cardWrapper: {
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  card: {
    borderRadius: 12,
    overflow: 'hidden',
    elevation: 3,
    backgroundColor: 'white',
    margin: 0,
  },
  cardTopSection: {
    position: 'relative',
    height: 160,
    width: '100%',
  },
  cardImage: {
    width: '100%',
    height: '100%',
  },
  cardOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 12,
  },
  cardContentPadded: {
    padding: 16,
    paddingTop: 12,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    elevation: 2,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  idIndicator: {
    backgroundColor: 'rgba(255,255,255,0.85)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    elevation: 2,
  },
  idNumber: {
    fontSize: 14,
    fontWeight: '700',
    color: '#003366',
  },
  tagRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  tagChip: {
    backgroundColor: 'rgba(0, 51, 102, 0.1)',
    height: 28,
  },
  tagChipText: {
    fontSize: 12,
    color: '#003366',
    fontWeight: '600',
  },
  dateIndicator: {
    fontSize: 12,
    color: '#78909C',
    fontWeight: '500',
  },
  description: {
    fontSize: 15,
    marginVertical: 10,
    color: '#34495e',
    lineHeight: 20,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 8,
  },
  locationText: {
    fontSize: 13,
    color: '#2980b9',
    marginLeft: 6,
    flex: 1,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 10,
  },
  priorityIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  highPriorityChip: {
    backgroundColor: 'rgba(231, 76, 60, 0.15)',
    height: 24,
  },
  mediumPriorityChip: {
    backgroundColor: 'rgba(243, 156, 18, 0.15)',
    height: 24,
  },
  lowPriorityChip: {
    backgroundColor: 'rgba(46, 204, 113, 0.15)',
    height: 24,
  },
  viewButton: {
    backgroundColor: '#003366',
    borderRadius: 20,
    height: 36,
  },
  viewButtonLabel: {
    fontSize: 14,
    marginLeft: 0,
  },
  // Empty state styling
  emptySurface: {
    padding: 32,
    borderRadius: 16,
    alignItems: 'center',
    elevation: 4,
    width: '100%',
    backgroundColor: 'white',
  },
  emptyStateImage: {
    width: 120,
    height: 120,
    marginBottom: 20,
    opacity: 0.8,
  },
  emptyText: {
    fontSize: 20,
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
    lineHeight: 20,
  },
  emptyButton: {
    backgroundColor: '#003366',
    paddingHorizontal: 16,
    borderRadius: 30,
    elevation: 2,
  },
  emptyButtonContent: {
    height: 48,
  },
  emptyButtonLabel: {
    fontSize: 16,
    letterSpacing: 0.5,
  },
  // FAB styling
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 20,
    backgroundColor: '#003366',
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
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
    color: '#78909C',
    marginTop: 8,
  },
});

export default ReportsScreen;