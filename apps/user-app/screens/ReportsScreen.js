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
import { Card, Title, Paragraph, Chip, ActivityIndicator, Button, Surface, Divider, Searchbar, Menu } from 'react-native-paper';
import { useFocusEffect } from '@react-navigation/native';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { getUserReports, getFilteredUserReports } from '../utils/reportAPI';

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
    navigation.navigate('ViewReport', { reportId: report._id || report.id });
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

  // Get appropriate image URL
  const getImageUrl = (report) => {
    if (!report) return 'https://via.placeholder.com/300';
    
    // Return direct URL if available
    if (report.imageUrl) return report.imageUrl;
    
    // For API-based reports
    if (report.images && report.images.length > 0) {
      return `${API_BASE_URL}/damage/report/${report._id}/image/thumbnail`;
    }
    
    return 'https://via.placeholder.com/300';
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
        
        {/* Search and filter bar */}
        <View style={styles.searchContainer}>
          <Searchbar
            placeholder="Search reports"
            onChangeText={handleSearch}
            value={searchQuery}
            style={styles.searchBar}
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
          >
            <Menu.Item onPress={() => handleStatusFilter('')} title="All" />
            <Menu.Item onPress={() => handleStatusFilter('pending')} title="Pending" />
            <Menu.Item onPress={() => handleStatusFilter('in_progress')} title="In Progress" />
            <Menu.Item onPress={() => handleStatusFilter('completed')} title="Completed" />
            <Menu.Item onPress={() => handleStatusFilter('cancelled')} title="Cancelled" />
          </Menu>
        </View>

        <View style={styles.centerContainer}>
          <Surface style={styles.emptySurface} elevation={2}>
            <Ionicons name="document-text-outline" size={70} color="#78909C" />
            <Text style={styles.emptyText}>
              {searchQuery || statusFilter 
                ? "No reports match your filters" 
                : "No Reports Submitted"}
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
              buttonColor="#003366"
            >
              Submit New Report
            </Button>
          </Surface>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#003366" barStyle="light-content" />
      
      {/* Search and filter bar */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search reports"
          onChangeText={handleSearch}
          value={searchQuery}
          style={styles.searchBar}
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
        >
          <Menu.Item onPress={() => handleStatusFilter('')} title="All" />
          <Menu.Item onPress={() => handleStatusFilter('pending')} title="Pending" />
          <Menu.Item onPress={() => handleStatusFilter('in_progress')} title="In Progress" />
          <Menu.Item onPress={() => handleStatusFilter('completed')} title="Completed" />
          <Menu.Item onPress={() => handleStatusFilter('cancelled')} title="Cancelled" />
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
          >
            <Card style={styles.card} elevation={2}>
              <Card.Content>
                <View style={styles.cardHeader}>
                  <View style={styles.idContainer}>
                    <Text style={styles.idLabel}>REPORT</Text>
                    <Text style={styles.idNumber}>
                      #{item.reportId || item._id || item.id}
                    </Text>
                  </View>
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
                </View>
                
                <Divider style={styles.divider} />
                
                <View style={styles.cardContent}>
                  <Image 
                    source={{ uri: getImageUrl(item) }} 
                    style={styles.thumbnail} 
                    defaultSource={require('../assets/icon.png')}
                  />
                  <View style={styles.details}>
                    <View style={styles.tagContainer}>
                      <Chip 
                        style={styles.tagChip}
                        textStyle={styles.tagChipText}
                      >
                        {item.damageType || 'Unknown'}
                      </Chip>
                    </View>
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
                          {item.location || 'Unknown location'}
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
    backgroundColor: '#f7f9fc',
  },
  listContent: {
    padding: 16,
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
  card: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(0, 51, 102, 0.08)',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  idContainer: {
    flexDirection: 'column',
    alignItems: 'flex-start',
  },
  idLabel: {
    fontSize: 12,
    color: '#78909C',
    marginBottom: 2,
    letterSpacing: 0.5,
  },
  idNumber: {
    fontSize: 16,
    fontWeight: '600',
    color: '#263238',
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 4,
  },
  statusText: {
    color: 'white',
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.5,
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