import React, { useState, useEffect, useRef } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity, 
  Image,
  FlatList,
  Dimensions,
  RefreshControl,
  StatusBar,
  ActivityIndicator,
  Alert
} from 'react-native';
import { Card, Title, Paragraph, Badge, IconButton, Divider, Button, Avatar, Chip, ProgressBar } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import { Platform } from 'react-native';
import { useAuth } from '../context/AuthContext';
import { updateRepairStatus } from '../utils/auth';
import { 
  getDashboardData, 
  getFilteredReports, 
  getTaskAnalytics, 
  getWeeklyReportStats,
  getReportStatusSummary,
  getNearbyReports,
  getWeatherInfo,
  getNotifications,
  markNotificationAsRead
} from '../utils/dashboardAPI';


const HomeScreen = ({ navigation }) => {
  const { fieldWorker } = useAuth();
  const [notifications, setNotifications] = useState([]);
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showWeather, setShowWeather] = useState(true);
  const [activeTab, setActiveTab] = useState('home');
  const [cityStats, setCityStats] = useState({
    reportsThisWeek: 0,
    repairsCompleted: 0,
    pendingIssues: 0,
    completionRate: 0
  });
  const [weatherData, setWeatherData] = useState(null);
  const [location, setLocation] = useState(null);
  const [locationName, setLocationName] = useState('Loading...');
  const [errorMsg, setErrorMsg] = useState(null);
  const [taskAnalytics, setTaskAnalytics] = useState(null);
  const [weeklyStats, setWeeklyStats] = useState([]);
  const [statusSummary, setStatusSummary] = useState(null);
  const [nearbyReports, setNearbyReports] = useState([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const animationRef = useRef(null);

  useEffect(() => {
    // Start animation when component mounts
    if (animationRef.current) {
      animationRef.current.play();
    }
    
    // Load initial data
    loadDashboardData();
    
    // Get current location
    getCurrentLocation();
  }, [fieldWorker]);

  const getCurrentLocation = async () => {
    try {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        setLocationName(fieldWorker?.region || 'Location unavailable');
        return;
      }

      let location = await Location.getCurrentPositionAsync({});
      setLocation(location);
      
      // Get location name from coordinates
      let geocode = await Location.reverseGeocodeAsync({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude
      });
      
      if (geocode && geocode.length > 0) {
        const { city, region } = geocode[0];
        setLocationName(city || region || fieldWorker?.region || 'Unknown location');
      }
      
      // Fetch weather data using coordinates
      fetchWeatherData(location.coords);
      
      // Get nearby reports
      fetchNearbyReports(location.coords);
    } catch (error) {
      setErrorMsg('Could not fetch location');
      setLocationName(fieldWorker?.region || 'Location unavailable');
    }
  };

  const fetchWeatherData = async (coordinates) => {
    try {
      const data = await getWeatherInfo(coordinates);
      setWeatherData(data);
    } catch (error) {
      console.error('Error fetching weather data:', error);
      // Weather is non-critical, so just log the error
    }
  };

  const fetchNearbyReports = async (coordinates) => {
    try {
      const data = await getNearbyReports(coordinates, 5); // 5km radius
      setNearbyReports(data);
    } catch (error) {
      console.error('Error fetching nearby reports:', error);
      // Non-critical, just log the error
    }
  };

  const loadDashboardData = async () => {
    if (!fieldWorker) {
      console.log('No field worker data available, redirecting to login');
      // Redirect to login if no field worker data
      navigation.reset({
        index: 0,
        routes: [{ name: 'Login' }],
      });
      return;
    }
    
    try {
      setLoading(true);
      console.log('Starting dashboard data load');
      
      // Load dashboard data with enhanced stats
      const dashboardData = await getDashboardData().catch(error => {
        console.log('Dashboard data fetch error:', error);
        // Return empty data structure to prevent app crashes
        return {
          stats: {
            reportsThisWeek: 0,
            repairsCompleted: 0,
            pendingIssues: 0,
            completionRate: 0,
            byDamageType: {},
            byStatus: {},
          },
          recentReports: [],
          urgentReports: []
        };
      });
      
      setReports(dashboardData.recentReports || []);
      setCityStats(dashboardData.stats || {
        reportsThisWeek: 0,
        repairsCompleted: 0,
        pendingIssues: 0,
        completionRate: 0
      });
      
      // Load notifications
      try {
        const notificationsData = await getNotifications();
        setNotifications(notificationsData);
      } catch (notifError) {
        console.error('Failed to load notifications:', notifError);
        // Continue without notifications
      }
      
      // Load task analytics
      loadTaskAnalytics();
      
      // Load weekly stats
      loadWeeklyStats();
      
      // Load status summary
      loadStatusSummary();
      
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      
      // Check if it's an authentication error
      if (error.message && (
          error.message.includes('token') || 
          error.message.includes('auth') || 
          error.message.includes('unauthorized') ||
          error.message.includes('401')
        )) {
        Alert.alert(
          'Session Expired',
          'Your session has expired. Please log in again.',
          [
            { 
              text: 'OK', 
              onPress: () => {
                // Navigate to login
                navigation.reset({
                  index: 0,
                  routes: [{ name: 'Login' }],
                });
              } 
            }
          ]
        );
      } else {
        Alert.alert('Error', 'Failed to load dashboard data. Please try again.');
      }
      
      // Set default empty state
      setReports([]);
      setCityStats({
        reportsThisWeek: 0,
        repairsCompleted: 0,
        pendingIssues: 0,
        completionRate: 0
      });
      setNotifications([]);
    } finally {
      setLoading(false);
    }
  };

  const loadTaskAnalytics = async () => {
    try {
      const data = await getTaskAnalytics();
      setTaskAnalytics(data);
    } catch (error) {
      console.error('Error loading task analytics:', error);
      // Non-critical, continue without analytics
    }
  };

  const loadWeeklyStats = async () => {
    try {
      const data = await getWeeklyReportStats();
      setWeeklyStats(data);
    } catch (error) {
      console.error('Error loading weekly stats:', error);
      // Non-critical, continue without weekly stats
    }
  };

  const loadStatusSummary = async () => {
    try {
      const data = await getReportStatusSummary('week');
      setStatusSummary(data);
    } catch (error) {
      console.error('Error loading status summary:', error);
      // Non-critical, continue without status summary
    }
  };

  const handleMarkAsRead = async (notification) => {
    try {
      const notificationId = notification.id;
      await markNotificationAsRead(notificationId);
      
      // Update the notifications list
      setNotifications(
        notifications.map(n => {
          if (n.id === notificationId) {
            return { ...n, read: true };
          }
          return n;
        })
      );
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  };

  const calculateStats = (reports) => {
    const now = new Date();
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    
    const reportsThisWeek = reports.filter(report => 
      new Date(report.createdAt) >= oneWeekAgo
    ).length;
    
    const repairsCompleted = reports.filter(report => 
      report.repairStatus === 'completed'
    ).length;
    
    const pendingIssues = reports.filter(report => 
      report.repairStatus === 'pending' || report.repairStatus === 'in_progress'
    ).length;
    
    const completionRate = reports.length > 0 ? repairsCompleted / reports.length : 0;
    
    return {
      reportsThisWeek,
      repairsCompleted,
      pendingIssues,
      completionRate
    };
  };

  const formatTimeAgo = (date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 60) {
      return `${diffInMinutes} minutes ago`;
    } else if (diffInMinutes < 1440) {
      const hours = Math.floor(diffInMinutes / 60);
      return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
      const days = Math.floor(diffInMinutes / 1440);
      return `${days} day${days > 1 ? 's' : ''} ago`;
    }
  };

  const onRefresh = React.useCallback(async () => {
    setRefreshing(true);
    try {
      await loadDashboardData();
      await getCurrentLocation(); // Also refresh location and weather
    } catch (error) {
      console.error('Error refreshing data:', error);
    } finally {
      setRefreshing(false);
    }
  }, [fieldWorker]);

  const handleNotificationPress = (notification) => {
    // Mark notification as read when pressed
    handleMarkAsRead(notification);
    
    // Navigate to the report details
    if (notification.reportId) {
      navigation.navigate('ViewReport', { reportId: notification.reportId });
    }
  };

  const handleQuickStatusUpdate = async (reportId, status) => {
    try {
      await updateRepairStatus(reportId, status, `Status updated from home screen`);
      Alert.alert('Success', 'Repair status updated successfully');
      await loadDashboardData(); // Reload data
    } catch (error) {
      Alert.alert('Error', 'Failed to update repair status');
    }
  };

  const quickActions = [
    {
      id: '1',
      title: 'Report Damage',
      icon: 'camera',
      color: ['#1a73e8', '#4285f4', '#5e97f6'],
      screen: 'Camera',
      animation: 'pulse'
    },
    {
      id: '2',
      title: 'My Reports',
      icon: 'clipboard-list',
      color: ['#0d47a1', '#1565c0', '#1976d2'],
      screen: 'Reports',
      animation: 'fadeIn'
    },
    {
      id: '3',
      title: 'Tasks',
      icon: 'check-circle',
      color: ['#2962ff', '#448aff', '#82b1ff'],
      screen: 'TaskManagement',
      animation: 'fadeIn'
    },
    {
      id: '4',
      title: 'Profile',
      icon: 'account',
      color: ['#0277bd', '#0288d1', '#039be5'],
      screen: 'Profile',
      animation: 'fadeIn'
    }
  ];

  const recentReports = [
    {
      id: '1',
      title: 'Pothole on Main St',
      status: 'approved',
      date: '2 days ago',
      image: 'https://via.placeholder.com/100'
    },
    {
      id: '2',
      title: 'Broken Sidewalk',
      status: 'pending',
      date: '4 days ago',
      image: 'https://via.placeholder.com/100'
    }
  ];

  const renderNotificationItem = ({ item }) => (
    <Animatable.View 
      animation="fadeIn" 
      duration={800} 
      delay={parseInt(item.id.split('_')[1] || item.id) * 100}
    >
      <TouchableOpacity onPress={() => handleNotificationPress(item)}>
        <Card style={[styles.notificationCard, !item.read && styles.unreadCard]}>
          <Card.Content style={styles.notificationContent}>
            <View style={styles.notificationHeader}>
              <View style={styles.notificationTitleContainer}>
                <Avatar.Icon 
                  size={36} 
                  icon={item.icon} 
                  style={{
                    backgroundColor: item.type === 'success' ? '#4caf50' : 
                                   item.type === 'info' ? '#2196f3' : '#ff9800'
                  }} 
                />
                <View style={{marginLeft: 10, flex: 1}}>
                  <Title style={styles.notificationTitle}>{item.title}</Title>
                  {!item.read && <Badge style={styles.badge}>New</Badge>}
                </View>
              </View>
              <Text style={styles.notificationTime}>{item.time}</Text>
            </View>
            <Paragraph style={styles.notificationMessage}>{item.message}</Paragraph>
            <View style={styles.notificationActions}>
              <Chip 
                icon={item.read ? "check" : "email-open"} 
                onPress={() => {
                  // Mark as read logic could be added here
                  const updatedNotifications = notifications.map(n => 
                    n.id === item.id ? { ...n, read: true } : n
                  );
                  setNotifications(updatedNotifications);
                }} 
                style={{height: 30}}
              >
                {item.read ? "Read" : "Mark as read"}
              </Chip>
              {item.reportId && (
                <IconButton 
                  icon="eye" 
                  size={20} 
                  onPress={() => navigation.navigate('ViewReport', { reportId: item.reportId })} 
                />
              )}
            </View>
          </Card.Content>
        </Card>
      </TouchableOpacity>
    </Animatable.View>
  );

  const renderRecentReportItem = (item) => (
    <TouchableOpacity 
      style={styles.recentReportCard}
      onPress={() => navigation.navigate('ViewReport', { reportId: item._id })}
    >
      <Image 
        source={{ 
          uri: item.imageUrl || 'https://via.placeholder.com/100?text=No+Image' 
        }} 
        style={styles.recentReportImage} 
      />
      <View style={styles.recentReportContent}>
        <Text style={styles.recentReportTitle}>
          {item.damageType || 'Road Damage'} - {item.location || 'Unknown Location'}
        </Text>
        <View style={styles.recentReportFooter}>
          <Chip 
            style={{
              backgroundColor: item.repairStatus === 'completed' ? '#e8f5e9' : 
                             item.repairStatus === 'in_progress' ? '#fff3e0' : '#e3f2fd',
              height: 24
            }}
            textStyle={{fontSize: 10}}
          >
            {item.repairStatus || 'pending'}
          </Chip>
          <Text style={styles.recentReportDate}>
            {formatTimeAgo(new Date(item.createdAt))}
          </Text>
        </View>
        {item.repairStatus === 'pending' && (
          <View style={styles.quickActions}>
            <TouchableOpacity 
              style={styles.quickActionButton}
              onPress={() => handleQuickStatusUpdate(item._id, 'in_progress')}
            >
              <MaterialCommunityIcons name="play" size={14} color="#fff" />
              <Text style={styles.quickActionText}>Start</Text>
            </TouchableOpacity>
          </View>
        )}
        {item.repairStatus === 'in_progress' && (
          <View style={styles.quickActions}>
            <TouchableOpacity 
              style={[styles.quickActionButton, { backgroundColor: '#4caf50' }]}
              onPress={() => handleQuickStatusUpdate(item._id, 'completed')}
            >
              <MaterialCommunityIcons name="check" size={14} color="#fff" />
              <Text style={styles.quickActionText}>Complete</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor="#003366" />
      <LinearGradient
        colors={['#003366', '#004080', '#0055a4']}
        style={styles.header}
        start={{x: 0, y: 0}}
        end={{x: 1, y: 1}}
      >
        <View style={styles.headerContent}>
          <View>
            <Animatable.Text 
              animation="fadeInDown" 
              duration={800} 
              style={styles.greeting}
            >
              Hello, {fieldWorker?.name?.split(' ')[0] || 'Field Worker'}
            </Animatable.Text>
            <Animatable.Text 
              animation="fadeInDown" 
              duration={800} 
              delay={200}
              style={styles.subGreeting}
            >
              {fieldWorker?.specialization || 'Road Maintenance'} - {fieldWorker?.region || 'Your Region'}
            </Animatable.Text>
          </View>
          <View style={styles.headerActions}>
            <TouchableOpacity 
              style={styles.avatarContainer}
              onPress={() => navigation.navigate('Profile')}
            >
              <Avatar.Text 
                size={40} 
                label={fieldWorker?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'FW'} 
                style={styles.avatar} 
              />
              <View style={styles.onlineIndicator} />
            </TouchableOpacity>
          </View>
        </View>
      </LinearGradient>

      <ScrollView 
        style={styles.content}
        contentContainerStyle={styles.contentContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#003366', '#0055a4']} tintColor="#003366" />
        }
        showsVerticalScrollIndicator={false}
      >
        <Animatable.View animation="fadeInUp" duration={800} delay={300}>
          <Card style={styles.locationCard} elevation={3}>
            <View style={styles.locationCardContent}>
              <View style={styles.locationIconContainer}>
                <MaterialCommunityIcons name="map-marker" size={24} color="#003366" />
              </View>
              <View style={styles.locationTextContainer}>
                <Text style={styles.locationTitle}>{locationName}</Text>
                <Text style={styles.locationDate}>
                  {new Date().toLocaleDateString('en-US', {
                    weekday: 'long', 
                    month: 'long', 
                    day: 'numeric',
                    year: 'numeric'
                  })}
                </Text>
              </View>
              <View style={styles.statusBadge}>
                <Text style={styles.statusText}>Active</Text>
              </View>
            </View>
          </Card>
        </Animatable.View>
        <Animatable.View animation="fadeInUp" duration={800} delay={300}>
          <Card style={styles.weatherCard}>
            <LinearGradient
              colors={['#4facfe', '#00f2fe']}
              style={styles.weatherGradient}
            >
              <View style={styles.weatherHeader}>
                <Text style={styles.weatherCity}>{locationName}</Text>
                <Text style={styles.weatherDate}>Today</Text>
              </View>
              <View style={styles.weatherContent}>
                <View style={styles.weatherMain}>
                  <Text style={styles.weatherTemp}>72°</Text>
                  <Text style={styles.weatherDesc}>Sunny</Text>
                </View>
                <View style={styles.weatherDetails}>
                  <View style={styles.weatherDetailItem}>
                    <IconButton icon="water" color="#fff" size={20} />
                    <Text style={styles.weatherDetailText}>10%</Text>
                  </View>
                  <View style={styles.weatherDetailItem}>
                    <IconButton icon="weather-windy" color="#fff" size={20} />
                    <Text style={styles.weatherDetailText}>8 mph</Text>
                  </View>
                </View>
              </View>
            </LinearGradient>
          </Card>
        </Animatable.View>

        <Animatable.View animation="fadeInUp" duration={800}>
          <Card style={styles.statsCard}>
            <LinearGradient
              colors={['rgba(26, 115, 232, 0.1)', 'rgba(66, 133, 244, 0.05)']}
              style={styles.statsGradient}
            >
              <Card.Content style={styles.statsContent}>
                <View style={styles.statItem}>
                  <Text style={styles.statNumber}>12</Text>
                  <Text style={styles.statLabel}>Reports</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statNumber}>8</Text>
                  <Text style={styles.statLabel}>Approved</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statNumber}>3</Text>
                  <Text style={styles.statLabel}>In Progress</Text>
                </View>
              </Card.Content>
            </LinearGradient>
          </Card>
        </Animatable.View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
        </View>

        <View style={styles.quickActionsContainer}>
          {quickActions.map((action, index) => (
            <Animatable.View 
              key={action.id} 
              animation={action.animation} 
              duration={500} 
              delay={index * 100}
              style={styles.quickActionWrapper}
            >
              <TouchableOpacity 
                style={styles.quickAction}
                onPress={() => navigation.navigate(action.screen)}
              >
                <LinearGradient
                  colors={action.color}
                  style={styles.quickActionGradient}
                  start={{x: 0, y: 0}}
                  end={{x: 1, y: 1}}
                >
                  <IconButton icon={action.icon} color="#fff" size={32} />
                  <Text style={styles.quickActionText}>{action.title}</Text>
                </LinearGradient>
              </TouchableOpacity>
            </Animatable.View>
          ))}
        </View>

        <Animatable.View animation="fadeIn" duration={800} delay={200}>
          <Card style={styles.cityStatsCard}>
            <Card.Content>
              <Title style={styles.cityStatsTitle}>City Statistics</Title>
              <View style={styles.cityStatsRow}>
                <View style={styles.cityStatItem}>
                  <IconButton icon="chart-line" color="#1a73e8" size={24} />
                  <View>
                    <Text style={styles.cityStatValue}>{cityStats.reportsThisWeek}</Text>
                    <Text style={styles.cityStatLabel}>Reports This Week</Text>
                  </View>
                </View>
                <View style={styles.cityStatItem}>
                  <IconButton icon="check-all" color="#4caf50" size={24} />
                  <View>
                    <Text style={styles.cityStatValue}>{cityStats.repairsCompleted}</Text>
                    <Text style={styles.cityStatLabel}>Repairs Completed</Text>
                  </View>
                </View>
              </View>
              <View style={styles.cityStatsRow}>
                <View style={styles.cityStatItem}>
                  <IconButton icon="clock-outline" color="#ff9800" size={24} />
                  <View>
                    <Text style={styles.cityStatValue}>{cityStats.pendingIssues}</Text>
                    <Text style={styles.cityStatLabel}>Pending Issues</Text>
                  </View>
                </View>
                <View style={styles.cityStatItem}>
                  <IconButton icon="percent" color="#9c27b0" size={24} />
                  <View>
                    <Text style={styles.cityStatValue}>{Math.round(cityStats.completionRate * 100)}%</Text>
                    <Text style={styles.cityStatLabel}>Completion Rate</Text>
                  </View>
                </View>
              </View>
              <View style={styles.progressContainer}>
                <View style={styles.progressLabelContainer}>
                  <Text style={styles.progressLabel}>Overall Repair Progress</Text>
                  <Text style={styles.progressValue}>{Math.round(cityStats.completionRate * 100)}%</Text>
                </View>
                <ProgressBar progress={cityStats.completionRate} color="#1a73e8" style={styles.progressBar} />
              </View>
            </Card.Content>
          </Card>
        </Animatable.View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Your Recent Reports</Text>
          <TouchableOpacity onPress={() => navigation.navigate('Reports')}>
            <Text style={styles.seeAllText}>See All</Text>
          </TouchableOpacity>
        </View>

        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false} 
          style={styles.recentReportsContainer}
          contentContainerStyle={styles.recentReportsContentContainer}
        >
          {reports.slice(0, 3).map(item => (
            <Animatable.View key={item._id} animation="fadeInRight" duration={500} delay={100}>
              {renderRecentReportItem(item)}
            </Animatable.View>
          ))}
          {reports.length === 0 && !loading && (
            <View style={styles.noReportsContainer}>
              <MaterialCommunityIcons name="clipboard-list" size={48} color="#ccc" />
              <Text style={styles.noReportsText}>No assignments yet</Text>
            </View>
          )}
          <TouchableOpacity 
            style={styles.newReportCard}
            onPress={() => navigation.navigate('Camera')}
          >
            <LinearGradient
              colors={['#1a73e8', '#4285f4']}
              style={styles.newReportGradient}
            >
              <IconButton icon="plus" color="#fff" size={32} />
              <Text style={styles.newReportText}>New Report</Text>
            </LinearGradient>
          </TouchableOpacity>
        </ScrollView>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Notifications</Text>
          <TouchableOpacity onPress={() => navigation.navigate('Notifications')}>
            <Text style={styles.seeAllText}>See All</Text>
          </TouchableOpacity>
        </View>

        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#1a73e8" />
            <Text style={styles.loadingText}>Loading notifications...</Text>
          </View>
        ) : (
          <FlatList
            data={notifications}
            renderItem={renderNotificationItem}
            keyExtractor={item => item.id}
            scrollEnabled={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <MaterialCommunityIcons name="bell-off-outline" size={48} color="#4285f4" />
                <Text style={styles.emptyText}>No notifications yet</Text>
              </View>
            }
          />
        )}
        
        <View style={styles.spacer} />
      </ScrollView>
    
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7f9fc',
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    borderBottomLeftRadius: 0,  // Remove rounded corners for more professional look
    borderBottomRightRadius: 0,
    elevation: 4,
    shadowColor: 'rgba(0,51,102,0.4)',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    marginTop: Platform.OS === 'ios' ? -50 : 0
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerActions: {
    flexDirection: 'row',
  },
  avatarContainer: {
    position: 'relative',
  },
  avatar: {
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  onlineIndicator: {
    width: 12,
    height: 12,
    backgroundColor: '#4CAF50',
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#003366',
    position: 'absolute',
    bottom: 0,
    right: 0,
  },
  headerButton: {
    position: 'relative',
    padding: 8,
  },
  notificationBadge: {
    position: 'absolute',
    top: 2,
    right: 2,
    backgroundColor: '#ff3b30',
    width: 16,
    height: 16,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1.5,
    borderColor: '#ffffff',
  },
  greeting: {
    fontSize: 24,
    fontWeight: '600',
    color: '#fff',
    letterSpacing: 0.5,
  },
  subGreeting: {
    fontSize: 15,
    color: 'rgba(255, 255, 255, 0.85)',
    marginTop: 4,
    letterSpacing: 0.2,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 120 : 100, // Adjusted for iOS
  },
  locationCard: {
    marginVertical: 12,
    borderRadius: 8,
    elevation: 2,
    overflow: 'hidden',
    borderWidth: 0,
    backgroundColor: '#ffffff',
    shadowColor: 'rgba(0,51,102,0.1)',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    padding: 16,
  },
  locationCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  locationIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,51,102,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 14,
  },
  locationTextContainer: {
    flex: 1,
  },
  locationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#263238',
    marginBottom: 3,
  },
  locationDate: {
    fontSize: 13,
    color: '#78909C',
  },
  statusBadge: {
    backgroundColor: 'rgba(0,153,102,0.15)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    color: '#00755E',
    fontSize: 12,
    fontWeight: '600',
  },
  weatherCard: {
    marginVertical: 12,
    borderRadius: 8,
    elevation: 3,
    overflow: 'hidden',
    borderWidth: 0,
    shadowColor: 'rgba(0,51,102,0.2)',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  weatherGradient: {
    borderRadius: 20,
    padding: 15,
  },
  weatherHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  weatherCity: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  weatherDate: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  weatherContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 10,
  },
  weatherMain: {
    alignItems: 'flex-start',
  },
  weatherTemp: {
    fontSize: 42,
    fontWeight: 'bold',
    color: '#fff',
  },
  weatherDesc: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
  },
  weatherDetails: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  weatherDetailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 15,
    marginTop: 5,
  },
  weatherDetailText: {
    color: '#fff',
    fontSize: 14,
  },
  statsCard: {
    marginVertical: 12,
    borderRadius: 8,
    elevation: 2,
    overflow: 'hidden',
    borderColor: 'rgba(0, 51, 102, 0.08)',
    borderWidth: 1,
    backgroundColor: '#fff',
    shadowColor: 'rgba(0, 51, 102, 0.1)',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  statsGradient: {
    borderRadius: 0,
  },
  statsContent: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 24,
  },
  statItem: {
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: '700',
    color: '#003366',
  },
  statLabel: {
    fontSize: 13,
    color: '#546e7a',
    marginTop: 6,
    fontWeight: '500',
    textAlign: 'center',
  },
  statDivider: {
    width: 1.5,
    backgroundColor: 'rgba(0, 51, 102, 0.08)',
    height: '70%',
    alignSelf: 'center',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 28,
    marginBottom: 16,
    paddingHorizontal: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#003366',
    letterSpacing: 0.3,
  },
  seeAllText: {
    color: '#0055a4',
    fontWeight: '500',
    padding: 8,
  },
  quickActionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickActionWrapper: {
    width: '48%',
    marginBottom: 16,
  },
  quickAction: {
    borderRadius: 8,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: 'rgba(0,51,102,0.2)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  quickActionGradient: {
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
    height: 110,
  },
  quickActions: {
    flexDirection: 'row',
    marginTop: 8,
    gap: 6,
  },
  quickActionButton: {
    backgroundColor: '#2196f3',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  quickActionText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '500',
  },
  cityStatsCard: {
    marginTop: 12,
    marginBottom: 12,
    borderRadius: 8,
    elevation: 2,
    shadowColor: 'rgba(0,51,102,0.1)',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    backgroundColor: '#fff',
    padding: 20, 
  },
  cityStatsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#003366',
    marginBottom: 16,
    letterSpacing: 0.3,
  },
  cityStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
    flexWrap: 'wrap',
  },
  cityStatItem: {
    flexDirection: 'column',
    alignItems: 'flex-start',
    width: '48%',
    marginBottom: 12,
    padding: 12,
    backgroundColor: 'rgba(0,51,102,0.03)',
    borderRadius: 6,
  },
  cityStatValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#003366',
    marginBottom: 4,
  },
  cityStatLabel: {
    fontSize: 12,
    color: '#546e7a',
    letterSpacing: 0.2,
  },
  progressContainer: {
    marginTop: 10,
  },
  progressLabelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  progressLabel: {
    fontSize: 14,
    color: '#0d47a1',
  },
  progressValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1a73e8',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  recentReportsContainer: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  recentReportsContentContainer: {
    paddingRight: 16,
    paddingLeft: 4, // Added left padding for consistency
  },
  recentReportCard: {
    width: 200,
    height: 180,
    marginRight: 15,
    borderRadius: 15,
    overflow: 'hidden',
    backgroundColor: '#fff',
    elevation: 4,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  recentReportImage: {
    width: '100%',
    height: 120,
    resizeMode: 'cover',
  },
  recentReportContent: {
    padding: 10,
  },
  recentReportTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#0d47a1',
    marginBottom: 5,
  },
  recentReportFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  recentReportDate: {
    fontSize: 12,
    color: '#4285f4',
  },
  newReportCard: {
    width: 200,
    height: 180,
    marginRight: 15,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  newReportGradient: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  newReportText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
    marginTop: 10,
  },
  notificationCard: {
    marginBottom: 12,
    borderRadius: 16,
    elevation: 3,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    borderColor: 'rgba(26, 115, 232, 0.1)',
    borderWidth: 1,
    backgroundColor: '#fff',
  },
  unreadCard: {
    borderLeftWidth: 4,
    borderLeftColor: '#1a73e8',
  },
  notificationContent: {
    padding: 16,
  },
  notificationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  notificationTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  notificationTitle: {
    fontSize: 16,
    marginRight: 8,
    color: '#0d47a1',
    fontWeight: 'bold',
    flexShrink: 1,
  },
  badge: {
    backgroundColor: '#1a73e8',
  },
  notificationTime: {
    fontSize: 12,
    color: '#4285f4',
    marginLeft: 5,
  },
  notificationMessage: {
    fontSize: 14,
    color: '#555',
    lineHeight: 20,
  },
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
    height: 150,
  },
  loadingText: {
    color: '#4285f4',
    fontSize: 16,
    marginTop: 10,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
  },
  emptyText: {
    textAlign: 'center',
    color: '#4285f4',
    fontSize: 16,
    marginTop: 10,
  },
  noReportsContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
    width: 200,
  },
  noReportsText: {
    color: '#666',
    fontSize: 14,
    marginTop: 10,
    textAlign: 'center',
  },
  spacer: {
    height: Platform.OS === 'ios' ? 100 : 80, 
  },
});

export default HomeScreen;
