import React, { useState, useEffect, useRef } from 'react';
import { 
  View, 
  StyleSheet, 
  ScrollView, 
  RefreshControl,
  StatusBar,
  Alert,
  Platform
} from 'react-native';
import { useTheme } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Location from 'expo-location';
import { useAuth } from '../context/AuthContext';
import { updateRepairStatus } from '../utils/auth';
import { 
  getDashboardData, 
  getFilteredReports, 
  getTaskAnalytics, 
  getWeeklyReportStats,
  getReportStatusSummary,
  getNotifications,
  markNotificationAsRead,
  getWeatherInfo
} from '../utils/dashboardAPI';
import { getReportImageUrlSync, preloadImageToken } from '../utils/imageUtils';
import { API_BASE_URL } from '../config';

// Import home components
import { 
  HeaderComponent,
  LocationInfoComponent,
  WeatherComponent,
  QuickActionsComponent,
  StatsComponent,
  RecentReportsComponent,
  NotificationsComponent
} from '../components/home';

// Demo weather data
const DEMO_WEATHER = {
  temperature: 72,
  condition: 'Sunny',
  humidity: 10,
  windSpeed: 8,
  icon: 'weather-sunny'
};

const quickActions = [
  {
    id: '1',
    title: 'Report Damage',
    icon: 'camera',
    gradientColors: ['#1a73e8', '#4285f4', '#5e97f6'],
    color: { primary: '#1a73e8' },
    screen: 'Camera',
    animation: 'pulse'
  },
  {
    id: '2',
    title: 'My Reports',
    icon: 'clipboard-list',
    gradientColors: ['#0d47a1', '#1565c0', '#1976d2'],
    color: { primary: '#0d47a1' },
    screen: 'Reports',
    animation: 'fadeIn'
  },
  {
    id: '3',
    title: 'Tasks',
    icon: 'check-circle',
    gradientColors: ['#2962ff', '#448aff', '#82b1ff'],
    color: { primary: '#2962ff' },
    screen: 'TaskManagement',
    animation: 'fadeIn'
  },
  {
    id: '4',
    title: 'Profile',
    icon: 'account',
    gradientColors: ['#0277bd', '#0288d1', '#039be5'],
    color: { primary: '#0277bd' },
    screen: 'Profile',
    animation: 'fadeIn'
  }
];

const HomeScreen = ({ navigation }) => {
  const { fieldWorker, logout } = useAuth();
  const theme = useTheme();
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
  const [authErrorShown, setAuthErrorShown] = useState(false);  // Track if auth error is shown

  useEffect(() => {
    // Start animation when component mounts
    if (animationRef.current) {
      animationRef.current.play();
    }
    
    // Preload image token for faster image loading - make this a priority
    const setupTokens = async () => {
      try {
        await preloadImageToken();
        console.log('Image tokens preloaded in HomeScreen');
      } catch (err) {
        console.error('Failed to preload image token in HomeScreen:', err);
      }
    };
    setupTokens();
    
    // Load initial data
    loadDashboardData();
    
    // Get current location for weather display
    getCurrentLocation();
    
    // Get field worker's reports
    fetchFieldWorkerReports();
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
      
      // Get field worker's reports instead of nearby reports
      fetchFieldWorkerReports();
    } catch (error) {
      setErrorMsg('Could not fetch location');
      setLocationName(fieldWorker?.region || 'Location unavailable');
    }
  };

  const fetchWeatherData = async (coordinates) => {
    try {
      console.log('Fetching weather with coordinates:', coordinates);
      const weatherResponse = await getWeatherInfo(coordinates);
      
      if (weatherResponse) {
        // Transform OpenWeatherMap data to our format
        const transformedWeather = {
          temperature: Math.round(weatherResponse.main?.temp || 72),
          condition: weatherResponse.weather?.[0]?.main || 'Sunny',
          humidity: weatherResponse.main?.humidity || 10,
          windSpeed: Math.round(weatherResponse.wind?.speed || 8),
          icon: mapWeatherIconToName(weatherResponse.weather?.[0]?.icon || '01d')
        };
        console.log('Weather data transformed:', transformedWeather);
        setWeatherData(transformedWeather);
      } else {
        // Fallback to demo data if weather response is empty
        console.log('Falling back to demo weather data');
        setWeatherData(DEMO_WEATHER);
      }
    } catch (error) {
      console.error('Error fetching weather data:', error);
      // Weather is non-critical, so fall back to demo data
      setWeatherData(DEMO_WEATHER);
    }
  };
  
  // Map OpenWeatherMap icon codes to MaterialCommunityIcons names
  const mapWeatherIconToName = (iconCode) => {
    const iconMap = {
      '01d': 'weather-sunny',
      '01n': 'weather-night',
      '02d': 'weather-partly-cloudy',
      '02n': 'weather-night-partly-cloudy',
      '03d': 'weather-cloudy',
      '03n': 'weather-cloudy',
      '04d': 'weather-cloudy',
      '04n': 'weather-cloudy',
      '09d': 'weather-pouring',
      '09n': 'weather-pouring',
      '10d': 'weather-rainy',
      '10n': 'weather-rainy',
      '11d': 'weather-lightning',
      '11n': 'weather-lightning',
      '13d': 'weather-snowy',
      '13n': 'weather-snowy',
      '50d': 'weather-fog',
      '50n': 'weather-fog'
    };
    
    return iconMap[iconCode] || 'weather-sunny';
  };

  const fetchFieldWorkerReports = async () => {
    try {
      if (!fieldWorker || !fieldWorker._id) {
        console.log('No field worker ID available');
        return;
      }
      
      // Instead of location-based, we'll use fieldWorker's reports
      const data = await getFilteredReports({ fieldWorkerId: fieldWorker._id, limit: 5 });
      
      // Extract reports array from the response
      const reports = data?.reports || [];
      setNearbyReports(reports); // Still using the same state variable for compatibility

    } catch (error) {
      console.error('Error fetching field worker reports:', error);
      // Check if it's an auth error
      handleAuthError(error);
      // Non-critical, just log the error if not auth related
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
      
      // Load dashboard data with enhanced stats
      let dashboardData;
      try {
        dashboardData = await getDashboardData();
      } catch (error) {
        console.log('Dashboard data fetch error:', error);
        
        // Use common auth error handler
        if (handleAuthError(error)) {
          setLoading(false);
          return; // Exit function early if auth error
        }
        
        // If not an auth error, use empty data structure
        dashboardData = {
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
      }
      
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
        // Merge backend notifications with generated notifications from reports
        const generatedNotifications = generateNotificationsFromReports();
        const allNotifications = [...generatedNotifications, ...notificationsData];
        setNotifications(allNotifications);
      } catch (notifError) {
        console.error('Failed to load notifications:', notifError);
        // Check if it's an auth error
        handleAuthError(notifError);
        // Continue with just the generated notifications
        const generatedNotifications = generateNotificationsFromReports();
        setNotifications(generatedNotifications);
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
      if (!handleAuthError(error)) {
        // Only show general error if it's not an auth error
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
      // Check if it's an auth error but don't exit early
      handleAuthError(error);
      // Non-critical, continue without analytics
    }
  };

  const loadWeeklyStats = async () => {
    try {
      const data = await getWeeklyReportStats();
      setWeeklyStats(data);
    } catch (error) {
      console.error('Error loading weekly stats:', error);
      // Check if it's an auth error but don't exit early
      handleAuthError(error);
      // Non-critical, continue without weekly stats
    }
  };

  const loadStatusSummary = async () => {
    try {
      const data = await getReportStatusSummary('week');
      setStatusSummary(data);
    } catch (error) {
      console.error('Error loading status summary:', error);
      // Check if it's an auth error but don't exit early
      handleAuthError(error);
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
      // Check if it's an auth error
      handleAuthError(error);
    }
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
      await getCurrentLocation(); // Get location for weather display
      await fetchFieldWorkerReports(); // Get field worker's reports
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
      // Check if it's an auth error
      if (!handleAuthError(error)) {
        // Only show this error if it's not auth-related
        Alert.alert('Error', 'Failed to update repair status');
      }
    }
  };

  // Function to handle authentication errors
  const handleAuthError = (error) => {
    if (authErrorShown) return false; // Prevent multiple alerts
    
    if (error?.message && (
      error.message.includes('token') || 
      error.message.includes('expired') ||
      error.message.includes('unauthorized') ||
      error.message.includes('401') ||
      error.message.includes('auth') ||
      error.message.toLowerCase().includes('no valid auth token')
    )) {
      setAuthErrorShown(true);
      Alert.alert(
        'Session Expired',
        'Your session has expired. Please log in again.',
        [{ 
          text: 'OK', 
          onPress: async () => {
            try {
              // Perform logout
              await logout();
              // Navigate to login
              navigation.reset({
                index: 0,
                routes: [{ name: 'Login' }],
              });
            } catch (logoutError) {
              console.error('Error during logout:', logoutError);
              // Force navigation to login even if logout fails
              navigation.reset({
                index: 0,
                routes: [{ name: 'Login' }],
              });
            } 
          } 
        }]
      );
      return true; // Indicates auth error was handled
    }
    return false; // Not an auth error
  };

  // Generate notifications from reports and AI data
  const generateNotificationsFromReports = () => {
    const notificationArray = [];
    
    // Add notifications for assigned reports
    if (Array.isArray(nearbyReports) && nearbyReports.length > 0) {
      nearbyReports.forEach((report, index) => {
        notificationArray.push({
          id: `report_${report._id}`,
          title: `Assigned: ${report.damageType}`,
          message: `You have been assigned to inspect a ${report.severity || 'medium'} severity ${report.damageType} damage at ${report.location}.`,
          time: formatTimeAgo(new Date(report.assignedAt || report.createdAt)),
          read: false,
          icon: 'clipboard-alert',
          type: report.severity === 'HIGH' ? 'warning' : 'info',
          reportId: report._id
        });
      });
    }
    
    // Add notifications based on upcoming tasks
    if (Array.isArray(nearbyReports) && nearbyReports.length > 0) {
      // High priority reports notification
      const highPriorityReports = nearbyReports.filter(report => 
        report.priority > 5 || report.severity === 'HIGH'
      );
      if (highPriorityReports.length > 0) {
        notificationArray.push({
          id: `high_priority`,
          title: 'High Priority Tasks',
          message: `You have ${highPriorityReports.length} high priority tasks that require attention.`,
          time: 'Now',
          read: false,
          icon: 'alert-circle',
          type: 'warning'
        });
      }
    }
    return notificationArray;
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />
      
      <HeaderComponent fieldWorker={fieldWorker} cityStats={cityStats} />

      <ScrollView 
        style={styles.content}
        contentContainerStyle={styles.contentContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[theme.colors.primary, theme.colors.primaryDark]} tintColor={theme.colors.primary} />
        }
        showsVerticalScrollIndicator={false}
      >
        <LocationInfoComponent locationName={locationName} />
        <WeatherComponent weatherData={weatherData} locationName={locationName} />
        <QuickActionsComponent actions={quickActions} navigation={navigation} />
        <StatsComponent cityStats={cityStats} />
        <RecentReportsComponent 
          reports={nearbyReports}
          navigation={navigation}
          formatTimeAgo={formatTimeAgo}
          handleQuickStatusUpdate={handleQuickStatusUpdate}
          loading={loading}
        />
        <NotificationsComponent 
          notifications={notifications}
          navigation={navigation}
          handleMarkAsRead={handleMarkAsRead}
          handleNotificationPress={handleNotificationPress}
          loading={loading}
        />
        <View style={styles.spacer} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 120 : 100, // Adjusted for iOS
  },
  spacer: {
    height: 50
  }
});

export default HomeScreen;
