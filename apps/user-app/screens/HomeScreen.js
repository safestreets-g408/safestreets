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
  ActivityIndicator
} from 'react-native';
import { Card, Title, Paragraph, Badge, IconButton, Divider, Button, Avatar, Chip, ProgressBar } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import { Platform } from 'react-native';


const HomeScreen = ({ navigation }) => {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showWeather, setShowWeather] = useState(true);
  const [activeTab, setActiveTab] = useState('home');
  const [cityStats, setCityStats] = useState({
    reportsThisWeek: 87,
    repairsCompleted: 42,
    pendingIssues: 31,
    completionRate: 0.68
  });
  const [location, setLocation] = useState(null);
  const [locationName, setLocationName] = useState('Loading...');
  const [errorMsg, setErrorMsg] = useState(null);
  const animationRef = useRef(null);

  useEffect(() => {
    // Start animation when component mounts
    if (animationRef.current) {
      animationRef.current.play();
    }
    
    // Get current location
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        setLocationName('Location unavailable');
        return;
      }

      try {
        let location = await Location.getCurrentPositionAsync({});
        setLocation(location);
        
        // Get location name from coordinates
        let geocode = await Location.reverseGeocodeAsync({
          latitude: location.coords.latitude,
          longitude: location.coords.longitude
        });
        
        if (geocode && geocode.length > 0) {
          const { city, region } = geocode[0];
          setLocationName(city || region || 'Unknown location');
        }
      } catch (error) {
        setErrorMsg('Could not fetch location');
        setLocationName('Location unavailable');
      }
    })();
    
    // Simulate fetching notifications
    setTimeout(() => {
      setNotifications([
        {
          id: '1',
          title: 'New Report Approved',
          message: 'Your pothole report on Main Street has been approved',
          time: '2 hours ago',
          read: false,
          type: 'success',
          icon: 'check-circle'
        },
        {
          id: '2',
          title: 'Maintenance Scheduled',
          message: 'Road repair scheduled for your reported damage on Oak Avenue',
          time: '1 day ago',
          read: true,
          type: 'info',
          icon: 'calendar'
        },
        {
          id: '3',
          title: 'Report Status Update',
          message: 'Your report #1234 is now under review by city officials',
          time: '2 days ago',
          read: true,
          type: 'info',
          icon: 'eye'
        },
        {
          id: '4',
          title: 'Repair Completed',
          message: 'The pothole you reported on Pine Street has been fixed',
          time: '3 days ago',
          read: true,
          type: 'success',
          icon: 'check-circle'
        }
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate data refresh
    setTimeout(() => {
      // Update with new data
      setCityStats({
        ...cityStats,
        reportsThisWeek: cityStats.reportsThisWeek + Math.floor(Math.random() * 5),
        repairsCompleted: cityStats.repairsCompleted + Math.floor(Math.random() * 3)
      });
      setRefreshing(false);
    }, 1500);
  }, [cityStats]);

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
      delay={parseInt(item.id) * 100}
    >
      <TouchableOpacity>
        <Card style={[styles.notificationCard, !item.read && styles.unreadCard]}>
          <Card.Content style={styles.notificationContent}>
            <View style={styles.notificationHeader}>
              <View style={styles.notificationTitleContainer}>
                <Avatar.Icon 
                  size={36} 
                  icon={item.icon} 
                  style={{
                    backgroundColor: item.type === 'success' ? '#4caf50' : '#2196f3'
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
                onPress={() => {}} 
                style={{height: 30}}
              >
                {item.read ? "Read" : "Mark as read"}
              </Chip>
              <IconButton icon="dots-vertical" size={20} onPress={() => {}} />
            </View>
          </Card.Content>
        </Card>
      </TouchableOpacity>
    </Animatable.View>
  );

  const renderRecentReportItem = (item) => (
    <TouchableOpacity 
      style={styles.recentReportCard}
      onPress={() => navigation.navigate('ViewReport', { reportId: item.id })}
    >
      <Image 
        source={{ uri: item.image }} 
        style={styles.recentReportImage} 
      />
      <View style={styles.recentReportContent}>
        <Text style={styles.recentReportTitle}>{item.title}</Text>
        <View style={styles.recentReportFooter}>
          <Chip 
            style={{
              backgroundColor: item.status === 'approved' ? '#e8f5e9' : '#e3f2fd',
              height: 24
            }}
            textStyle={{fontSize: 10}}
          >
            {item.status}
          </Chip>
          <Text style={styles.recentReportDate}>{item.date}</Text>
        </View>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor="#1a73e8" />
      <LinearGradient
        colors={['#4285f4', '#4285f4', '#4285f4']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <View>
            <Text style={styles.greeting}>Hello, John</Text>
            <Text style={styles.subGreeting}>Welcome back to Road Reporter</Text>
          </View>
          <View style={styles.headerActions}>
            <TouchableOpacity style={styles.headerButton}>
              <IconButton icon="magnify" color="#fff" size={28} />
            </TouchableOpacity>
          </View>
        </View>
      </LinearGradient>

      <ScrollView 
        style={styles.content}
        contentContainerStyle={styles.contentContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={['#1a73e8']} />
        }
        showsVerticalScrollIndicator={false}
      >
        {showWeather && (
          <Animatable.View animation="fadeIn" duration={800}>
            <Card style={styles.weatherCard}>
              <LinearGradient
                colors={['#1a73e8', '#4285f4', '#5e97f6']}
                style={styles.weatherGradient}
                start={{x: 0, y: 0}}
                end={{x: 1, y: 0}}
              >
                <View style={styles.weatherHeader}>
                  <View>
                    <Text style={styles.weatherCity}>{locationName}</Text>
                    <Text style={styles.weatherDate}>Today, June 15</Text>
                  </View>
                  <TouchableOpacity onPress={() => setShowWeather(false)}>
                    <IconButton icon="close" color="#fff" size={20} />
                  </TouchableOpacity>
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
        )}

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
          {recentReports.map(item => (
            <Animatable.View key={item.id} animation="fadeInRight" duration={500} delay={parseInt(item.id) * 100}>
              {renderRecentReportItem(item)}
            </Animatable.View>
          ))}
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
    backgroundColor: '#f5f8ff',
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
    elevation: 8,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
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
  headerButton: {
    position: 'relative',
    padding: 5,
  },
  notificationBadge: {
    position: 'absolute',
    top: 5,
    right: 5,
    backgroundColor: '#ff5252',
  },
  greeting: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#fff',
  },
  subGreeting: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    marginTop: 4,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 120 : 100, // Adjusted for iOS
  },
  weatherCard: {
    marginVertical: 10,
    borderRadius: 20,
    elevation: 4,
    overflow: 'hidden',
    borderWidth: 0,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
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
    marginVertical: 10,
    borderRadius: 20,
    elevation: 4,
    overflow: 'hidden',
    borderColor: 'rgba(26, 115, 232, 0.2)',
    borderWidth: 1,
    backgroundColor: '#fff',
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  statsGradient: {
    borderRadius: 20,
  },
  statsContent: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 20,
  },
  statItem: {
    alignItems: 'center',
    paddingHorizontal: 5,
  },
  statNumber: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1a73e8',
  },
  statLabel: {
    fontSize: 14,
    color: '#4285f4',
    marginTop: 4,
    fontWeight: '500',
    textAlign: 'center',
  },
  statDivider: {
    width: 1,
    backgroundColor: 'rgba(26, 115, 232, 0.2)',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 16,
    paddingHorizontal: 4,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#0d47a1',
  },
  seeAllText: {
    color: '#1a73e8',
    fontWeight: '500',
    padding: 8, // Added padding for better touch target
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
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  quickActionGradient: {
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
    height: 130,
  },
  quickActionText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
    marginTop: 8,
    textAlign: 'center',
  },
  cityStatsCard: {
    marginTop: 10,
    borderRadius: 20,
    elevation: 4,
    shadowColor: '#1a73e8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    backgroundColor: '#fff',
    padding: 16, // Added explicit padding
  },
  cityStatsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#0d47a1',
    marginBottom: 15,
  },
  cityStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
    flexWrap: 'wrap',
  },
  cityStatItem: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '48%',
    marginBottom: 10,
  },
  cityStatValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#0d47a1',
  },
  cityStatLabel: {
    fontSize: 12,
    color: '#4285f4',
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
  spacer: {
    height: Platform.OS === 'ios' ? 100 : 80, 
  },
});

export default HomeScreen;
