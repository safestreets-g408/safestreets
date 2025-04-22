// App.js - Main application file
import React, { useState, useEffect, useRef } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Provider as PaperProvider, DefaultTheme, FAB, Portal, Modal, Button as PaperButton } from 'react-native-paper';
import { View, Text, StatusBar, StyleSheet, ActivityIndicator, Dimensions, TouchableOpacity, Animated, Easing } from 'react-native';
import { Camera, User, Clipboard, Home, Bell, Settings, HelpCircle } from 'react-native-feather';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { BlurView } from 'expo-blur';

// Import screens
import CameraScreen from './screens/CameraScreen';
import ReportsScreen from './screens/ReportsScreen';
import ViewReportScreen from './screens/ViewReportScreen';
import ProfileScreen from './screens/ProfileScreen';
import LoginScreen from './screens/LoginScreen';
import TaskManagement from './screens/TaskManagement';
import HomeScreen from './screens/HomeScreen';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();
const { width, height } = Dimensions.get('window');

// Enhanced theme with blue color palette
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#1a73e8',
    accent: '#4285f4',
    background: '#f5f8ff',
    surface: '#ffffff',
    text: '#1a237e',
    error: '#d32f2f',
    success: '#43a047',
    warning: '#ffa000',
    info: '#2196f3',
  },
  roundness: 12,
  animation: {
    scale: 1.0,
  },
};

// Main Tab Navigation
const MainTabs = () => {
  const [fabOpen, setFabOpen] = useState(false);
  const [showTips, setShowTips] = useState(false);
  const spinValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (fabOpen) {
      Animated.timing(spinValue, {
        toValue: 1,
        duration: 300,
        easing: Easing.linear,
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(spinValue, {
        toValue: 0,
        duration: 300,
        easing: Easing.linear,
        useNativeDriver: true,
      }).start();
    }
  }, [fabOpen, spinValue]);

  const spin = spinValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '45deg'],
  });


  const renderTabBarBadge = (count) => {
    return count > 0 ? (
      <View style={styles.badge}>
        <Text style={styles.badgeText}>{count}</Text>
      </View>
    ) : null;
  };

  return (
    <>
      <Tab.Navigator
        screenOptions={{
          tabBarActiveTintColor: '#1a73e8',
          tabBarInactiveTintColor: '#90a4ae',
          tabBarStyle: { 
            height: 65,
            paddingBottom: 10,
            paddingTop: 8,
            borderTopWidth: 0,
            elevation: 8,
            shadowColor: '#1565c0',
            shadowOffset: { width: 0, height: -3 },
            shadowOpacity: 0.1,
            shadowRadius: 5,
            borderTopLeftRadius: 20,
            borderTopRightRadius: 20,
            backgroundColor: '#ffffff',
          },
          tabBarLabelStyle: {
            fontSize: 11,
            fontWeight: '600',
            marginTop: 2,
          },
          headerStyle: {
            elevation: 0,
            shadowOpacity: 0,
            borderBottomWidth: 0,
          },
          headerTitleStyle: {
            fontWeight: 'bold',
            fontSize: 18,
          },
        }}
      >
        <Tab.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{
            tabBarIcon: ({ color, focused }) => (
              <Animatable.View
                animation={focused ? 'pulse' : undefined}
                iterationCount={focused ? 'infinite' : 1}
                duration={2000}
              >
                <Home color={color} size={24} />
                {renderTabBarBadge(3)}
              </Animatable.View>
            ),
            headerTitle: "Road Damage Reporter",
            headerTitleAlign: 'center',
            headerBackground: () => (
              <LinearGradient
                colors={['#2196f3', '#1976d2', '#0d47a1']}
                style={{ flex: 1 }}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              />
            ),
            headerTintColor: '#ffffff',
            headerRight: () => (
              <TouchableOpacity style={styles.headerButton}>
                <Bell color="#ffffff" size={22} />
                <View style={styles.notificationBadge}>
                  <Text style={styles.notificationText}>5</Text>
                </View>
              </TouchableOpacity>
            ),
          }}
        />
        <Tab.Screen 
          name="Camera" 
          component={CameraScreen} 
          options={{
            tabBarIcon: ({ color, focused }) => (
              <Animatable.View
                animation={focused ? 'pulse' : undefined}
                iterationCount={focused ? 'infinite' : 1}
                duration={2000}
              >
                <Camera color={color} size={24} />
              </Animatable.View>
            ),
            headerTitle: "Report Damage",
            headerTitleAlign: 'center',
            headerBackground: () => (
              <LinearGradient
                colors={['#2196f3', '#1976d2', '#0d47a1']}
                style={{ flex: 1 }}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              />
            ),
            headerTintColor: '#ffffff',
          }}
        />
        <Tab.Screen 
          name="Reports" 
          component={ReportsScreen}
          options={{
            tabBarIcon: ({ color, focused }) => (
              <Animatable.View
                animation={focused ? 'pulse' : undefined}
                iterationCount={focused ? 'infinite' : 1}
                duration={2000}
              >
                <Clipboard color={color} size={24} />
                {renderTabBarBadge(2)}
              </Animatable.View>
            ),
            headerTitle: "My Reports",
            headerTitleAlign: 'center',
            headerBackground: () => (
              <LinearGradient
                colors={['#2196f3', '#1976d2', '#0d47a1']}
                style={{ flex: 1 }}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              />
            ),
            headerTintColor: '#ffffff',
          }}
        />
        <Tab.Screen 
          name="Task" 
          component={TaskManagement}
          options={{
            tabBarIcon: ({ color, focused }) => (
              <Animatable.View
                animation={focused ? 'pulse' : undefined}
                iterationCount={focused ? 'infinite' : 1}
                duration={2000}
              >
                <Animatable.Text 
                  animation={focused ? "bounceIn" : undefined}
                  style={{fontSize: 22, color: color}}
                >
                  📋
                </Animatable.Text>
                {renderTabBarBadge(1)}
              </Animatable.View>
            ),
            headerTitle: "Task Management",
            headerTitleAlign: 'center',
            headerBackground: () => (
              <LinearGradient
                colors={['#2196f3', '#1976d2', '#0d47a1']}
                style={{ flex: 1 }}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              />
            ),
            headerTintColor: '#ffffff',
          }}
        />
        <Tab.Screen 
          name="Profile" 
          component={ProfileScreen}
          options={{
            tabBarIcon: ({ color, focused }) => (
              <Animatable.View
                animation={focused ? 'pulse' : undefined}
                iterationCount={focused ? 'infinite' : 1}
                duration={2000}
              >
                <User color={color} size={24} />
              </Animatable.View>
            ),
            headerTitle: "My Profile",
            headerTitleAlign: 'center',
            headerBackground: () => (
              <LinearGradient
                colors={['#2196f3', '#1976d2', '#0d47a1']}
                style={{ flex: 1 }}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              />
            ),
            headerTintColor: '#ffffff',
          }}
        />
      </Tab.Navigator>

      <Modal
        visible={showTips}
        onDismiss={() => setShowTips(false)}
        contentContainerStyle={styles.modalContainer}
      >
        <BlurView intensity={90} style={styles.blurContainer}>
          <Animatable.View animation="fadeIn" duration={500} style={styles.tipContent}>
            <Text style={styles.tipTitle}>Quick Tips</Text>
            <View style={styles.tipItem}>
              <HelpCircle color="#1a73e8" size={20} />
              <Text style={styles.tipText}>Swipe right on reports to share them</Text>
            </View>
            <View style={styles.tipItem}>
              <HelpCircle color="#1a73e8" size={20} />
              <Text style={styles.tipText}>Long press on tasks to see more options</Text>
            </View>
            <View style={styles.tipItem}>
              <HelpCircle color="#1a73e8" size={20} />
              <Text style={styles.tipText}>Double tap on map to drop a pin</Text>
            </View>
            <PaperButton 
              mode="contained" 
              onPress={() => setShowTips(false)}
              style={styles.tipButton}
              labelStyle={{ color: '#ffffff' }}
            >
              Got it!
            </PaperButton>
          </Animatable.View>
        </BlurView>
      </Modal>
    </>
  );
};

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const animationRef = useRef(null);
  
  // Enhanced loading with progress
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 3000);
    
    // Simulate loading progress
    const interval = setInterval(() => {
      setLoadingProgress(prev => {
        const newProgress = prev + 0.1;
        return newProgress > 1 ? 1 : newProgress;
      });
    }, 300);
    
    if (animationRef.current) {
      animationRef.current.play();
    }
    
    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, []);
  
  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <Animatable.View 
          animation="pulse" 
          easing="ease-out" 
          iterationCount="infinite"
          style={styles.indicatorContainer}
        >
          <View style={styles.progressBarContainer}>
            <Animated.View 
              style={[
                styles.progressBar, 
                { width: loadingProgress * (width - 80) }
              ]} 
            />
          </View>
        </Animatable.View>
        <Animatable.Text 
          animation="fadeIn" 
          delay={400}
          duration={800}
          style={styles.loadingText}
        >
          Road Damage Reporter
        </Animatable.Text>
        <Animatable.Text 
          animation="fadeIn" 
          delay={800}
          duration={800}
          style={styles.loadingSubText}
        >
          Making roads safer together
        </Animatable.Text>
        <Animatable.Text 
          animation="fadeIn" 
          delay={1200}
          duration={800}
          style={styles.versionText}
        >
          v2.0.1
        </Animatable.Text>
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <PaperProvider theme={theme}>
        <NavigationContainer>
          <StatusBar barStyle="light-content" backgroundColor="#0d47a1" />
          <Stack.Navigator
            screenOptions={{
              headerStyle: {
                elevation: 0,
                shadowOpacity: 0,
              },
              cardStyle: { backgroundColor: '#f5f8ff' },
              headerTitleStyle: {
                fontWeight: 'bold',
              },
              headerBackTitleVisible: false,
              cardStyleInterpolator: ({ current, layouts }) => {
                return {
                  cardStyle: {
                    transform: [
                      {
                        translateX: current.progress.interpolate({
                          inputRange: [0, 1],
                          outputRange: [layouts.screen.width, 0],
                        }),
                      },
                    ],
                  },
                };
              },
            }}
          >
            {!isLoggedIn ? (
              <Stack.Screen 
                name="Login" 
                component={LoginScreen}
                options={{ headerShown: false }}
                initialParams={{ onLogin: handleLogin }}
              />
            ) : (
              <>
                <Stack.Screen 
                  name="Main" 
                  component={MainTabs} 
                  options={{ headerShown: false }}
                />
                <Stack.Screen
                  name="ViewReport"
                  component={ViewReportScreen}
                  options={{ 
                    title: "Report Details",
                    headerTitleAlign: 'center',
                    headerBackground: () => (
                      <LinearGradient
                        colors={['#2196f3', '#1976d2', '#0d47a1']}
                        style={{ flex: 1 }}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                      />
                    ),
                    headerTintColor: '#ffffff',
                    headerBackTitleVisible: false,
                  }}
                />
              </>
            )}
          </Stack.Navigator>
        </NavigationContainer>
      </PaperProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f8ff',
  },
  logoContainer: {
    marginBottom: 35,
    shadowColor: '#1565c0',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 8,
    elevation: 10,
  },
  loadingGradient: {
    width: 150,
    height: 150,
    borderRadius: 75,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  lottieAnimation: {
    width: 120,
    height: 120,
  },
  logoText: {
    fontSize: 42,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  indicatorContainer: {
    marginBottom: 25,
    width: width - 80,
  },
  progressBarContainer: {
    height: 6,
    backgroundColor: 'rgba(25, 118, 210, 0.2)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#1976d2',
    borderRadius: 4,
  },
  loadingText: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1976d2',
    marginBottom: 10,
  },
  loadingSubText: {
    fontSize: 16,
    color: '#2196f3',
    fontStyle: 'italic',
    marginBottom: 10,
  },
  versionText: {
    fontSize: 12,
    color: '#90caf9',
    position: 'absolute',
    bottom: 20,
    right: 20,
  },
  badge: {
    position: 'absolute',
    right: -6,
    top: -3,
    backgroundColor: '#f44336',
    borderRadius: 10,
    width: 16,
    height: 16,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ffffff',
  },
  badgeText: {
    color: '#ffffff',
    fontSize: 9,
    fontWeight: 'bold',
  },
  headerButton: {
    marginRight: 15,
    position: 'relative',
  },
  notificationBadge: {
    position: 'absolute',
    right: -5,
    top: -5,
    backgroundColor: '#f44336',
    borderRadius: 10,
    width: 16,
    height: 16,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ffffff',
  },
  notificationText: {
    color: '#ffffff',
    fontSize: 9,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    margin: 20,
  },
  blurContainer: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  tipContent: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 16,
    padding: 24,
    width: '85%',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  tipTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1a73e8',
    marginBottom: 20,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
    width: '100%',
    paddingHorizontal: 10,
  },
  tipText: {
    fontSize: 16,
    color: '#333',
    marginLeft: 10,
  },
  tipButton: {
    marginTop: 20,
    backgroundColor: '#1a73e8',
    paddingHorizontal: 30,
    borderRadius: 8,
  },
});