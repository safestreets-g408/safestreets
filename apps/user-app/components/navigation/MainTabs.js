import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useTheme } from 'react-native-paper';
import { useThemeContext } from '../../context/ThemeContext';

// Import screens
import HomeScreen from '../../screens/HomeScreen';
import ReportsScreen from '../../screens/ReportsScreen';
import CameraScreen from '../../screens/CameraScreen';
import ProfileScreen from '../../screens/ProfileScreen';
import TaskManagementScreen from '../../screens/TaskManagementScreen';
import ChatScreen from '../../screens/ChatScreen';

const Tab = createBottomTabNavigator();

const MainTabs = () => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;
          // Use slightly smaller icon size for better fit
          const iconSize = size - 2;

          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'Reports') {
            iconName = focused ? 'clipboard-text' : 'clipboard-text-outline';
          } else if (route.name === 'Camera') {
            iconName = focused ? 'camera' : 'camera-outline';
          } else if (route.name === 'Tasks') {
            iconName = focused ? 'clipboard-check' : 'clipboard-check-outline';
          } else if (route.name === 'Chat') {
            iconName = focused ? 'chat' : 'chat-outline';
          } else if (route.name === 'Profile') {
            iconName = focused ? 'account' : 'account-outline';
          }

          return <MaterialCommunityIcons name={iconName} size={iconSize} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: isDarkMode ? '#94a3b8' : '#6b7280',
        tabBarHideOnKeyboard: true,
        tabBarStyle: {
          backgroundColor: isDarkMode ? theme.colors.surface : theme.colors.surface,
          borderTopColor: isDarkMode ? theme.colors.border : '#e5e7eb',
          borderTopWidth: 1,
          elevation: isDarkMode ? 0 : 10,
          shadowColor: isDarkMode ? 'transparent' : '#000',
          shadowOffset: { width: 0, height: -2 },
          shadowOpacity: isDarkMode ? 0 : 0.1,
          shadowRadius: isDarkMode ? 0 : 8,
          paddingBottom: 4,
          paddingTop: 4,
          height: 70,
        },
        tabBarLabelStyle: {
          fontSize: 10,
          fontWeight: '500',
          marginTop: 2,
        },
        tabBarItemStyle: {
          paddingHorizontal: 4,
        },
        headerShown: false,
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Reports" component={ReportsScreen} />
      <Tab.Screen name="Camera" component={CameraScreen} />
      <Tab.Screen name="Tasks" component={TaskManagementScreen} />
      <Tab.Screen name="Chat" component={ChatScreen} options={{ tabBarBadge: null }} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
};

export default MainTabs;
