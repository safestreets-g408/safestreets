import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { ActivityIndicator } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from 'react-native-paper';
import { useAuth } from '../context/AuthContext';

// Import screens
import LoginScreen from '../screens/LoginScreen';
import MainTabs from './MainTabs'; // Assuming MainTabs is defined elsewhere

const Stack = createStackNavigator();

const Navigation = ({ MainTabs }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const theme = useTheme();

  if (isLoading) {
    return <ActivityIndicator size="large" color={theme.colors.primary} />;
  }

  return (
    <NavigationContainer
      fallback={<ActivityIndicator size="large" color={theme.colors.primary} />}
    >
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            elevation: 0,
            shadowOpacity: 0,
          },
        }}
      >
        {!isAuthenticated ? (
          <Stack.Screen 
            name="Login" 
            component={LoginScreen}
            options={{ headerShown: false }}
          />
        ) : (
          <>
            <Stack.Screen 
              name="MainTabs"
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
  );
};

export default Navigation;
