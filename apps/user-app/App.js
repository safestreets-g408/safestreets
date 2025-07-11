import React, { useState, useEffect } from 'react';
import { StatusBar } from 'react-native';
import { useTheme } from 'react-native-paper';
import { useThemeContext } from './context/ThemeContext';

// Import layout components
import AppProvider from './components/layout/AppProvider';

// Import navigation
import AppNavigator from './components/navigation/AppNavigator';

// Import UI components
import { LoadingSpinner } from './components/ui';
import { NotificationToast } from './components/notifications';

// Import hooks
import { useAppState, useNotificationHandler } from './hooks';

// App content component with access to AuthContext
const AppContent = () => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const { showToast, currentToast, hideToast } = useNotificationHandler();
  const [showNotificationModal, setShowNotificationModal] = useState(false);
  
  try {
    const { isLoading } = useAppState();
    
    if (isLoading) {
      return <LoadingSpinner />;
    }
    
    return (
      <>
        <StatusBar 
          barStyle={isDarkMode ? "light-content" : "dark-content"} 
          backgroundColor={theme?.colors?.background || '#f9fafb'}
          animated={true}
        />
        <AppNavigator />
        
        {/* Toast notification */}
        {showToast && currentToast && (
          <NotificationToast
            notification={currentToast}
            onPress={() => {
              // Handle toast press - navigate or show details
              hideToast();
            }}
            onDismiss={hideToast}
          />
        )}
      </>
    );
  } catch (error) {
    console.error('Error in AppContent:', error);
    return (
      <>
        <StatusBar 
          barStyle="dark-content" 
          backgroundColor={theme?.colors?.background || '#f9fafb'} 
          animated={true}
        />
        <AppNavigator />
      </>
    );
  }
};

// Main App component
export default function App() {
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Handle global errors
  const handleGlobalError = (error, stackTrace) => {
    console.error('Global error:', error, stackTrace);
    setError(error);
  };
  
  // Simple initial loading effect
  useEffect(() => {
    const timer = setTimeout(() => {
      setInitialLoading(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  if (initialLoading) {
    return (
      <AppProvider>
        <LoadingSpinner />
      </AppProvider>
    );
  }

  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
