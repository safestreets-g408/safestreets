import React, { useState, useEffect } from 'react';
import { StatusBar } from 'react-native';

// Import layout components
import AppProvider from './components/layout/AppProvider';

// Import navigation
import AppNavigator from './components/navigation/AppNavigator';

// Import UI components
import { LoadingSpinner } from './components/ui';

// Import hooks
import { useAppState } from './hooks';

// App content component with access to AuthContext
const AppContent = () => {
  try {
    const { isLoading } = useAppState();
    
    if (isLoading) {
      return <LoadingSpinner />;
    }
    
    return (
      <>
        <StatusBar barStyle="light-content" backgroundColor="#003366" />
        <AppNavigator />
      </>
    );
  } catch (error) {
    console.error('Error in AppContent:', error);
    return (
      <>
        <StatusBar barStyle="light-content" backgroundColor="#003366" />
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
