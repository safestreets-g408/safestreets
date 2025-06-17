import React from 'react';
import { StatusBar } from 'react-native';

// Import layout components
import { AppProvider } from './components/layout';

// Import navigation
import AppNavigator from './components/navigation/AppNavigator';

// Import UI components
import { LoadingSpinner, ErrorFallback } from './components/ui';

// Import hooks
import { useAppState } from './hooks';

// App content component
const AppContent = () => {
  return <AppNavigator />;
};

// Main App component
export default function App() {
  const { isLoading, handleError } = useAppState();

  if (isLoading) {
    return (
      <AppProvider onError={handleError}>
        <LoadingSpinner />
      </AppProvider>
    );
  }

  return (
    <AppProvider onError={handleError}>
      <StatusBar barStyle="light-content" backgroundColor="#003366" />
      <AppContent />
    </AppProvider>
  );
}
