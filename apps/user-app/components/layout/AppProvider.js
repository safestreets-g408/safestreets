import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AuthProvider } from '../../context/AuthContext';
import { ThemeProvider } from '../../context/ThemeContext';
import { SocketProvider } from '../../context/SocketContext';
import { DefaultTheme, Provider as PaperProvider } from 'react-native-paper';

// Default fallback theme - very simple to avoid any potential errors
const fallbackTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#2563eb',
    background: '#f9fafb',
    surface: '#ffffff',
    text: '#111827',
    error: '#dc2626',
  }
};

// React Error Boundary as a class component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render shows fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI when an error occurs
      return this.props.fallback;
    }

    return this.props.children;
  }
}

// Theme provider wrapped in error boundary
function SafeThemeProvider({ children }) {
  return (
    <ErrorBoundary
      fallback={
        <PaperProvider theme={fallbackTheme}>
          <View style={styles.container}>
            <Text style={{ textAlign: 'center', padding: 20, color: '#666' }}>
              There was an issue loading the app theme. Using default appearance.
            </Text>
            {children}
          </View>
        </PaperProvider>
      }
    >
      <ThemeProvider>
        {children}
      </ThemeProvider>
    </ErrorBoundary>
  );
}

// Main app provider component
function AppProvider({ children }) {
  return (
    <SafeAreaProvider>
      <SafeThemeProvider>
        <AuthProvider>
          <SocketProvider>
            <View style={styles.container}>
              {children}
            </View>
          </SocketProvider>
        </AuthProvider>
      </SafeThemeProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  }
});

export default AppProvider;
