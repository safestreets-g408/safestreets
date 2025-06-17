import React from 'react';
import { View, StyleSheet } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AuthProvider } from '../../context/AuthContext';
import { ErrorBoundary } from 'react-native-error-boundary';
import { DefaultTheme, Provider as PaperProvider } from 'react-native-paper';
import theme from '../../theme';

const AppProvider = ({ children, onError }) => {
  const handleError = (error, stackTrace) => {
    console.error('Error caught by ErrorBoundary:', error);
    if (onError) {
      onError(error, stackTrace);
    }
  };

  return (
    <ErrorBoundary onError={handleError} FallbackComponent={() => <ErrorFallback />}>
      <SafeAreaProvider>
        <PaperProvider theme={DefaultTheme}>
          <AuthProvider>
            <View style={styles.container}>{children}</View>
          </AuthProvider>
        </PaperProvider>
      </SafeAreaProvider>
    </ErrorBoundary>
  );
};

const ErrorFallback = () => {
  return (
    <View style={styles.errorContainer}>
      <Text style={styles.errorText}>Something went wrong</Text>
      <Button 
        mode="contained" 
        onPress={() => RNRestart.Restart()}
        style={styles.restartButton}
      >
        Restart App
      </Button>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 18,
    marginBottom: 20,
    textAlign: 'center',
  },
  restartButton: {
    marginTop: 20,
  },
});

export default AppProvider;
