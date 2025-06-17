import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Button } from 'react-native-paper';

const ErrorFallback = ({ error, resetError }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Something went wrong</Text>
      <Text style={styles.subtitle}>We're sorry for the inconvenience</Text>
      {error && <Text style={styles.errorMessage}>{error.toString()}</Text>}
      {resetError && (
        <Button 
          mode="contained" 
          onPress={resetError}
          style={styles.button}
        >
          Try Again
        </Button>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 20,
  },
  errorMessage: {
    color: 'red',
    marginBottom: 20,
    textAlign: 'center',
  },
  button: {
    marginTop: 10,
  },
});

export default ErrorFallback;
