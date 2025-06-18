import React from 'react';
import { View, StyleSheet } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AuthProvider } from '../../context/AuthContext';
import { Provider as PaperProvider } from 'react-native-paper';
import theme from '../../theme';

// Simple function component without error handling for now
function AppProvider(props) {
  return (
    <SafeAreaProvider>
      <PaperProvider theme={theme}>
        <AuthProvider>
          <View style={styles.container}>
            {props.children}
          </View>
        </AuthProvider>
      </PaperProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  }
});

export default AppProvider;
