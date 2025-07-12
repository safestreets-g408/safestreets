import React, { useEffect, useState } from 'react';
import { StyleSheet, View, StatusBar, Text, ActivityIndicator, RefreshControl, ScrollView, Alert } from 'react-native';
import { useTheme, Button, Snackbar } from 'react-native-paper';
import { useThemeContext } from '../context/ThemeContext';
import AdminChatList from '../components/chat/AdminChatList';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSocket } from '../context/SocketContext';
import { chatService } from '../utils/chatAPI';

const ChatScreen = () => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const { socket, connected, reconnectSocket, connectionError } = useSocket();
  const [refreshing, setRefreshing] = useState(false);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  
  // Track reconnection attempts to prevent excessive retries
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastReconnectTime, setLastReconnectTime] = useState(0);
  const MAX_RECONNECT_ATTEMPTS = 3;
  const RECONNECT_COOLDOWN = 60000; // 1 minute cooldown between reconnect attempts
  
  // Check socket connection and try to reconnect if needed
  useEffect(() => {
    const checkConnection = async () => {
      const now = Date.now();
      const timeSinceLastReconnect = now - lastReconnectTime;
      
      // Check if we should attempt reconnection based on state and cooldown
      const shouldAttemptReconnect = 
        !connected && 
        reconnectAttempts < MAX_RECONNECT_ATTEMPTS && 
        timeSinceLastReconnect > RECONNECT_COOLDOWN;
      
      if (shouldAttemptReconnect) {
        console.log(`ChatScreen: Socket not connected, attempting to reconnect (attempt ${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})`);
        setLastReconnectTime(now);
        setReconnectAttempts(prev => prev + 1);
        reconnectSocket();
        
        // Also check chat service health
        try {
          const health = await chatService.healthCheck();
          console.log('Chat service health status:', health);
          setServiceStatus(health);
          
          if (!health.healthy) {
            setSnackbarMessage('Chat service experiencing issues. Some features may be limited.');
            setSnackbarVisible(true);
          }
        } catch (err) {
          console.error('Health check error:', err);
          setServiceStatus({ healthy: false, error: err.message });
        }
      } else if (socket && connected) {
        // Connection looks good - reset counters and check with a ping
        console.log('ChatScreen: Socket connection verified');
        socket.emit('ping_server'); // Test the connection
        setReconnectAttempts(0);
      } else if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        // We've hit the limit - notify user
        if (timeSinceLastReconnect > RECONNECT_COOLDOWN) {
          // Reset counter after cooldown
          setReconnectAttempts(0);
          console.log('Resetting reconnect attempts after cooldown period');
        }
      }
    };
    
    checkConnection();
    
    // Set up periodic connection checks with a smarter interval
    // Check every 15 seconds if disconnected, every 60 seconds if connected
    const intervalId = setInterval(checkConnection, connected ? 60000 : 15000);
    
    return () => {
      clearInterval(intervalId);
    };
  }, [socket, connected, reconnectSocket, reconnectAttempts, lastReconnectTime]);
  
  // Handle pull-to-refresh
  const onRefresh = async () => {
    setRefreshing(true);
    try {
      // Check chat service health
      const health = await chatService.healthCheck();
      setServiceStatus(health);
      
      // Try to reconnect socket if needed
      if (!connected) {
        reconnectSocket();
      }
      
      // If we have issues, show a message
      if (!health.healthy || !connected) {
        setSnackbarMessage('Reconnecting to chat service...');
        setSnackbarVisible(true);
      }
    } catch (err) {
      console.error('Error during refresh:', err);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Function to manually fix connection issues
  const handleFixConnection = async () => {
    try {
      Alert.alert('Reconnecting', 'Attempting to reconnect to chat services...');
      
      // Reconnect socket
      if (reconnectSocket) {
        reconnectSocket();
      }
      
      // Check health and apply fixes
      const health = await chatService.healthCheck();
      setServiceStatus(health);
      
      if (!health.healthy) {
        Alert.alert(
          'Connection Issues',
          'Could not establish connection to chat services. Please try again later.',
          [{ text: 'OK' }]
        );
      } else {
        setSnackbarMessage('Connection restored successfully!');
        setSnackbarVisible(true);
      }
    } catch (err) {
      console.error('Failed to fix connection:', err);
      Alert.alert('Error', 'Failed to reconnect. Please try again later.');
    }
  };
  
  return (
    <SafeAreaView 
      style={[
        styles.container, 
        { backgroundColor: isDarkMode ? theme.colors.background : '#f5f7fa' }
      ]}
      edges={['top', 'left', 'right']}
    >
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={isDarkMode ? theme.colors.background : '#f5f7fa'}
      />
      <View style={styles.header}>
        <Text style={[styles.headerTitle, { color: theme.colors.text }]}>Messages</Text>
        {/* Show connection status if there are issues */}
        {(!connected || (serviceStatus && !serviceStatus.healthy)) && (
          <View style={styles.statusContainer}>
            <Text style={[styles.statusText, { color: theme.colors.error }]}>
              {!connected ? 'Connection issues' : 'Service issues'}
            </Text>
            <Button 
              mode="text" 
              compact
              onPress={handleFixConnection}
              style={styles.reconnectButton}
            >
              Reconnect
            </Button>
          </View>
        )}
      </View>
      
      <View style={styles.content}>
        <AdminChatList refreshing={refreshing} onRefresh={onRefresh} />
      </View>
      
      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => setSnackbarVisible(false)}
        duration={3000}
        action={{
          label: 'Close',
          onPress: () => setSnackbarVisible(false),
        }}
      >
        {snackbarMessage}
      </Snackbar>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
  },
  statusText: {
    fontSize: 14,
    fontStyle: 'italic',
  },
  reconnectButton: {
    marginLeft: 8,
  },
});

export default ChatScreen;
