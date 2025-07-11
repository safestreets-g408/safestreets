import React, { useEffect } from 'react';
import { StyleSheet, View, StatusBar, Text } from 'react-native';
import { useTheme } from 'react-native-paper';
import { useThemeContext } from '../context/ThemeContext';
import AdminChatList from '../components/chat/AdminChatList';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSocket } from '../context/SocketContext';

const ChatScreen = () => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const { socket, connected } = useSocket();
  
  useEffect(() => {
    // Ensure socket connection is active when entering chat screen
    if (socket && !connected) {
      console.log('ChatScreen: Socket not connected, attempting to reconnect');
      socket.connect();
    } else if (socket && connected) {
      console.log('ChatScreen: Socket connection verified');
      socket.emit('ping_server'); // Test the connection
    }
    
    return () => {
      // No need to disconnect on leaving, keep the socket alive for notifications
    };
  }, [socket, connected]);
  
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
      </View>
      <View style={styles.content}>
        <AdminChatList />
      </View>
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
  }
});

export default ChatScreen;
