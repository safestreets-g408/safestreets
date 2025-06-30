import React from 'react';
import { StyleSheet, View, StatusBar, Text } from 'react-native';
import { useTheme } from 'react-native-paper';
import { useThemeContext } from '../context/ThemeContext';
import AdminChatList from '../components/chat/AdminChatList';
import { SafeAreaView } from 'react-native-safe-area-context';

const ChatScreen = () => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  
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
        <Text style={styles.headerTitle}>Messages</Text>
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
    color: '#1F2937',
  }
});

export default ChatScreen;
