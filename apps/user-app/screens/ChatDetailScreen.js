import React, { useEffect, useState } from 'react';
import { StyleSheet, View, StatusBar, Text } from 'react-native';
import { useTheme, ActivityIndicator } from 'react-native-paper';
import { useThemeContext } from '../context/ThemeContext';
import ChatWindow from '../components/chat/ChatWindow';
import { SafeAreaView } from 'react-native-safe-area-context';
import { chatService } from '../utils/chatAPI';

const ChatDetailScreen = ({ route, navigation }) => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const { adminName } = route.params;
  const [chatRoomId, setChatRoomId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    navigation.setOptions({
      title: adminName || 'Admin Support',
    });

    const fetchChatRoom = async () => {
      try {
        setLoading(true);
        const roomData = await chatService.getChatRoom();
        setChatRoomId(roomData.roomId);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching chat room:', err);
        setError(err.message || 'Failed to load chat room');
        setLoading(false);
      }
    };

    fetchChatRoom();
  }, [adminName, navigation]);

  if (loading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.errorContainer, { backgroundColor: theme.colors.background }]}>
        <Text style={{ color: theme.colors.error }}>{error}</Text>
      </View>
    );
  }

  return (
    <SafeAreaView
      style={[
        styles.container,
        { backgroundColor: isDarkMode ? theme.colors.background : '#f5f7fa' }
      ]}
      edges={['left', 'right', 'bottom']}
    >
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={isDarkMode ? theme.colors.background : '#f5f7fa'}
      />
      <View style={styles.content}>
        <ChatWindow 
          chatRoomId={chatRoomId} 
          receiverName={adminName} 
        />
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  }
});

export default ChatDetailScreen;
