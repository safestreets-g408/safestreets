import React, { useEffect, useState } from 'react';
import { StyleSheet, View, StatusBar, Text, TouchableOpacity } from 'react-native';
import { useTheme, ActivityIndicator } from 'react-native-paper';
import { useThemeContext } from '../context/ThemeContext';
import ChatWindow from '../components/chat/ChatWindow';
import { SafeAreaView } from 'react-native-safe-area-context';
import { chatService } from '../utils/chatAPI';

const ChatDetailScreen = ({ route, navigation }) => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  
  // Safely extract route params with validation
  const { adminId, adminName } = route.params || {};
  
  const [chatRoomId, setChatRoomId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  
  // Validate required params
  useEffect(() => {
    if (!adminId) {
      console.error('ChatDetailScreen opened without adminId');
      setError('Missing admin information. Please go back and try again.');
      setLoading(false);
    }
  }, [adminId]);

  useEffect(() => {
    navigation.setOptions({
      title: adminName || 'Admin Support',
    });

    // Skip fetch if we don't have an adminId
    if (!adminId) return;

    const fetchChatRoom = async (currentRetry = 0) => {
      try {
        setLoading(true);
        console.log(`Fetching chat room for admin ID: ${adminId} (Attempt: ${currentRetry + 1})`);
        const roomData = await chatService.getChatRoom(adminId);
        console.log('Chat room data received:', roomData);
        
        // Check if this is a fallback/temporary chat room
        if (roomData.roomId && (
          roomData.roomId.includes('_temporary') ||
          roomData.roomId.includes('_error') ||
          roomData.isErrorFallback
        )) {
          console.warn('Received fallback/temporary chat room:', roomData);
          setChatRoomId(roomData.roomId);
          
          // Don't reset the retryCount, but don't increment it either
          // This allows a retry attempt after a short delay
          
          // Fetch chat service health in background
          chatService.healthCheck().catch(healthErr => {
            console.error('Health check error:', healthErr);
          });
        } else {
          // Normal chat room received
          setChatRoomId(roomData.roomId);
          setLoading(false);
          // Reset retry count on success
          setRetryCount(0);
        }
      } catch (err) {
        console.error('Error fetching chat room:', err);
        
        // Enhanced error logging for Axios errors
        if (err.isAxiosError) {
          console.error('Axios error details:');
          console.error('Request URL:', err.config?.url);
          console.error('Request method:', err.config?.method);
          console.error('Request data:', err.config?.data);
          
          if (err.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error('Response status:', err.response.status);
            console.error('Response data:', err.response.data);
            console.error('Response headers:', err.response.headers);
            
            // Handle specific error cases
            if (err.response.status === 400 && 
                (err.response.data?.message?.includes('recipient type') || 
                 err.response.data?.error?.includes('recipient type'))) {
              console.error('Invalid recipient type error detected');
              setError('Server configuration error. Please contact support.');
              
              // Log detailed information for debugging
              console.error('This is likely due to a mismatch between API expectations and what we\'re sending.');
              console.error('Request payload:', err.config?.data);
            } 
            // Handle 500 server errors
            else if (err.response.status === 500) {
              console.error('Server error 500 detected');
              setError('The server encountered an error. Please try again in a few moments.');
              
              // Log details for debugging
              console.error('Server error details:', err.response.data);
            } else {
              setError(`Error ${err.response.status}: ${err.response.data?.message || 'Server error'}`);
            }
          } else if (err.request) {
            // The request was made but no response was received
            console.error('No response received');
            setError('Server not responding. Please check your connection.');
          } else {
            // Something happened in setting up the request
            console.error('Request setup error:', err.message);
            setError(`Request error: ${err.message}`);
          }
        } else {
          setError(err.message || 'Failed to load chat room');
        }
        
        // For 500 errors, implement automatic retry with exponential backoff
        const isServer500Error = err.isAxiosError && 
                                err.response && 
                                err.response.status === 500;
                                
        const MAX_RETRIES = 3;
        if (isServer500Error && currentRetry < MAX_RETRIES) {
          const nextRetry = currentRetry + 1;
          setRetryCount(nextRetry);
          
          // Calculate exponential backoff wait time (1s, 2s, 4s, etc.)
          const waitTime = Math.min(1000 * Math.pow(2, currentRetry), 8000);
          
          console.log(`Scheduling retry ${nextRetry}/${MAX_RETRIES} in ${waitTime}ms for 500 error`);
          
          setTimeout(() => {
            console.log(`Executing retry ${nextRetry}/${MAX_RETRIES} for 500 error`);
            fetchChatRoom(nextRetry);
          }, waitTime);
        } else {
          // If we've exhausted retries or it's not a 500 error, show the error
          setLoading(false);
        }
      }
    };

    fetchChatRoom();
  }, [adminId, adminName, navigation]);

  if (loading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        {retryCount > 0 && (
          <Text style={[styles.retryText, { color: theme.colors.textSecondary }]}>
            Connection issue. Retrying... ({retryCount}/3)
          </Text>
        )}
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.errorContainer, { backgroundColor: theme.colors.background }]}>
        <Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>
        
        {/* Show more details for specific errors */}
        {error.includes('invalid recipient type') && (
          <Text style={[styles.errorDetails, { color: theme.colors.textSecondary }]}>
            There was a configuration issue connecting to the chat server.
          </Text>
        )}
        
        <TouchableOpacity 
          style={[styles.retryButton, { backgroundColor: theme.colors.primary }]}
          onPress={() => {
            setError(null);
            setLoading(true);
            
            // Add a slight delay before retrying to ensure any pending operations are cleared
            setTimeout(() => {
              const fetchChatRoom = async () => {
                try {
                  console.log('Retrying to fetch chat room for admin ID:', adminId);
                  const roomData = await chatService.getChatRoom(adminId);
                  console.log('Chat room data received on retry:', roomData);
                  setChatRoomId(roomData.roomId);
                  setLoading(false);
                } catch (err) {
                  console.error('Error on retry:', err);
                  setError('Failed to connect after retry. Please try again later.');
                  setLoading(false);
                }
              };
              
              fetchChatRoom();
            }, 500);
          }}
        >
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
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
          adminId={adminId}
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
  retryText: {
    marginTop: 12,
    fontSize: 14,
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 12,
  },
  errorDetails: {
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 20,
  },
  retryButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  retryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  }
});

export default ChatDetailScreen;
