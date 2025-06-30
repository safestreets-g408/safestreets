import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Image,
  Animated,
  Easing
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useTheme } from 'react-native-paper';
import { format, isToday, isYesterday } from 'date-fns';
import { chatService } from '../../utils/chatAPI';
import { useSocket } from '../../context/SocketContext';
import { useAuth } from '../../context/AuthContext';

// Message bubble component
const MessageBubble = ({ message, isOwn }) => {
  const theme = useTheme();
  
  const formatMessageTime = (timestamp) => {
    const date = new Date(timestamp);
    if (isToday(date)) {
      return format(date, 'HH:mm');
    } else if (isYesterday(date)) {
      return `Yesterday ${format(date, 'HH:mm')}`;
    } else {
      return format(date, 'MMM dd, HH:mm');
    }
  };
  
  const isReportMessage = (messageText) => {
    return typeof messageText === 'string' && messageText.startsWith('__REPORT_JSON__:');
  };
  
  const renderReportContent = (messageText) => {
    try {
      const jsonString = messageText.replace('__REPORT_JSON__:', '');
      const reportData = JSON.parse(jsonString);
      
      return (
        <View style={styles.reportContainer}>
          <View style={styles.reportHeader}>
            <MaterialCommunityIcons name="clipboard-text" size={18} color={theme.colors.primary} />
            <Text style={[styles.reportTitle, { color: theme.colors.primary }]}>
              Damage Report: {reportData.reportId}
            </Text>
          </View>
          
          <View style={styles.reportContent}>
            <Text style={styles.reportText}>
              <Text style={styles.reportBold}>Type:</Text> {reportData.damageType}
            </Text>
            <Text style={styles.reportText}>
              <Text style={styles.reportBold}>Severity:</Text> {reportData.severity}
            </Text>
            <Text style={styles.reportText}>
              <Text style={styles.reportBold}>Status:</Text> {reportData.status}
            </Text>
            <Text style={styles.reportText}>
              <Text style={styles.reportBold}>Location:</Text> {reportData.location}
            </Text>
          </View>
        </View>
      );
    } catch (err) {
      console.error('Error parsing report message:', err);
      return <Text style={styles.messageText}>Could not display report data</Text>;
    }
  };
  
  const renderMessageContent = (message) => {
    if (!message || !message.message) {
      return <Text style={styles.messageText}>No message content</Text>;
    }
    
    const messageText = message.message;
    
    if (isReportMessage(messageText)) {
      return renderReportContent(messageText);
    }
    
    return <Text style={styles.messageText}>{messageText}</Text>;
  };
  
  return (
    <View style={[
      styles.messageBubble,
      isOwn ? styles.ownMessage : styles.otherMessage,
      isOwn 
        ? { backgroundColor: theme.colors.primary + 'F0' } 
        : { backgroundColor: '#FFFFFF', borderColor: '#E5E7EB' }
    ]}>
      {renderMessageContent(message)}
      <Text style={[
        styles.timeText,
        isOwn ? { color: '#FFFFFF90' } : { color: '#9CA3AF' }
      ]}>
        {formatMessageTime(message.createdAt)}
      </Text>
    </View>
  );
};

// Typing indicator component
const TypingIndicator = () => {
  const theme = useTheme();
  const dotAnimation1 = useRef(new Animated.Value(0)).current;
  const dotAnimation2 = useRef(new Animated.Value(0)).current;
  const dotAnimation3 = useRef(new Animated.Value(0)).current;
  
  useEffect(() => {
    const animateDots = () => {
      Animated.sequence([
        Animated.timing(dotAnimation1, { toValue: 1, duration: 300, useNativeDriver: true, easing: Easing.ease }),
        Animated.timing(dotAnimation2, { toValue: 1, duration: 300, useNativeDriver: true, easing: Easing.ease }),
        Animated.timing(dotAnimation3, { toValue: 1, duration: 300, useNativeDriver: true, easing: Easing.ease }),
        Animated.timing(dotAnimation1, { toValue: 0, duration: 300, useNativeDriver: true, easing: Easing.ease }),
        Animated.timing(dotAnimation2, { toValue: 0, duration: 300, useNativeDriver: true, easing: Easing.ease }),
        Animated.timing(dotAnimation3, { toValue: 0, duration: 300, useNativeDriver: true, easing: Easing.ease })
      ]).start(() => animateDots());
    };
    
    animateDots();
    
    return () => {
      dotAnimation1.stopAnimation();
      dotAnimation2.stopAnimation();
      dotAnimation3.stopAnimation();
    };
  }, []);
  
  return (
    <View style={styles.typingIndicator}>
      <Animated.View style={[styles.typingDot, { backgroundColor: theme.colors.primary, transform: [{ translateY: dotAnimation1.interpolate({ inputRange: [0, 1], outputRange: [0, -5] }) }] }]} />
      <Animated.View style={[styles.typingDot, { backgroundColor: theme.colors.primary, transform: [{ translateY: dotAnimation2.interpolate({ inputRange: [0, 1], outputRange: [0, -5] }) }] }]} />
      <Animated.View style={[styles.typingDot, { backgroundColor: theme.colors.primary, transform: [{ translateY: dotAnimation3.interpolate({ inputRange: [0, 1], outputRange: [0, -5] }) }] }]} />
    </View>
  );
};

// Main ChatWindow Component
const ChatWindow = ({ chatRoomId, receiverName }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [typingUsers, setTypingUsers] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [page, setPage] = useState(1);
  const [loadingMore, setLoadingMore] = useState(false);
  
  const flatListRef = useRef(null);
  const typingTimeoutRef = useRef(null);
  
  const theme = useTheme();
  const { socket, joinChat, markAsRead, startTyping, stopTyping } = useSocket();
  const { fieldWorker } = useAuth();
  
  const loadMessages = useCallback(async (pageNum = 1, append = false) => {
    try {
      if (pageNum === 1) setLoading(true);
      else setLoadingMore(true);
      
      const response = await chatService.getChatMessages(pageNum);
      
      if (append) {
        setMessages(prev => [...response.messages, ...prev]);
      } else {
        setMessages(response.messages);
        setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
      }
      
      setHasMore(response.hasMore);
      setPage(pageNum);
    } catch (err) {
      console.error('Error loading messages:', err);
      setError('Failed to load messages');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, []);
  
  const handleSendMessage = async () => {
    if (!newMessage.trim() || sending) return;
    
    setSending(true);
    const messageText = newMessage.trim();
    setNewMessage('');
    
    try {
      await chatService.sendMessage(messageText);
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message');
      setNewMessage(messageText);
    } finally {
      setSending(false);
    }
  };
  
  const handleTyping = () => {
    startTyping();
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => stopTyping(), 3000);
  };
  
  useEffect(() => {
    if (chatRoomId) {
      joinChat();
      loadMessages();
      markAsRead();
    }
    return () => {
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    };
  }, [chatRoomId, joinChat, loadMessages, markAsRead]);
  
  useEffect(() => {
    if (!socket) return;
    
    const handleNewMessage = (message) => {
      if (message.chatId === chatRoomId) {
        setMessages(prev => [...prev, message]);
        setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
        if (message.senderModel !== 'FieldWorker' || message.senderId !== fieldWorker._id) {
          markAsRead();
        }
      }
    };
    
    const handleUserTyping = (data) => {
      if (data.tenantId === fieldWorker.tenant) {
        setTypingUsers(prev => {
          if (data.isTyping) {
            return prev.includes(data.userName) ? prev : [...prev, data.userName];
          } else {
            return prev.filter(name => name !== data.userName);
          }
        });
      }
    };
    
    socket.on('new_message', handleNewMessage);
    socket.on('admin_typing', handleUserTyping);
    
    return () => {
      socket.off('new_message', handleNewMessage);
      socket.off('admin_typing', handleUserTyping);
    };
  }, [socket, chatRoomId, fieldWorker, markAsRead]);
  
  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading conversation...</Text>
      </View>
    );
  }
  
  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : null}
      style={styles.container}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <FlatList
        ref={flatListRef}
        data={messages}
        keyExtractor={(item) => item._id}
        onEndReached={() => hasMore && !loadingMore && loadMessages(page + 1, true)}
        onEndReachedThreshold={0.2}
        inverted={false}
        ListHeaderComponent={loadingMore ? (
          <View style={styles.loadingMoreContainer}>
            <ActivityIndicator size="small" color={theme.colors.primary} />
          </View>
        ) : null}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <MaterialCommunityIcons name="message-text-outline" size={60} color={theme.colors.disabled} />
            <Text style={styles.emptyText}>No messages yet</Text>
            <Text style={styles.emptySubText}>Start the conversation with {receiverName}</Text>
          </View>
        }
        renderItem={({ item }) => (
          <MessageBubble 
            message={item} 
            isOwn={item.senderModel === 'FieldWorker' && item.senderId === fieldWorker._id} 
          />
        )}
        contentContainerStyle={styles.messagesList}
      />
      
      {typingUsers.length > 0 && (
        <View style={styles.typingContainer}>
          <TypingIndicator />
          <Text style={styles.typingText}>
            {typingUsers.join(', ')} {typingUsers.length === 1 ? 'is' : 'are'} typing...
          </Text>
        </View>
      )}
      
      {error && (
        <View style={[styles.errorContainer, { backgroundColor: theme.colors.error + '20' }]}>
          <Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>
          <TouchableOpacity onPress={() => setError(null)}>
            <MaterialCommunityIcons name="close" size={18} color={theme.colors.error} />
          </TouchableOpacity>
        </View>
      )}
      
      <View style={styles.inputContainer}>
        <TextInput
          style={[styles.input, { backgroundColor: '#FFFFFF', borderColor: newMessage.length > 0 ? theme.colors.primary : '#E5E7EB' }]}
          placeholder="Type your message..."
          value={newMessage}
          onChangeText={(text) => {
            setNewMessage(text);
            handleTyping();
          }}
          multiline
          maxLength={1000}
          editable={!sending}
        />
        <TouchableOpacity
          style={[styles.sendButton, !newMessage.trim() || sending ? { backgroundColor: theme.colors.disabled } : { backgroundColor: theme.colors.primary }]}
          onPress={handleSendMessage}
          disabled={!newMessage.trim() || sending}
        >
          {sending ? (
            <ActivityIndicator size="small" color="#FFFFFF" />
          ) : (
            <MaterialCommunityIcons name="send" size={20} color="#FFFFFF" />
          )}
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#6B7280',
  },
  messagesList: {
    padding: 16,
    flexGrow: 1,
  },
  loadingMoreContainer: {
    padding: 10,
    alignItems: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6B7280',
    marginTop: 16,
  },
  emptySubText: {
    fontSize: 14,
    color: '#9CA3AF',
    marginTop: 8,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
    marginVertical: 4,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 1,
  },
  ownMessage: {
    alignSelf: 'flex-end',
    borderBottomRightRadius: 4,
  },
  otherMessage: {
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  timeText: {
    fontSize: 11,
    alignSelf: 'flex-end',
    marginTop: 4,
  },
  typingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 8,
  },
  typingDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginHorizontal: 2,
  },
  typingText: {
    fontSize: 12,
    color: '#6B7280',
    fontStyle: 'italic',
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 16,
    marginVertical: 8,
    padding: 10,
    borderRadius: 8,
  },
  errorText: {
    fontSize: 14,
    flex: 1,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
  },
  input: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 24,
    borderWidth: 1,
    fontSize: 16,
    maxHeight: 100,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  reportContainer: {
    borderWidth: 1,
    borderRadius: 8,
    overflow: 'hidden',
  },
  reportHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    gap: 8,
  },
  reportTitle: {
    fontSize: 14,
    fontWeight: '600',
  },
  reportContent: {
    padding: 8,
    backgroundColor: 'transparent',
  },
  reportText: {
    fontSize: 14,
    color: '#374151',
    marginBottom: 4,
  },
  reportBold: {
    fontWeight: '600',
  },
});

export default ChatWindow;