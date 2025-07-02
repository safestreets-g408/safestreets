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
  const isSystemMessage = message.isSystem;
  const isPending = message.isPending;
  const isFailed = message.isFailed;
  
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
        <View style={[
          styles.reportContainer, 
          { backgroundColor: theme.dark ? theme.colors.surfaceVariant : '#f8fafc' }
        ]}>
          <View style={styles.reportHeader}>
            <MaterialCommunityIcons name="clipboard-text" size={18} color={theme.colors.primary} />
            <Text style={[styles.reportTitle, { color: theme.colors.primary }]}>
              Damage Report: {reportData.reportId}
            </Text>
          </View>
          
          <View style={styles.reportContent}>
            <Text style={[styles.reportText, { color: theme.colors.text }]}>
              <Text style={[styles.reportBold, { color: theme.colors.text }]}>Type:</Text> {reportData.damageType}
            </Text>
            <Text style={[styles.reportText, { color: theme.colors.text }]}>
              <Text style={[styles.reportBold, { color: theme.colors.text }]}>Severity:</Text> {reportData.severity}
            </Text>
            <Text style={[styles.reportText, { color: theme.colors.text }]}>
              <Text style={[styles.reportBold, { color: theme.colors.text }]}>Status:</Text> {reportData.status}
            </Text>
            <Text style={[styles.reportText, { color: theme.colors.text }]}>
              <Text style={[styles.reportBold, { color: theme.colors.text }]}>Location:</Text> {reportData.location}
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
      return <Text style={[
        styles.messageText, 
        isOwn ? { color: '#FFFFFF' } : { color: theme.colors.text }
      ]}>No message content</Text>;
    }
    
    const messageText = message.message;
    
    if (isReportMessage(messageText)) {
      return renderReportContent(messageText);
    }
    
    if (isSystemMessage) {
      return <Text style={[
        styles.messageText,
        styles.systemMessageText,
        { color: '#5D4037' }
      ]}>{messageText}</Text>;
    }
    
    return <Text style={[
      styles.messageText,
      isOwn ? { color: '#FFFFFF' } : { color: theme.colors.text }
    ]}>{messageText}</Text>;
  };
  
  return (
    <View style={[
      styles.messageBubble,
      isOwn ? styles.ownMessage : styles.otherMessage,
      isSystemMessage && styles.systemMessage,
      isPending && styles.pendingMessage,
      isFailed && styles.failedMessage,
      isOwn 
        ? { 
            backgroundColor: 
              isFailed ? '#FFCDD2' : 
              isPending ? theme.colors.primary + '90' : 
              theme.colors.primary + 'F0' 
          } 
        : isSystemMessage
          ? { backgroundColor: '#FFF9C4', borderColor: '#FFE082' }  
          : { 
              backgroundColor: theme.dark ? theme.colors.surfaceVariant : '#FFFFFF', 
              borderColor: theme.colors.border 
            }
    ]}>
      {isSystemMessage && (
        <View style={styles.systemMessageIconContainer}>
          <MaterialCommunityIcons name="information" size={16} color="#FB8C00" />
        </View>
      )}
      {renderMessageContent(message)}
      <View style={styles.messageFooter}>
        {isPending && (
          <MaterialCommunityIcons name="clock-outline" size={12} color={isOwn ? '#FFFFFF90' : theme.colors.placeholder} style={styles.statusIcon} />
        )}
        {isFailed && (
          <MaterialCommunityIcons name="alert-circle-outline" size={12} color="#F44336" style={styles.statusIcon} />
        )}
        <Text style={[
          styles.timeText,
          isOwn ? { color: '#FFFFFF90' } : 
          isSystemMessage ? { color: '#FB8C00' } : { color: theme.colors.textSecondary }
        ]}>
          {formatMessageTime(message.createdAt)}
        </Text>
      </View>
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
const ChatWindow = ({ chatRoomId, receiverName, adminId }) => {
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
      if (!adminId) {
        console.error('Cannot load messages: missing adminId');
        setError('Cannot load messages without admin information');
        setLoading(false);
        return;
      }

      // Skip API call for mock chat rooms
      if (chatRoomId && (chatRoomId.includes('_mock') || chatRoomId.includes('_fallback'))) {
        console.log('Using mock chat room - skipping message loading from API');
        if (pageNum === 1) setLoading(false);
        else setLoadingMore(false);
        return;
      }

      if (pageNum === 1) setLoading(true);
      else setLoadingMore(true);
      
      console.log(`Loading messages for admin ID: ${adminId}, page: ${pageNum}`);
      const response = await chatService.getChatMessages(adminId, pageNum);
      
      if (!response || !response.messages) {
        console.error('Invalid response format for messages:', response);
        setError('Received invalid data from server');
        return;
      }
      
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
  }, [adminId]);
  
  const handleSendMessage = async () => {
    if (!newMessage.trim() || sending) return;
    
    setSending(true);
    const messageText = newMessage.trim();
    setNewMessage('');
    
    try {
      // For mock chat rooms, add message locally without sending to API
      if (isMockChatRoom()) {
        console.log('Mock chat room - adding message locally without API call');
        
        // Add message locally
        const newMsg = {
          _id: `local-${Date.now()}`,
          message: messageText,
          senderId: fieldWorker?._id || 'local-user',
          senderModel: 'FieldWorker',
          createdAt: new Date().toISOString(),
          isLocal: true
        };
        
        setMessages(prev => [...prev, newMsg]);
        setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
        
        // Add a simulated response after a short delay
        setTimeout(() => {
          const simulatedResponse = {
            _id: `simulated-${Date.now()}`,
            message: "Your message has been queued and will be delivered when the system is back online. Thank you for your patience.",
            senderId: adminId,
            senderModel: 'Admin',
            createdAt: new Date().toISOString(),
            isSystem: true
          };
          setMessages(prev => [...prev, simulatedResponse]);
          setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
        }, 1000);
      } else {
        // Normal flow - send message to API
        console.log(`Sending message to admin ID: ${adminId}`);
        
        // First, add a temporary message to the UI with a pending status
        const tempMessageId = `temp-${Date.now()}`;
        const tempMessage = {
          _id: tempMessageId,
          message: messageText,
          senderId: fieldWorker?._id || 'local-user',
          senderModel: 'FieldWorker',
          createdAt: new Date().toISOString(),
          isPending: true
        };
        
        // Add the temporary message to the UI
        setMessages(prev => [...prev, tempMessage]);
        setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
        
        try {
          // Send the message to the server
          const response = await chatService.sendMessage(adminId, messageText);
          console.log('Message sent successfully:', response);
          
          // If successful, update the temporary message with the real ID from the server
          if (response && response._id) {
            setMessages(prev => 
              prev.map(msg => 
                msg._id === tempMessageId 
                  ? { ...response, isPending: false } 
                  : msg
              )
            );
            
            // For redundancy, manually emit a socket event with the message
            // This acts as a fallback in case the server's emit didn't work
            if (socket && chatRoomId) {
              console.log('Manually emitting message via socket as backup');
              socket.emit('manual_message', {
                ...response,
                adminId: adminId,
                chatRoomId: chatRoomId
              });
            }
          }
        } catch (err) {
          // If there's an error, mark the message as failed
          console.error('Error sending message to server:', err);
          setMessages(prev => 
            prev.map(msg => 
              msg._id === tempMessageId 
                ? { ...msg, isPending: false, isFailed: true } 
                : msg
            )
          );
          // Show error but don't reset the message text since it's already in the UI
          setError('Message sent locally but failed to reach the server. Tap to retry.');
        }
      }
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
  
  const markMessagesRead = useCallback(async () => {
    try {
      // Skip for mock chat rooms
      if (isMockChatRoom()) {
        console.log('Mock chat room - skipping markMessagesAsRead');
        return;
      }
      
      // Mark messages as read both on socket and via API
      markAsRead();
      if (adminId) {
        await chatService.markMessagesAsRead(adminId);
      }
    } catch (err) {
      console.error('Error marking messages as read:', err);
    }
  }, [adminId, markAsRead, isMockChatRoom]);

  const isMockChatRoom = useCallback(() => {
    return chatRoomId && (chatRoomId.includes('_mock') || chatRoomId.includes('_fallback'));
  }, [chatRoomId]);

  useEffect(() => {
    if (!chatRoomId) return;
    
    if (isMockChatRoom()) {
      console.log('Using mock/fallback chat room:', chatRoomId);
      // For mock chat rooms, provide a nicer user experience
      setMessages([{
        _id: 'welcome-message',
        senderId: adminId,
        senderModel: 'Admin',
        message: 'Welcome to the support chat! This is a temporary chat session. The system is experiencing some technical difficulties, but you can still send messages that will be delivered when connectivity is restored.',
        createdAt: new Date().toISOString(),
        isSystem: true
      }]);
      setLoading(false);
    } else {
      // Normal flow for real chat rooms
      joinChat(adminId, chatRoomId);
      loadMessages();
      markMessagesRead();
    }
    
    return () => {
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    };
  }, [chatRoomId, joinChat, loadMessages, markMessagesRead, adminId, isMockChatRoom]);
  
  useEffect(() => {
    if (!socket) return;
    
    const handleNewMessage = (message) => {
      console.log('Received new_message event:', message);
      console.log('Current chatRoomId:', chatRoomId);
      
      // Check if the message is for this chat room
      const isForThisChat = message.chatId === chatRoomId;
      
      // If not a direct match, log details to debug
      if (!isForThisChat) {
        console.log('Message not matching this chatRoomId. Details:');
        console.log('- Expected chatRoomId:', chatRoomId);
        console.log('- Message chatId:', message.chatId);
        console.log('- Message sender:', message.senderModel, message.senderId);
        console.log('- Current user:', 'FieldWorker', fieldWorker?._id);
      }
      
      if (isForThisChat) {
        console.log('Adding message to chat display');
        setMessages(prev => [...prev, message]);
        setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
        if (message.senderModel !== 'FieldWorker' || message.senderId !== fieldWorker._id) {
          markMessagesRead();
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
  }, [socket, chatRoomId, fieldWorker, markMessagesRead]);
  
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
          style={[
            styles.input, 
            { 
              backgroundColor: theme.dark ? theme.colors.surfaceVariant : '#FFFFFF',
              borderColor: newMessage.length > 0 ? theme.colors.primary : theme.colors.border,
              color: theme.colors.text
            }
          ]}
          placeholder="Type your message..."
          placeholderTextColor={theme.colors.placeholder}
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
  systemMessage: {
    alignSelf: 'center',
    borderRadius: 12,
    maxWidth: '90%',
    borderWidth: 1,
  },
  systemMessageIconContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  systemMessageText: {
    fontStyle: 'italic',
  },
  pendingMessage: {
    opacity: 0.8,
  },
  failedMessage: {
    borderWidth: 1,
    borderColor: '#E53935',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  messageFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginTop: 4,
  },
  statusIcon: {
    marginRight: 4,
  },
  timeText: {
    fontSize: 11,
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