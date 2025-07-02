import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Image,
  TextInput,
  ActivityIndicator
} from 'react-native';
import { useTheme } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { format } from 'date-fns';
import { chatService } from '../../utils/chatAPI';
import { LinearGradient } from 'expo-linear-gradient';
import { checkNetworkConnectivity, showNetworkError } from '../../utils/auth';
import { useThemeContext } from '../../context/ThemeContext';

const AdminChatList = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [admins, setAdmins] = useState([]);
  const [filteredAdmins, setFilteredAdmins] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const navigation = useNavigation();
  
  // Function to fetch admin list with proper error handling
  const loadAdmins = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Check network connectivity first
      const isConnected = await checkNetworkConnectivity();
      if (!isConnected) {
        console.error('Network connectivity issue detected');
        setError('Network connection error. Please check your internet connection.');
        setLoading(false);
        return;
      }
      
      console.log('Fetching admin chat list...');
      const response = await chatService.getAdminChatList();
      console.log('Admin list response:', response);
      
      if (!response || !Array.isArray(response)) {
        console.error('Invalid response format:', response);
        setError('Received invalid data from server');
        setLoading(false);
        return;
      }
      
      setAdmins(response);
      setFilteredAdmins(response);
      console.log(`Loaded ${response.length} admins successfully`);
    } catch (err) {
      console.error('Error loading admins:', err);
      
      if (err.response) {
        // Server responded with an error status code
        const statusCode = err.response.status;
        const errorMessage = err.response.data?.message || 'Unknown server error';
        console.error(`Server error ${statusCode}: ${errorMessage}`);
        
        if (statusCode === 401) {
          setError('Authentication error. Please log in again.');
        } else {
          setError(`Failed to load admin list: ${errorMessage}`);
        }
      } else if (err.request) {
        // Request was made but no response received
        console.error('No response received from server');
        setError('Server not responding. Please try again later.');
      } else {
        // Error setting up the request
        setError('Failed to load admin list. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch available admins
  useEffect(() => {
    loadAdmins();
  }, []);
  
  // Filter admins based on search query
  useEffect(() => {
    if (!admins.length) return;
    
    if (!searchQuery.trim()) {
      setFilteredAdmins(admins);
      return;
    }
    
    const query = searchQuery.toLowerCase();
    const filtered = admins.filter(admin => 
      admin.name.toLowerCase().includes(query) ||
      admin.role.toLowerCase().includes(query)
    );
    
    setFilteredAdmins(filtered);
  }, [searchQuery, admins]);
  
  // Format last message time
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return format(date, 'HH:mm');
  };
  
  // Navigate to chat with admin
  const openChat = (admin) => {
    navigation.navigate('ChatDetail', {
      adminId: admin._id,
      adminName: admin.name
    });
  };
  
  // Render admin list item
  const renderAdminItem = ({ item }) => {
    const unreadCount = item.unreadCount || 0;
    
    return (
      <TouchableOpacity
        style={[
          styles.adminItem,
          { 
            borderBottomColor: theme.colors.border,
            backgroundColor: theme.colors.surface
          }
        ]}
        onPress={() => openChat(item)}
      >
        <View style={styles.avatarContainer}>
          {item.profileImage ? (
            <Image source={{ uri: item.profileImage }} style={styles.avatar} />
          ) : (
            <View style={[styles.avatarFallback, { 
              backgroundColor: item.role === 'super-admin' ? '#8B5CF6' : '#3B82F6' 
            }]}>
              <Text style={styles.avatarText}>
                {item.name.charAt(0).toUpperCase()}
              </Text>
            </View>
          )}
          {item.isOnline && (
            <View style={styles.onlineIndicator} />
          )}
        </View>
        
        <View style={styles.adminInfo}>
          <View style={styles.adminNameRow}>
            <Text style={[styles.adminName, { color: theme.colors.text }]}>
              {item.name}
            </Text>
            {item.lastMessage?.timestamp && (
              <Text style={[styles.timestamp, { color: theme.colors.textSecondary }]}>
                {formatTime(item.lastMessage.timestamp)}
              </Text>
            )}
          </View>
          
          <View style={styles.adminSubRow}>
            <View style={styles.roleContainer}>
              <Text style={[
                styles.roleBadge,
                { 
                  backgroundColor: item.role === 'super-admin' ? '#8B5CF680' : '#3B82F680',
                  color: '#FFFFFF'
                }
              ]}>
                {item.role === 'super-admin' ? 'SUPER' : 'ADMIN'}
              </Text>
            </View>
            
            {item.lastMessage?.message ? (
              <Text 
                style={[styles.lastMessage, { color: theme.colors.textSecondary }]}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                {item.lastMessage.message}
              </Text>
            ) : (
              <Text style={[styles.noMessages, { color: theme.colors.placeholder }]}>No messages yet</Text>
            )}
            
            {unreadCount > 0 && (
              <View style={[styles.unreadBadge, { backgroundColor: theme.colors.primary }]}>
                <Text style={styles.unreadText}>
                  {unreadCount > 99 ? '99+' : unreadCount}
                </Text>
              </View>
            )}
          </View>
        </View>
      </TouchableOpacity>
    );
  };
  
  // Empty state
  const renderEmptyState = () => {
    if (loading) return null;
    
    return (
      <View style={styles.emptyContainer}>
        <MaterialCommunityIcons
          name="account-supervisor-outline"
          size={60}
          color={theme.colors.disabled}
        />
        {searchQuery ? (
          <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>
            No admins found matching "{searchQuery}"
          </Text>
        ) : (
          <>
            <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>No admins available</Text>
            <Text style={[styles.emptySubtext, { color: theme.colors.placeholder }]}>
              There are no admins available to chat with at the moment
            </Text>
          </>
        )}
      </View>
    );
  };
  
  // Loading state
  if (loading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>Loading admins...</Text>
      </View>
    );
  }
  
  // Error state
  if (error) {
    return (
      <View style={[styles.errorContainer, { backgroundColor: theme.colors.background }]}>
        <MaterialCommunityIcons name="alert-circle" size={60} color={theme.colors.error} />
        <Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>
        
        <View style={styles.errorDetailsContainer}>
          <Text style={[styles.errorDetailsText, { color: theme.colors.textSecondary }]}>
            If this problem persists, please try:
          </Text>
          <Text style={[styles.errorDetailItem, { color: theme.colors.textSecondary }]}>• Checking your internet connection</Text>
          <Text style={[styles.errorDetailItem, { color: theme.colors.textSecondary }]}>• Making sure the server is running</Text>
          <Text style={[styles.errorDetailItem, { color: theme.colors.textSecondary }]}>• Logging out and logging back in</Text>
        </View>
        
        <TouchableOpacity 
          style={[styles.retryButton, { backgroundColor: theme.colors.primary }]}
          onPress={loadAdmins}
        >
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }
  
  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {/* Search bar */}
      <View style={[styles.searchContainer, { 
        backgroundColor: theme.colors.surface, 
        borderColor: theme.colors.border 
      }]}>
        <MaterialCommunityIcons 
          name="magnify" 
          size={20} 
          color={theme.colors.placeholder} 
          style={styles.searchIcon} 
        />
        <TextInput
          style={[styles.searchInput, { color: theme.colors.text }]}
          placeholder="Search admins..."
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholderTextColor={theme.colors.placeholder}
          clearButtonMode="while-editing"
        />
      </View>
      
      {/* Admin list */}
      <FlatList
        data={filteredAdmins}
        renderItem={renderAdminItem}
        keyExtractor={(item) => item._id}
        contentContainerStyle={styles.listContainer}
        ListEmptyComponent={renderEmptyState}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 12,
    marginHorizontal: 16,
    marginVertical: 12,
    paddingHorizontal: 12,
    height: 44,
    borderWidth: 1,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    height: '100%',
  },
  listContainer: {
    flexGrow: 1,
    paddingBottom: 20,
  },
  adminItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
  },
  avatarContainer: {
    position: 'relative',
    marginRight: 16,
  },
  avatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#E5E7EB',
  },
  avatarFallback: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: 'bold',
  },
  onlineIndicator: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: '#10B981',
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  adminInfo: {
    flex: 1,
  },
  adminNameRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  adminName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  timestamp: {
    fontSize: 12,
    color: '#6B7280',
  },
  adminSubRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  roleContainer: {
    marginRight: 8,
  },
  roleBadge: {
    fontSize: 10,
    fontWeight: 'bold',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    overflow: 'hidden',
  },
  lastMessage: {
    flex: 1,
    fontSize: 14,
    color: '#6B7280',
    marginRight: 8,
  },
  noMessages: {
    flex: 1,
    fontSize: 14,
    color: '#9CA3AF',
    fontStyle: 'italic',
  },
  unreadBadge: {
    minWidth: 20,
    height: 20,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 6,
  },
  unreadText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: 'bold',
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
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
    padding: 20,
  },
  errorText: {
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 16,
  },
  errorDetailsContainer: {
    marginBottom: 20,
    width: '100%',
    paddingHorizontal: 20,
  },
  errorDetailsText: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 8,
  },
  errorDetailItem: {
    fontSize: 14,
    color: '#6B7280',
    marginVertical: 3,
    marginLeft: 10,
  },
  retryButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    marginTop: 80,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6B7280',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    marginTop: 8,
  },
});

export default AdminChatList;
