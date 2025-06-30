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

const AdminChatList = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [admins, setAdmins] = useState([]);
  const [filteredAdmins, setFilteredAdmins] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  
  const theme = useTheme();
  const navigation = useNavigation();
  
  // Fetch available admins
  useEffect(() => {
    const loadAdmins = async () => {
      try {
        setLoading(true);
        const response = await chatService.getAdminChatList();
        setAdmins(response);
        setFilteredAdmins(response);
      } catch (err) {
        console.error('Error loading admins:', err);
        setError('Failed to load admin list');
      } finally {
        setLoading(false);
      }
    };
    
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
          { borderBottomColor: '#E5E7EB' }
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
            <Text style={styles.adminName}>
              {item.name}
            </Text>
            {item.lastMessage?.timestamp && (
              <Text style={styles.timestamp}>
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
                style={styles.lastMessage}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                {item.lastMessage.message}
              </Text>
            ) : (
              <Text style={styles.noMessages}>No messages yet</Text>
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
          <Text style={styles.emptyText}>
            No admins found matching "{searchQuery}"
          </Text>
        ) : (
          <>
            <Text style={styles.emptyText}>No admins available</Text>
            <Text style={styles.emptySubtext}>
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
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading admins...</Text>
      </View>
    );
  }
  
  // Error state
  if (error) {
    return (
      <View style={styles.errorContainer}>
        <MaterialCommunityIcons name="alert-circle" size={60} color={theme.colors.error} />
        <Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>
        <TouchableOpacity 
          style={[styles.retryButton, { backgroundColor: theme.colors.primary }]}
          onPress={() => {
            setError(null);
            setLoading(true);
            chatService.getAdminChatList()
              .then(response => {
                setAdmins(response);
                setFilteredAdmins(response);
              })
              .catch(err => {
                setError('Failed to load admin list');
              })
              .finally(() => {
                setLoading(false);
              });
          }}
        >
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      {/* Search bar */}
      <View style={styles.searchContainer}>
        <MaterialCommunityIcons name="magnify" size={20} color="#9CA3AF" style={styles.searchIcon} />
        <TextInput
          style={styles.searchInput}
          placeholder="Search admins..."
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholderTextColor="#9CA3AF"
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
    backgroundColor: '#F9FAFB',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    marginHorizontal: 16,
    marginVertical: 12,
    paddingHorizontal: 12,
    height: 44,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    color: '#1F2937',
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
    fontSize: 16,
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 20,
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
