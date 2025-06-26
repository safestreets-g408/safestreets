import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useTheme } from 'react-native-paper';
import * as Animatable from 'react-native-animatable';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ModernCard } from '../ui';

const NotificationsComponent = ({ notifications, navigation, handleMarkAsRead, handleNotificationPress, loading }) => {
  const theme = useTheme();
  
  const renderNotificationItem = ({ item }) => (
    <Animatable.View 
      animation="fadeIn" 
      duration={800} 
      delay={parseInt(item.id.split('_')[1] || item.id) * 100}
    >
      <TouchableOpacity onPress={() => handleNotificationPress(item)}>
        <View style={styles.cardWrapper}>
          <ModernCard style={[styles.notificationCard, !item.read && styles.unreadCard]}>
            <View style={styles.notificationContent}>
              <View style={styles.notificationHeader}>
                <View style={styles.notificationTitleContainer}>
                  <View style={styles.iconBadge}>
                    <MaterialCommunityIcons 
                      name={item.icon} 
                      size={24} 
                      color="#fff" 
                      style={{
                        backgroundColor: item.type === 'success' ? theme.colors.success : 
                                      item.type === 'info' ? theme.colors.info : theme.colors.warning,
                        width: 36,
                        height: 36,
                        borderRadius: 18,
                        textAlign: 'center',
                        lineHeight: 36
                      }}
                    />
                  </View>
                  <View style={{marginLeft: 10, flex: 1}}>
                    <Text style={styles.notificationTitle}>{item.title}</Text>
                    {!item.read && <View style={styles.badge}><Text style={styles.badgeText}>NEW</Text></View>}
                  </View>
                </View>
                <Text style={styles.notificationTime}>{item.time}</Text>
              </View>
              <Text style={styles.notificationMessage}>{item.message}</Text>
              <View style={styles.notificationActions}>
                <TouchableOpacity 
                  style={styles.actionButton}
                  onPress={() => handleMarkAsRead(item)}
                >
                  <MaterialCommunityIcons name="email-open" size={16} color={theme.colors.primary} />
                  <Text style={styles.actionButtonText}>
                    {item.read ? "Read" : "Mark as read"}
                  </Text>
                </TouchableOpacity>
                {item.reportId && (
                  <TouchableOpacity 
                    style={styles.actionButton}
                    onPress={() => navigation.navigate('ViewReport', { reportId: item.reportId })}
                  >
                    <MaterialCommunityIcons name="eye" size={16} color={theme.colors.secondary} />
                    <Text style={styles.actionButtonText}>View</Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          </ModernCard>
        </View>
      </TouchableOpacity>
    </Animatable.View>
  );
  
  return (
    <>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Recent Notifications</Text>
        <TouchableOpacity onPress={() => navigation.navigate('Notifications')}>
          <Text style={styles.seeAllText}>See All</Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#1a73e8" />
          <Text style={styles.loadingText}>Loading notifications...</Text>
        </View>
      ) : (
        <>
          {notifications.map(item => renderNotificationItem({ item }))}
          
          {notifications.length === 0 && (
            <View style={styles.emptyContainer}>
              <MaterialCommunityIcons name="bell-off-outline" size={48} color="#4285f4" />
              <Text style={styles.emptyText}>No notifications yet</Text>
            </View>
          )}
        </>
      )}
    </>
  );
};

const styles = StyleSheet.create({
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 12,
    paddingHorizontal: 4
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold'
  },
  seeAllText: {
    color: '#1a73e8',
    fontWeight: '500'
  },
  cardWrapper: {
    marginBottom: 16,
    width: '100%'
  },
  notificationCard: {
    margin: 0,
    borderRadius: 12
  },
  unreadCard: {
    borderLeftWidth: 4,
    borderLeftColor: '#1a73e8'
  },
  notificationContent: {
    padding: 16
  },
  notificationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8
  },
  notificationTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1
  },
  iconBadge: {
    alignItems: 'center',
    justifyContent: 'center'
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1
  },
  notificationTime: {
    fontSize: 12,
    color: '#666',
    marginLeft: 8
  },
  notificationMessage: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    marginBottom: 12
  },
  notificationActions: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    marginTop: 8
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16
  },
  actionButtonText: {
    fontSize: 14,
    color: '#1a73e8',
    marginLeft: 4
  },
  badge: {
    backgroundColor: '#f44336',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
    marginLeft: 8
  },
  badgeText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold'
  },
  loadingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24
  },
  loadingText: {
    marginTop: 8,
    color: '#666'
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    backgroundColor: '#f9f9f9',
    borderRadius: 12
  },
  emptyText: {
    marginTop: 12,
    color: '#666',
    fontSize: 16
  }
});

export default NotificationsComponent;
