import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useNotifications } from '../../context/NotificationContext';

const NotificationBadge = ({ onPress, style }) => {
  const theme = useTheme();
  const { activeNotifications } = useNotifications();
  
  const count = activeNotifications.length;
  const hasNotifications = count > 0;
  
  return (
    <TouchableOpacity 
      style={[styles.container, style]} 
      onPress={onPress}
    >
      <MaterialCommunityIcons 
        name={hasNotifications ? 'bell' : 'bell-outline'} 
        size={24} 
        color={hasNotifications ? theme.colors.primary : theme.colors.onSurface} 
      />
      {hasNotifications && (
        <View style={[styles.badge, { backgroundColor: theme.colors.error }]}>
          <Text style={styles.badgeText}>
            {count > 99 ? '99+' : count}
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badge: {
    position: 'absolute',
    top: -2,
    right: -2,
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 4,
  },
  badgeText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
});

export default NotificationBadge;
