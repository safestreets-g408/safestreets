import React from 'react';
import { View, Text, StyleSheet, Platform, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Avatar, useTheme, IconButton } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useThemeContext, THEME_MODE } from '../../context/ThemeContext';
import { NotificationBadge } from '../notifications';

const HeaderComponent = ({ fieldWorker, cityStats, navigation }) => {
  const theme = useTheme();
  const { themeMode, isDarkMode, changeThemeMode } = useThemeContext();
  
  const handleThemeToggle = () => {
    // Cycle through theme modes: Light -> Dark -> System -> Light
    switch (themeMode) {
      case THEME_MODE.LIGHT:
        changeThemeMode(THEME_MODE.DARK);
        break;
      case THEME_MODE.DARK:
        changeThemeMode(THEME_MODE.SYSTEM);
        break;
      case THEME_MODE.SYSTEM:
        changeThemeMode(THEME_MODE.LIGHT);
        break;
      default:
        changeThemeMode(THEME_MODE.LIGHT);
    }
  };

  const handleProfilePress = () => {
    if (navigation) {
      navigation.navigate('Profile');
    }
  };

  const getThemeIcon = () => {
    switch (themeMode) {
      case THEME_MODE.LIGHT:
        return 'white-balance-sunny';
      case THEME_MODE.DARK:
        return 'moon-waning-crescent';
      case THEME_MODE.SYSTEM:
        return isDarkMode ? 'theme-light-dark' : 'theme-light-dark';
      default:
        return 'white-balance-sunny';
    }
  };
  
  return (
    <LinearGradient
      colors={[theme.colors.primary, theme.colors.primaryDark]}
      style={styles.header}
      start={{x: 0, y: 0}}
      end={{x: 1, y: 1}}
    >
      <Animatable.View 
        animation="fadeIn" 
        duration={800}
        style={styles.headerContent}
      >
        <View>
          <Animatable.Text 
            animation="fadeInDown" 
            duration={800} 
            style={styles.greeting}
          >
            Hello, {fieldWorker?.name?.split(' ')[0] || 'Field Worker'}
          </Animatable.Text>
          <Animatable.Text 
            animation="fadeInDown" 
            duration={800} 
            delay={200}
            style={styles.subGreeting}
          >
            {fieldWorker?.specialization || 'Road Maintenance'} • {fieldWorker?.region || 'Your Region'}
          </Animatable.Text>
        </View>
        
        <Animatable.View 
          animation="fadeIn"
          duration={1000}
          style={styles.headerActions}
        >
          {/* Notifications Badge */}
          <TouchableOpacity 
            onPress={() => navigation.navigate('Notifications')}
            style={styles.notificationButton}
            activeOpacity={0.7}
          >
            <NotificationBadge />
          </TouchableOpacity>
          
          {/* Theme Toggle Button */}
          <TouchableOpacity 
            onPress={handleThemeToggle}
            style={styles.themeToggleButton}
            activeOpacity={0.7}
          >
            <MaterialCommunityIcons 
              name={getThemeIcon()} 
              size={22} 
              color="rgba(255,255,255,0.9)" 
            />
          </TouchableOpacity>
          
          {/* Profile Avatar */}
          <TouchableOpacity 
            onPress={handleProfilePress}
            style={styles.avatarContainer}
            activeOpacity={0.8}
          >
            {fieldWorker?.profileImage ? (
              <Avatar.Image 
                size={44} 
                source={{ uri: fieldWorker.profileImage }} 
                style={styles.avatar}
              />
            ) : (
              <Avatar.Text 
                size={44} 
                label={fieldWorker?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'FW'} 
                style={styles.avatar}
                color="#fff"
                backgroundColor="rgba(255,255,255,0.2)" 
              />
            )}
            <View style={styles.onlineIndicator} />
          </TouchableOpacity>
        </Animatable.View>
      </Animatable.View>
      
      <View style={styles.headerStatsContainer}>
        <View style={styles.headerStat}>
          <Text style={styles.headerStatNumber}>{cityStats.reportsThisWeek || 0}</Text>
          <Text style={styles.headerStatLabel}>New Reports</Text>
        </View>
        <View style={styles.headerStatDivider} />
        <View style={styles.headerStat}>
          <Text style={styles.headerStatNumber}>{cityStats.pendingIssues || 0}</Text>
          <Text style={styles.headerStatLabel}>Pending</Text>
        </View>
        <View style={styles.headerStatDivider} />
        <View style={styles.headerStat}>
          <Text style={styles.headerStatNumber}>{cityStats.repairsCompleted || 0}</Text>
          <Text style={styles.headerStatLabel}>Completed</Text>
        </View>
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  header: {
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 24,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    elevation: 8,
    shadowColor: 'rgba(0,0,0,0.4)',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    marginTop: Platform.OS === 'ios' ? -48 : 0,
    zIndex: 10,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  notificationButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 2,
  },
  themeToggleButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  avatarContainer: {
    position: 'relative',
  },
  avatar: {
    elevation: 4,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  onlineIndicator: {
    width: 12,
    height: 12,
    backgroundColor: '#4CAF50',
    borderRadius: 6,
    borderWidth: 2,
    borderColor: 'white',
    position: 'absolute',
    bottom: 0,
    right: 0,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    letterSpacing: 0.5,
  },
  subGreeting: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.85)',
    marginTop: 4,
    letterSpacing: 0.3,
  },
  headerStatsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 16,
    paddingVertical: 12,
    marginHorizontal: 4,
    marginTop: 8,
  },
  headerStat: {
    alignItems: 'center',
  },
  headerStatNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  headerStatLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  headerStatDivider: {
    width: 1,
    height: '60%',
    backgroundColor: 'rgba(255,255,255,0.3)',
    alignSelf: 'center',
  },
});

export default HeaderComponent;
