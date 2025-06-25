import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Switch,
} from 'react-native';
import { Card, List, Divider, Avatar } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { useAuth } from '../context/AuthContext';
import { theme } from '../theme';

const SettingsScreen = ({ navigation }) => {
  const { fieldWorker, logout } = useAuth();
  const [notifications, setNotifications] = useState(true);
  const [locationServices, setLocationServices] = useState(true);
  const [autoSync, setAutoSync] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  const settingsSections = [
    {
      title: 'Account',
      icon: 'account-cog',
      items: [
        {
          title: 'Personal Information',
          subtitle: 'Update your profile details',
          icon: 'account-edit',
          onPress: () => navigation.navigate('Profile'),
        },
        {
          title: 'Change Password',
          subtitle: 'Update your account password',
          icon: 'lock-reset',
          onPress: () => Alert.alert('Change Password', 'Contact administrator to change password'),
        },
        {
          title: 'Work Preferences',
          subtitle: 'Set your work schedule and availability',
          icon: 'briefcase-clock',
          onPress: () => Alert.alert('Work Preferences', 'Feature coming soon'),
        },
      ],
    },
    {
      title: 'App Settings',
      icon: 'cog',
      items: [
        {
          title: 'Push Notifications',
          subtitle: 'Receive alerts for new tasks and updates',
          icon: 'bell',
          toggle: true,
          value: notifications,
          onToggle: setNotifications,
        },
        {
          title: 'Location Services',
          subtitle: 'Allow app to access your location',
          icon: 'map-marker',
          toggle: true,
          value: locationServices,
          onToggle: setLocationServices,
        },
        {
          title: 'Auto Sync',
          subtitle: 'Automatically sync data when connected',
          icon: 'sync',
          toggle: true,
          value: autoSync,
          onToggle: setAutoSync,
        },
        {
          title: 'Dark Mode',
          subtitle: 'Switch to dark theme',
          icon: 'theme-light-dark',
          toggle: true,
          value: darkMode,
          onToggle: setDarkMode,
        },
      ],
    },
    {
      title: 'Data & Storage',
      icon: 'database',
      items: [
        {
          title: 'Cache Management',
          subtitle: 'Clear app cache to free up space',
          icon: 'delete-sweep',
          onPress: () => Alert.alert('Clear Cache', 'Are you sure you want to clear the app cache?'),
        },
        {
          title: 'Data Usage',
          subtitle: 'View data consumption statistics',
          icon: 'chart-donut',
          onPress: () => Alert.alert('Data Usage', 'Feature coming soon'),
        },
        {
          title: 'Offline Mode',
          subtitle: 'Manage offline data and sync settings',
          icon: 'cloud-off',
          onPress: () => Alert.alert('Offline Mode', 'Feature coming soon'),
        },
      ],
    },
    {
      title: 'Support & Feedback',
      icon: 'help-circle',
      items: [
        {
          title: 'Help Center',
          subtitle: 'Get help and find answers',
          icon: 'help-circle-outline',
          onPress: () => Alert.alert('Help Center', 'Visit our help center for assistance'),
        },
        {
          title: 'Send Feedback',
          subtitle: 'Share your thoughts and suggestions',
          icon: 'message-text',
          onPress: () => Alert.alert('Send Feedback', 'Thank you for your feedback!'),
        },
        {
          title: 'Report a Bug',
          subtitle: 'Report issues or bugs',
          icon: 'bug',
          onPress: () => Alert.alert('Report Bug', 'Bug reporting feature coming soon'),
        },
        {
          title: 'Rate the App',
          subtitle: 'Rate us on the app store',
          icon: 'star',
          onPress: () => Alert.alert('Rate App', 'Thank you for rating our app!'),
        },
      ],
    },
    {
      title: 'Legal & Privacy',
      icon: 'shield-check',
      items: [
        {
          title: 'Privacy Policy',
          subtitle: 'Read our privacy policy',
          icon: 'shield-account',
          onPress: () => Alert.alert('Privacy Policy', 'Opening privacy policy...'),
        },
        {
          title: 'Terms of Service',
          subtitle: 'Read our terms of service',
          icon: 'file-document',
          onPress: () => Alert.alert('Terms of Service', 'Opening terms of service...'),
        },
        {
          title: 'Data Protection',
          subtitle: 'Learn about data protection',
          icon: 'security',
          onPress: () => Alert.alert('Data Protection', 'Your data is protected with us'),
        },
      ],
    },
  ];

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            await logout();
            navigation.reset({
              index: 0,
              routes: [{ name: 'Login' }],
            });
          },
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={theme.gradients.primary}
        style={styles.header}
      >
        <Animatable.View 
          animation="fadeInDown" 
          duration={800}
          style={styles.headerContent}
        >
          <Avatar.Text
            size={80}
            label={fieldWorker?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'FW'}
            style={styles.avatar}
            labelStyle={styles.avatarLabel}
          />
          <Text style={styles.userName}>{fieldWorker?.name || 'Field Worker'}</Text>
          <Text style={styles.userEmail}>{fieldWorker?.email || 'field@worker.com'}</Text>
          <View style={styles.userBadge}>
            <MaterialCommunityIcons 
              name="shield-check" 
              size={16} 
              color={theme.colors.surface} 
            />
            <Text style={styles.userBadgeText}>Verified Worker</Text>
          </View>
        </Animatable.View>
      </LinearGradient>

      {/* Settings Sections */}
      <View style={styles.content}>
        {settingsSections.map((section, sectionIndex) => (
          <Animatable.View
            key={section.title}
            animation="fadeInUp"
            duration={600}
            delay={sectionIndex * 100}
          >
            <Card style={styles.sectionCard}>
              <View style={styles.sectionHeader}>
                <MaterialCommunityIcons
                  name={section.icon}
                  size={24}
                  color={theme.colors.primary}
                />
                <Text style={styles.sectionTitle}>{section.title}</Text>
              </View>
              
              <Divider style={styles.sectionDivider} />
              
              {section.items.map((item, itemIndex) => (
                <TouchableOpacity
                  key={item.title}
                  style={styles.settingItem}
                  onPress={item.onPress}
                  activeOpacity={0.7}
                >
                  <View style={styles.settingLeft}>
                    <View style={styles.settingIconContainer}>
                      <MaterialCommunityIcons
                        name={item.icon}
                        size={20}
                        color={theme.colors.primary}
                      />
                    </View>
                    <View style={styles.settingText}>
                      <Text style={styles.settingTitle}>{item.title}</Text>
                      <Text style={styles.settingSubtitle}>{item.subtitle}</Text>
                    </View>
                  </View>
                  
                  <View style={styles.settingRight}>
                    {item.toggle ? (
                      <Switch
                        value={item.value}
                        onValueChange={item.onToggle}
                        trackColor={{
                          false: theme.colors.surfaceVariant,
                          true: theme.colors.primaryContainer,
                        }}
                        thumbColor={item.value ? theme.colors.primary : theme.colors.outline}
                      />
                    ) : (
                      <MaterialCommunityIcons
                        name="chevron-right"
                        size={20}
                        color={theme.colors.onSurfaceVariant}
                      />
                    )}
                  </View>
                </TouchableOpacity>
              ))}
            </Card>
          </Animatable.View>
        ))}

        {/* App Info */}
        <Animatable.View
          animation="fadeInUp"
          duration={600}
          delay={600}
        >
          <Card style={styles.appInfoCard}>
            <View style={styles.appInfo}>
              <MaterialCommunityIcons
                name="information"
                size={20}
                color={theme.colors.onSurfaceVariant}
              />
              <Text style={styles.appInfoText}>
                Safe Streets v1.0.0 • Build 2024.1
              </Text>
            </View>
          </Card>
        </Animatable.View>

        {/* Logout Button */}
        <Animatable.View
          animation="fadeInUp"
          duration={600}
          delay={700}
        >
          <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
            <LinearGradient
              colors={[theme.colors.error, '#d32f2f']}
              style={styles.logoutGradient}
            >
              <MaterialCommunityIcons
                name="logout"
                size={20}
                color={theme.colors.surface}
              />
              <Text style={styles.logoutText}>Logout</Text>
            </LinearGradient>
          </TouchableOpacity>
        </Animatable.View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  header: {
    paddingTop: 60,
    paddingBottom: 40,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  headerContent: {
    alignItems: 'center',
  },
  avatar: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    marginBottom: 16,
    ...theme.shadows.medium,
  },
  avatarLabel: {
    fontSize: 28,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  userName: {
    fontSize: 24,
    fontWeight: '700',
    color: theme.colors.surface,
    marginBottom: 4,
    ...theme.typography.headlineMedium,
  },
  userEmail: {
    fontSize: 16,
    color: theme.colors.surfaceVariant,
    opacity: 0.9,
    marginBottom: 12,
  },
  userBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: theme.borderRadius.medium,
    gap: 6,
  },
  userBadgeText: {
    color: theme.colors.surface,
    fontSize: 12,
    fontWeight: '600',
  },
  content: {
    padding: 16,
    marginTop: -20,
  },
  sectionCard: {
    marginBottom: 16,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius.large,
    ...theme.shadows.small,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 12,
    gap: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: theme.colors.onSurface,
    ...theme.typography.titleMedium,
  },
  sectionDivider: {
    marginHorizontal: 20,
    marginBottom: 8,
    backgroundColor: theme.colors.outline,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surfaceVariant,
    borderBottomOpacity: 0.3,
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: theme.colors.primaryContainer,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.onSurface,
    marginBottom: 2,
  },
  settingSubtitle: {
    fontSize: 13,
    color: theme.colors.onSurfaceVariant,
    lineHeight: 16,
  },
  settingRight: {
    marginLeft: 16,
  },
  appInfoCard: {
    marginBottom: 16,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius.large,
    ...theme.shadows.small,
  },
  appInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    gap: 8,
  },
  appInfoText: {
    fontSize: 14,
    color: theme.colors.onSurfaceVariant,
    fontWeight: '500',
  },
  logoutButton: {
    marginBottom: 32,
    borderRadius: theme.borderRadius.large,
    overflow: 'hidden',
    ...theme.shadows.small,
  },
  logoutGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 8,
  },
  logoutText: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.surface,
  },
});

export default SettingsScreen;
