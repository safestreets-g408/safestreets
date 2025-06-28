import React, { useState, useEffect, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  StatusBar,
  Dimensions,
  Platform,
  ActivityIndicator as RNActivityIndicator,
} from 'react-native';
import { 
  Divider, 
  Avatar,
  useTheme,
  Title,
  Switch as PaperSwitch,
  Button,
  RadioButton,
  Portal,
  ActivityIndicator
} from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { useAuth } from '../context/AuthContext';
import { useThemeContext, THEME_MODE } from '../context/ThemeContext';
import { ModernCard, ConsistentHeader } from '../components/ui';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: screenWidth } = Dimensions.get('window');

// Default theme colors to use when theme context is not yet available
const defaultColors = {
  background: '#f9fafb',
  surface: '#ffffff',
  text: '#111827',
  textSecondary: '#6b7280',
  primary: '#2563eb',
  primaryDark: '#1d4ed8',
  error: '#dc2626',
  outline: '#d1d5db',
  surfaceVariant: '#f3f4f6',
};

// Separate ThemeSafeWrapper into its own component file (for future refactoring)
// For now, keep it here but make it more robust
const ThemeSafeWrapper = ({ children }) => {
  // Always provide default fallback without depending on theme context
  return (
    <View style={{ flex: 1, backgroundColor: defaultColors.background }}>
      {children}
    </View>
  );
};

const SettingsScreen = ({ navigation }) => {
  // Access auth context
  const { fieldWorker, logout } = useAuth();
  
  // Safely get theme values with fallbacks
  const paperTheme = useTheme();
  
  // Safely destructure theme context with fallbacks for everything
  const themeContext = useThemeContext();
  const { 
    themeMode = THEME_MODE.SYSTEM, 
    isDarkMode = false, 
    changeThemeMode = () => {}, 
    theme: contextTheme = {}, 
    systemTheme = 'light',
    getEffectiveThemeMode = () => THEME_MODE.LIGHT
  } = themeContext || {};
  
  // Create a stable theme object that combines both sources safely
  const theme = React.useMemo(() => {
    const baseTheme = {
      roundness: 12,
      colors: {
        ...defaultColors,
        ...(paperTheme?.colors || {}),
        ...(contextTheme?.colors || {})
      }
    };
    return baseTheme;
  }, [paperTheme, contextTheme]);
  
  const [notifications, setNotifications] = useState(true);
  const [locationServices, setLocationServices] = useState(true);
  const [autoSync, setAutoSync] = useState(true);
  
  // Theme selection state - keep it simple
  const [selectedTheme, setSelectedTheme] = useState(themeMode);
  
  // Update selectedTheme when themeMode changes from context
  React.useEffect(() => {
    if (selectedTheme !== themeMode) {
      setSelectedTheme(themeMode);
    }
  }, [themeMode]);
  
  // Handle theme change - debounced to prevent rapid state changes
  const handleThemeChange = React.useCallback((newMode) => {
    console.log('Theme change requested:', newMode);
    setSelectedTheme(newMode);
    // Use a timeout to debounce rapid changes
    setTimeout(() => {
      changeThemeMode(newMode);
    }, 100);
  }, [changeThemeMode]);

  // Theme options
  const themeOptions = React.useMemo(() => [
    { 
      value: THEME_MODE.SYSTEM, 
      label: `Use System Settings (${systemTheme === 'dark' ? 'Dark' : 'Light'})`, 
      icon: systemTheme === 'dark' ? 'moon-waning-crescent' : 'white-balance-sunny' 
    },
    { value: THEME_MODE.LIGHT, label: 'Light Mode', icon: 'white-balance-sunny' },
    { value: THEME_MODE.DARK, label: 'Dark Mode', icon: 'moon-waning-crescent' }
  ], [systemTheme]);

  const settingsSections = [
    {
      title: 'App Settings',
      icon: 'cog',
      items: [
        {
          title: 'Theme',
          subtitle: 'Choose your preferred app theme',
          icon: 'theme-light-dark',
          type: 'theme-selector',
        },
        {
          title: 'Push Notifications',
          subtitle: 'Receive alerts for new tasks and updates',
          icon: 'bell',
          type: 'toggle',
          value: notifications,
          onToggle: setNotifications,
        },
        {
          title: 'Location Services',
          subtitle: 'Allow app to access your location',
          icon: 'map-marker',
          type: 'toggle',
          value: locationServices,
          onToggle: setLocationServices,
        },
        {
          title: 'Auto Sync',
          subtitle: 'Automatically sync data when connected',
          icon: 'sync',
          type: 'toggle',
          value: autoSync,
          onToggle: setAutoSync,
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
          type: 'action',
          onPress: () => Alert.alert('Clear Cache', 'Are you sure you want to clear the app cache?'),
        },
        {
          title: 'Data Usage',
          subtitle: 'View data consumption statistics',
          icon: 'chart-donut',
          type: 'action',
          onPress: () => Alert.alert('Data Usage', 'Feature coming soon'),
        },
        {
          title: 'Offline Mode',
          subtitle: 'Manage offline data and sync settings',
          icon: 'cloud-off',
          type: 'action',
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
          type: 'action',
          onPress: () => Alert.alert('Help Center', 'Visit our help center for assistance'),
        },
        {
          title: 'Send Feedback',
          subtitle: 'Share your thoughts and suggestions',
          icon: 'message-text',
          type: 'action',
          onPress: () => Alert.alert('Send Feedback', 'Thank you for your feedback!'),
        },
        {
          title: 'Report a Bug',
          subtitle: 'Report issues or bugs',
          icon: 'bug',
          type: 'action',
          onPress: () => Alert.alert('Report Bug', 'Bug reporting feature coming soon'),
        },
        {
          title: 'Rate the App',
          subtitle: 'Rate us on the app store',
          icon: 'star',
          type: 'action',
          onPress: () => Alert.alert('Rate App', 'Thank you for rating our app!'),
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

  // Wrap render in try/catch for safety
  try {
    return (
      <ThemeSafeWrapper>
        <SafeAreaView style={[styles.container, { backgroundColor: theme?.colors?.background || defaultColors.background }]} edges={['top']}>
          <StatusBar 
            barStyle={isDarkMode ? "light-content" : "dark-content"} 
            backgroundColor={isDarkMode ? theme?.colors?.surface || defaultColors.surface : theme?.colors?.background || defaultColors.background} 
            translucent={false}
            animated={true}
          />
      
      {/* Header */}
      <ConsistentHeader
        title="Settings"
        subtitle="Manage your preferences"
        useGradient={true}
        elevated={true}
        user={fieldWorker}
        showAvatar={true}
      />

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Settings Sections */}
        {settingsSections.map((section, sectionIndex) => (
          <Animatable.View
            key={section.title}
            animation="fadeInUp"
            duration={600}
            delay={sectionIndex * 100}
            style={styles.sectionWrapper}
          >
            <ModernCard style={styles.sectionCard}>
              <View style={styles.sectionHeader}>
                <MaterialCommunityIcons
                  name={section.icon}
                  size={24}
                  color={theme?.colors?.primary || defaultColors.primary}
                />
                <Title style={[styles.sectionTitle, { color: theme?.colors?.text || defaultColors.text }]}>
                  {section.title}
                </Title>
              </View>
              
              <Divider style={styles.sectionDivider} />
              
              {section.items.map((item, itemIndex) => (
                <View key={item.title}>
                  {item.type === 'theme-selector' ? (
                    <>
                      <View style={styles.settingItem}>
                        <View style={styles.settingItemContent}>
                          <MaterialCommunityIcons
                            name={item.icon}
                            size={20}
                            color={theme?.colors?.primary || defaultColors.primary}
                            style={styles.itemIcon}
                          />
                          <View style={styles.settingTextContainer}>
                            <Text style={[styles.itemTitle, { color: theme?.colors?.text || defaultColors.text }]}>
                              {item.title}
                            </Text>
                            <Text style={[styles.itemSubtitle, { color: theme?.colors?.textSecondary || defaultColors.textSecondary }]}>
                              {item.subtitle}
                            </Text>
                          </View>
                        </View>
                      </View>
                      <View style={styles.themeSelectorContainer}>
                        <RadioButton.Group 
                          onValueChange={value => setSelectedTheme(value)} 
                          value={selectedTheme}
                        >
                          {themeOptions.map((option) => (
                            <View key={option.value} style={{ overflow: 'hidden' }}>
                              <TouchableOpacity
                                style={[
                                  styles.themeOption,
                                  selectedTheme === option.value && {
                                    backgroundColor: theme?.colors?.surfaceVariant || defaultColors.surfaceVariant,
                                    borderRadius: 8,
                                    borderWidth: 1,
                                    borderColor: theme?.colors?.primary || defaultColors.primary,
                                  }
                                ]}
                                onPress={() => setSelectedTheme(option.value)}
                              >
                                <View style={styles.themeOptionContent}>
                                  <MaterialCommunityIcons
                                    name={option.icon}
                                    size={24}
                                    color={selectedTheme === option.value 
                                      ? theme?.colors?.primary || defaultColors.primary
                                      : theme?.colors?.secondary || defaultColors.textSecondary}
                                    style={styles.themeOptionIcon}
                                  />
                                  <View>
                                    <Text style={[
                                      styles.themeOptionLabel, 
                                      { 
                                        color: selectedTheme === option.value 
                                          ? theme?.colors?.primary || defaultColors.primary
                                          : theme?.colors?.text || defaultColors.text,
                                        fontWeight: selectedTheme === option.value ? '600' : '400'
                                      }
                                    ]}>
                                      {option.label}
                                    </Text>
                                    
                                    {option.value === THEME_MODE.SYSTEM && (
                                      <Text style={{ 
                                        fontSize: 12, 
                                        color: theme?.colors?.textSecondary || defaultColors.textSecondary,
                                        marginTop: 2
                                      }}>
                                        Using {systemTheme === 'dark' ? 'dark' : 'light'} theme from your device settings
                                      </Text>
                                    )}
                                  </View>
                                </View>
                                <RadioButton 
                                  value={option.value} 
                                  color={theme?.colors?.primary || defaultColors.primary}
                                  status={selectedTheme === option.value ? 'checked' : 'unchecked'}
                                />
                              </TouchableOpacity>
                            </View>
                          ))}
                        </RadioButton.Group>
                      </View>
                    </>
                  ) : item.type === 'toggle' ? (
                    <View style={styles.toggleItem}>
                      <View style={styles.toggleItemContent}>
                        <MaterialCommunityIcons
                          name={item.icon}
                          size={20}
                          color={theme?.colors?.primary || defaultColors.primary}
                          style={styles.itemIcon}
                        />
                        <View style={styles.toggleTextContainer}>
                          <Text style={[styles.itemTitle, { color: theme?.colors?.text || defaultColors.text }]}>
                            {item.title}
                          </Text>
                          <Text style={[styles.itemSubtitle, { color: theme?.colors?.textSecondary || defaultColors.textSecondary }]}>
                            {item.subtitle}
                          </Text>
                        </View>
                      </View>
                      <PaperSwitch
                        value={item.value}
                        onValueChange={item.onToggle}
                        color={theme?.colors?.primary || defaultColors.primary}
                      />
                    </View>
                  ) : (
                    <TouchableOpacity
                      style={styles.settingItem}
                      onPress={item.onPress}
                      activeOpacity={0.7}
                    >
                      <View style={styles.settingItemContent}>
                        <MaterialCommunityIcons
                          name={item.icon}
                          size={20}
                          color={theme?.colors?.primary || defaultColors.primary}
                          style={styles.itemIcon}
                        />
                        <View style={styles.settingTextContainer}>
                          <Text style={[styles.itemTitle, { color: theme?.colors?.text || defaultColors.text }]}>
                            {item.title}
                          </Text>
                          <Text style={[styles.itemSubtitle, { color: theme?.colors?.textSecondary || defaultColors.textSecondary }]}>
                            {item.subtitle}
                          </Text>
                        </View>
                      </View>
                      <MaterialCommunityIcons
                        name="chevron-right"
                        size={20}
                        color={theme?.colors?.outline || defaultColors.outline}
                      />
                    </TouchableOpacity>
                  )}
                  {itemIndex < section.items.length - 1 && (
                    <Divider style={styles.itemDivider} />
                  )}
                </View>
              ))}
            </ModernCard>
          </Animatable.View>
        ))}

        {/* Logout Section */}
        <Animatable.View
          animation="fadeInUp"
          duration={600}
          delay={settingsSections.length * 100}
          style={styles.logoutSection}
        >
          <Button
            mode="outlined"
            onPress={handleLogout}
            icon="logout"
            textColor={theme?.colors?.error || defaultColors.error}
            style={[styles.logoutButton, { borderColor: theme?.colors?.error || defaultColors.error }]}
            contentStyle={styles.logoutButtonContent}
          >
            Logout
          </Button>
        </Animatable.View>

        <View style={styles.bottomSpacing} />
      </ScrollView>
    </SafeAreaView>
    </ThemeSafeWrapper>
    );
  } catch (error) {
    console.error('Error rendering SettingsScreen:', error);
    // Return a simple fallback UI
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: defaultColors.background }}>
        <Text style={{ color: defaultColors.text, fontSize: 16, textAlign: 'center', marginHorizontal: 20 }}>
          There was a problem loading the settings. Please try again.
        </Text>
      </View>
    );
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 100,
  },
  sectionWrapper: {
    marginBottom: 16,
  },
  sectionCard: {
    overflow: 'hidden',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    gap: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  sectionDivider: {
    marginHorizontal: 20,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  settingItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  itemIcon: {
    marginRight: 12,
  },
  settingTextContainer: {
    flex: 1,
  },
  itemTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 2,
  },
  itemSubtitle: {
    fontSize: 14,
  },
  toggleItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  toggleItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  toggleTextContainer: {
    flex: 1,
  },
  itemDivider: {
    marginHorizontal: 20,
  },
  // Theme selector styles
  themeSelectorContainer: {
    paddingHorizontal: 20,
    paddingBottom: 12,
  },
  themeOption: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    marginVertical: 4,
    overflow: 'hidden',
  },
  themeOptionContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  themeOptionIcon: {
    marginRight: 16,
  },
  themeOptionLabel: {
    fontSize: 16,
    fontWeight: '500',
  },
  // Logout section
  logoutSection: {
    marginTop: 24,
    paddingHorizontal: 16,
  },
  logoutButton: {
    borderRadius: 12,
  },
  logoutButtonContent: {
    paddingVertical: 8,
  },
  bottomSpacing: {
    height: 32,
  },
});

export default SettingsScreen;
