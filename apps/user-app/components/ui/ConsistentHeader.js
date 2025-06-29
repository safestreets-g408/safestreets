import React from 'react';
import { View, StyleSheet, Text, StatusBar, Platform } from 'react-native';
import { Avatar, useTheme, IconButton } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { BlurView } from 'expo-blur';
import { useThemeContext } from '../../context/ThemeContext';

const ConsistentHeader = ({
  title,
  subtitle,
  user,
  actions = [],
  style,
  showAvatar = true,
  useGradient = false,
  gradientColors,
  elevated = true,
  back,
  transparent = false,
  centered = false,
  blurEffect = false, // iOS blur effect option
}) => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  const insets = useSafeAreaInsets();
  
  // Calculate proper padding for both platforms
  const isIOS = Platform.OS === 'ios';
  const topPadding = isIOS ? insets.top : (StatusBar.currentHeight || 0);
  const bottomPadding = isIOS ? 12 : 16; // Adjusted for iOS
  const horizontalPadding = isIOS ? 20 : 16; // More side padding on iOS
  
  // Default gradient colors
  const defaultGradientColors = transparent 
    ? ['transparent', 'transparent'] 
    : gradientColors || theme.colors.gradient?.primary || [theme.colors.primary, theme.colors.primaryDark];

  const headerStyles = {
    paddingTop: topPadding + (isIOS ? 10 : 8), // More top padding on iOS
    paddingBottom: bottomPadding,
    paddingHorizontal: horizontalPadding,
  };

  const  renderContent = () => {
    const isIOS = Platform.OS === 'ios';
    
    return (
      <View style={[styles.content, headerStyles]}>
        {/* Left side - Back button or avatar */}
        <View style={styles.leftSection}>
          {back && back.visible ? (
            <IconButton
              icon={isIOS ? "chevron-left" : "arrow-left"} // Use chevron on iOS
              iconColor={useGradient ? "#fff" : theme.colors.onSurface}
              size={isIOS ? 28 : 24} // Slightly larger on iOS
              onPress={back.onPress}
              style={[styles.backButton, isIOS && styles.iosBackButton]}
            />
          ) : showAvatar && user ? (
            <Avatar.Text
              size={isIOS ? 34 : 36} // Slightly smaller on iOS
              label={user.name ? user.name.charAt(0).toUpperCase() : 'U'}
              style={[
                styles.avatar, 
                { backgroundColor: theme.colors.primaryLight },
                isIOS && { marginLeft: 6 } // Adjust for iOS
              ]}
              labelStyle={{ color: theme.colors.onPrimary }}
            />
          ) : (
            <View style={styles.spacer} />
          )}
        </View>
      
        {/* Center - Title and subtitle */}
        <View style={[styles.titleContainer, centered && styles.titleCentered]}>
          <Text 
            style={[
              styles.title, 
              { color: useGradient ? "#fff" : theme.colors.onSurface },
              centered && styles.titleTextCentered,
              isIOS && styles.iosTitleText // iOS-specific text style
            ]}
            numberOfLines={1}
            adjustsFontSizeToFit={isIOS} // Dynamic font sizing for iOS
            minimumFontScale={0.8}
          >
            {title}
          </Text>
          {subtitle && (
            <Text 
              style={[
                styles.subtitle, 
                { color: useGradient ? "rgba(255,255,255,0.8)" : theme.colors.onSurfaceVariant },
                centered && styles.subtitleTextCentered,
                isIOS && styles.iosSubtitleText // iOS-specific subtitle style
              ]}
              numberOfLines={1}
            >
              {subtitle}
            </Text>
          )}
        </View>
        
        {/* Right side - Actions */}
        <View style={styles.rightSection}>
          {actions.map((action, index) => (
            <IconButton
              key={index}
              icon={action.icon}
              iconColor={useGradient ? "#fff" : theme.colors.onSurface}
              size={isIOS ? 22 : 24} // Slightly smaller on iOS
              onPress={action.onPress}
              style={[styles.actionButton, isIOS && { marginRight: -4 }]} // Adjusted for iOS
            />
          ))}
          {actions.length === 0 && <View style={styles.spacer} />}
        </View>
      </View>
    );
  };

  // For iOS with blur effect
  if (isIOS && blurEffect && !useGradient && !transparent) {
    return (
      <BlurView 
        intensity={60} 
        tint="light" 
        style={[
          styles.header,
          elevated && styles.iosElevated, // iOS-specific elevation
          style
        ]}
      >
        <StatusBar barStyle="dark-content" />
        {renderContent()}
      </BlurView>
    );
  }
  
  // For gradient headers
  if (useGradient) {
    return (
      <LinearGradient
        colors={defaultGradientColors}
        style={[
          styles.header,
          elevated && (isIOS ? styles.iosElevated : styles.elevated),
          style
        ]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <StatusBar barStyle="light-content" />
        {renderContent()}
      </LinearGradient>
    );
  }

  // Standard header
  return (
    <View
      style={[
        styles.header,
        { backgroundColor: transparent ? 'transparent' : theme.colors.surface },
        elevated && (isIOS ? styles.iosElevated : styles.elevated),
        style
      ]}
    >
      <StatusBar barStyle={transparent || useGradient ? "light-content" : "dark-content"} />
      {renderContent()}
    </View>
  );
};

const styles = StyleSheet.create({
  header: {
    zIndex: 1000,
  },
  elevated: {
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  // iOS-specific elevation with better shadows
  iosElevated: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.18,
    shadowRadius: 2.0,
    zIndex: 1000,
    marginTop: Platform.OS === 'ios' ? -50 : 0, // Adjust for iOS notch
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: Platform.OS === 'ios' ? 50 : 56, // Slightly smaller height on iOS
  },
  leftSection: {
    width: 60,
    alignItems: 'flex-start',
    justifyContent: 'center',
  },
  rightSection: {
    width: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  titleContainer: {
    flex: 1,
    paddingHorizontal: 8,
    justifyContent: 'center',
  },
  titleCentered: {
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
    lineHeight: 24,
  },
  // iOS-specific title styling
  iosTitleText: {
    fontSize: 18,
    fontWeight: '600', // iOS prefers slightly lighter font weights
    letterSpacing: -0.5, // Tighter letter spacing looks better on iOS
  },
  titleTextCentered: {
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    lineHeight: 16,
    marginTop: 2,
  },
  // iOS-specific subtitle styling
  iosSubtitleText: {
    fontSize: 13,
    fontWeight: '400',
    opacity: 0.8,
    marginTop: 1, // Less space between title and subtitle on iOS
  },
  subtitleTextCentered: {
    textAlign: 'center',
  },
  avatar: {
    marginLeft: 4,
  },
  backButton: {
    margin: 0,
    marginLeft: -8,
  },
  // iOS-specific back button
  iosBackButton: {
    marginLeft: -4, // Less negative margin on iOS
  },
  actionButton: {
    margin: 0,
    marginRight: -8,
  },
  spacer: {
    width: 40,
  },
});

export default ConsistentHeader;
