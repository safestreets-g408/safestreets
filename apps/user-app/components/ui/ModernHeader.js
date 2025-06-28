import React from 'react';
import { Platform } from 'react-native';
import { useTheme } from 'react-native-paper';
import ConsistentHeader from './ConsistentHeader';

/**
 * ModernHeader - A simplified wrapper around ConsistentHeader
 * Provides a modern header look with predefined defaults suitable for most screens
 */
const ModernHeader = ({
  title,
  subtitle,
  user,
  actions,
  showAvatar = false,
  useGradient = true,
  elevated = true,
  back,
  transparent = false,
  centered = false,
  style,
  blurEffect = Platform.OS === 'ios', // Enable blur by default on iOS
}) => {
  const theme = useTheme();
  
  // Default gradient colors
  const defaultGradientColors = [theme.colors.primary, theme.colors.primaryDark];

  return (
    <ConsistentHeader
      title={title}
      subtitle={subtitle}
      user={user}
      actions={actions}
      showAvatar={showAvatar}
      useGradient={useGradient}
      gradientColors={defaultGradientColors}
      elevated={elevated}
      back={back}
      transparent={transparent}
      centered={centered}
      style={style}
      blurEffect={blurEffect}
    />
  );
};

export default ModernHeader;
