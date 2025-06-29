import React from 'react';
import { StyleSheet, View, TouchableOpacity } from 'react-native';
import { Surface, useTheme } from 'react-native-paper';
import { useThemeContext } from '../../context/ThemeContext';


const ModernCard = ({
  children,
  style,
  onPress,
  elevation = 'medium',
  outlined = false,
  interactive = false,
  bgColor,
}) => {
  const theme = useTheme();
  const { isDarkMode } = useThemeContext();
  
  // Define shadow styles based on elevation level
  const getShadowStyle = () => {
    if (outlined) return styles.outlined;
    
    // Dark mode shadows should be more subtle
    const darkShadows = {
      small: {
        elevation: 2,
        shadowColor: 'rgba(0,0,0,0.5)',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.3,
        shadowRadius: 2,
      },
      medium: {
        elevation: 3,
        shadowColor: 'rgba(0,0,0,0.6)',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.4,
        shadowRadius: 4,
      },
      large: {
        elevation: 4,
        shadowColor: 'rgba(0,0,0,0.7)',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.5,
        shadowRadius: 6,
      }
    };
    
    // Light mode shadows
    const lightShadows = {
      small: {
        elevation: 2,
        shadowColor: 'rgba(0,0,0,0.25)',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.08,
        shadowRadius: 3,
      },
      medium: {
        elevation: 4,
        shadowColor: 'rgba(0,0,0,0.25)',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.12,
        shadowRadius: 6,
      },
      large: {
        elevation: 8,
        shadowColor: 'rgba(0,0,0,0.25)',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.2,
        shadowRadius: 12,
      }
    };
    
    const shadows = isDarkMode ? darkShadows : lightShadows;
    
    // Use theme shadows if available, otherwise use mode-specific shadows
    return (theme.shadows && theme.shadows[elevation]) || shadows[elevation] || shadows.medium;
  };
  
  const shadowStyle = getShadowStyle();
  
  const cardStyle = [
    styles.card,
    { 
      backgroundColor: bgColor || theme.colors.surface,
      borderRadius: theme.roundness * 1.25, // More rounded for modern look
      borderColor: outlined ? theme.colors.border : 'transparent',
      // Add subtle border for dark mode to improve definition
      ...(isDarkMode && !outlined && {
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.1)'
      })
    },
    shadowStyle,
    interactive && styles.interactive,
    style,
  ];

  if (onPress) {
    return (
      <TouchableOpacity 
        style={cardStyle} 
        onPress={onPress}
        activeOpacity={0.9}
      >
        {children}
      </TouchableOpacity>
    );
  }

  return (
    <Surface style={cardStyle}>
      {children}
    </Surface>
  );
};

const styles = StyleSheet.create({
  card: {
    padding: 16,
    margin: 4,
    borderWidth: 0,
    overflow: 'hidden',
  },
  interactive: {
    transform: [{scale: 1}],
    transition: '0.3s',
  },
  outlined: {
    elevation: 0,
    shadowColor: 'transparent',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0,
    shadowRadius: 0,
    borderWidth: 1,
  },
});

export default ModernCard;
