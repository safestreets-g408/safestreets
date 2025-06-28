import React from 'react';
import { StyleSheet, View, TouchableOpacity } from 'react-native';
import { Surface, useTheme } from 'react-native-paper';


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
  
  // Define shadow styles based on elevation level
  const getShadowStyle = () => {
    if (outlined) return styles.outlined;
    
    // Default shadow styles if theme.shadows is not available
    const defaultShadows = {
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
    
    // Use theme shadows if available, otherwise use default shadows
    return (theme.shadows && theme.shadows[elevation]) || defaultShadows[elevation] || defaultShadows.medium;
  };
  
  const shadowStyle = getShadowStyle();
  
  const cardStyle = [
    styles.card,
    { 
      backgroundColor: bgColor || theme.colors.surface,
      borderRadius: theme.roundness * 1.25, // More rounded for modern look
      borderColor: outlined ? theme.colors.border : 'transparent',
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
