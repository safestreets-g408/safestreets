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
  
  // Get shadow based on elevation
  const shadowStyle = outlined ? styles.outlined : theme.shadows[elevation] || theme.shadows.medium;
  
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
