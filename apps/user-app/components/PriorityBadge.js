import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useThemeContext } from '../context/ThemeContext';

const PriorityBadge = ({ 
  priority = 'medium', 
  size = 'medium', 
  showLabel = true,
  style 
}) => {
  const { theme, isDarkMode } = useThemeContext();
  
  const getPriorityConfig = (priority) => {
    const priorityLower = priority?.toLowerCase() || 'medium';
    
    const configs = {
      critical: {
        colors: isDarkMode ? ['#ef4444', '#dc2626'] : ['#dc2626', '#b91c1c'],
        textColor: 'white',
        icon: 'alert-circle',
        label: 'Critical',
        pulse: true,
      },
      high: {
        colors: isDarkMode ? ['#f97316', '#ea580c'] : ['#ea580c', '#c2410c'],
        textColor: 'white',
        icon: 'arrow-up-circle',
        label: 'High',
        pulse: false,
      },
      medium: {
        colors: isDarkMode ? ['#eab308', '#d97706'] : ['#d97706', '#b45309'],
        textColor: 'white',
        icon: 'remove-circle',
        label: 'Medium',
        pulse: false,
      },
      low: {
        colors: isDarkMode ? ['#22c55e', '#16a34a'] : ['#059669', '#047857'],
        textColor: 'white',
        icon: 'arrow-down-circle',
        label: 'Low',
        pulse: false,
      },
      info: {
        colors: isDarkMode ? ['#3b82f6', '#2563eb'] : ['#0ea5e9', '#0284c7'],
        textColor: 'white',
        icon: 'information-circle',
        label: 'Info',
        pulse: false,
      },
      default: {
        colors: [theme.colors.textMuted, theme.colors.textSecondary],
        textColor: 'white',
        icon: 'help-circle',
        label: priority || 'Unknown',
        pulse: false,
      }
    };
    
    return configs[priorityLower] || configs.default;
  };

  const getSizeConfig = (size) => {
    const sizes = {
      small: {
        height: 20,
        paddingHorizontal: 8,
        fontSize: 10,
        iconSize: 12,
        borderRadius: 10,
      },
      medium: {
        height: 24,
        paddingHorizontal: 10,
        fontSize: 11,
        iconSize: 14,
        borderRadius: 12,
      },
      large: {
        height: 28,
        paddingHorizontal: 12,
        fontSize: 12,
        iconSize: 16,
        borderRadius: 14,
      }
    };
    
    return sizes[size] || sizes.medium;
  };

  const priorityConfig = getPriorityConfig(priority);
  const sizeConfig = getSizeConfig(size);

  return (
    <LinearGradient
      colors={priorityConfig.colors}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 0 }}
      style={[
        styles.container,
        {
          height: sizeConfig.height,
          paddingHorizontal: sizeConfig.paddingHorizontal,
          borderRadius: sizeConfig.borderRadius,
        },
        style
      ]}
    >
      <Ionicons
        name={priorityConfig.icon}
        size={sizeConfig.iconSize}
        color={priorityConfig.textColor}
        style={!showLabel && styles.iconOnly}
      />
      {showLabel && (
        <Text
          style={[
            styles.text,
            {
              color: priorityConfig.textColor,
              fontSize: sizeConfig.fontSize,
            }
          ]}
        >
          {priorityConfig.label}
        </Text>
      )}
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-start',
    shadowColor: theme.shadows.small.shadowColor,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  text: {
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginLeft: 4,
  },
  iconOnly: {
    marginLeft: 0,
  },
});

export default PriorityBadge;
