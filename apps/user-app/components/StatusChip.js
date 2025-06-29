import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Chip } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { useThemeContext } from '../context/ThemeContext';

const StatusChip = ({ 
  status, 
  size = 'medium', 
  showIcon = true, 
  style,
  textStyle 
}) => {
  const { theme, isDarkMode } = useThemeContext();
  
  const getStatusConfig = (status) => {
    const statusLower = status?.toLowerCase() || 'unknown';
    
    const configs = {
      pending: {
        color: theme.colors.warning,
        bgColor: theme.colors.warningBg,
        textColor: theme.colors.warningDark,
        icon: 'clock-outline',
        label: 'Pending'
      },
      in_progress: {
        color: theme.colors.info,
        bgColor: theme.colors.infoBg,
        textColor: theme.colors.infoDark,
        icon: 'refresh-outline',
        label: 'In Progress'
      },
      under_review: {
        color: isDarkMode ? '#a855f7' : '#7c3aed',
        bgColor: isDarkMode ? '#312e81' : '#f3f4f6',
        textColor: isDarkMode ? '#c4b5fd' : '#5b21b6',
        icon: 'eye-outline',
        label: 'Under Review'
      },
      resolved: {
        color: theme.colors.success,
        bgColor: theme.colors.successBg,
        textColor: theme.colors.successDark,
        icon: 'checkmark-circle-outline',
        label: 'Resolved'
      },
      approved: {
        color: theme.colors.success,
        bgColor: theme.colors.successBg,
        textColor: theme.colors.successDark,
        icon: 'checkmark-circle',
        label: 'Approved'
      },
      cancelled: {
        color: theme.colors.error,
        bgColor: theme.colors.errorBg,
        textColor: theme.colors.errorDark,
        icon: 'close-circle-outline',
        label: 'Cancelled'
      },
      rejected: {
        color: theme.colors.error,
        bgColor: theme.colors.errorBg,
        textColor: theme.colors.errorDark,
        icon: 'close-circle',
        label: 'Rejected'
      },
      default: {
        color: theme.colors.textMuted,
        bgColor: theme.colors.surfaceVariant,
        textColor: theme.colors.textSecondary,
        icon: 'help-circle-outline',
        label: status || 'Unknown'
      }
    };
    
    return configs[statusLower] || configs.default;
  };

  const getSizeConfig = (size) => {
    const sizes = {
      small: {
        height: 24,
        paddingHorizontal: 8,
        fontSize: 10,
        iconSize: 12,
        borderRadius: 12,
      },
      medium: {
        height: 28,
        paddingHorizontal: 12,
        fontSize: 12,
        iconSize: 14,
        borderRadius: 14,
      },
      large: {
        height: 32,
        paddingHorizontal: 16,
        fontSize: 14,
        iconSize: 16,
        borderRadius: 16,
      }
    };
    
    return sizes[size] || sizes.medium;
  };

  const statusConfig = getStatusConfig(status);
  const sizeConfig = getSizeConfig(size);

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: statusConfig.bgColor,
          borderColor: statusConfig.color,
          height: sizeConfig.height,
          paddingHorizontal: sizeConfig.paddingHorizontal,
          borderRadius: sizeConfig.borderRadius,
        },
        style
      ]}
    >
      {showIcon && (
        <Ionicons
          name={statusConfig.icon}
          size={sizeConfig.iconSize}
          color={statusConfig.textColor}
          style={styles.icon}
        />
      )}
      <Text
        style={[
          styles.text,
          {
            color: statusConfig.textColor,
            fontSize: sizeConfig.fontSize,
          },
          textStyle
        ]}
      >
        {statusConfig.label}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  icon: {
    marginRight: 4,
  },
  text: {
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
});

export default StatusChip;
