import React from 'react';
import { View, StyleSheet, Text, TouchableOpacity } from 'react-native';
import { useTheme, Surface } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';


const StatusCard = ({
  status,
  icon,
  type = 'info',
  style,
  onPress,
  count,
  subtitle,
}) => {
  const theme = useTheme();
  
  // Define status types and their colors
  const statusTypes = {
    success: {
      backgroundColor: theme.colors.successLight + '20',
      iconColor: theme.colors.success,
      borderColor: theme.colors.success + '40',
      icon: icon || 'check-circle',
    },
    warning: {
      backgroundColor: theme.colors.warningLight + '20', 
      iconColor: theme.colors.warning,
      borderColor: theme.colors.warning + '40',
      icon: icon || 'alert',
    },
    error: {
      backgroundColor: theme.colors.errorLight + '20',
      iconColor: theme.colors.error,
      borderColor: theme.colors.error + '40',
      icon: icon || 'alert-circle',
    },
    info: {
      backgroundColor: theme.colors.infoLight + '20',
      iconColor: theme.colors.info,
      borderColor: theme.colors.info + '40',
      icon: icon || 'information',
    },
    neutral: {
      backgroundColor: theme.colors.secondaryLight + '20',
      iconColor: theme.colors.secondary,
      borderColor: theme.colors.secondary + '40',
      icon: icon || 'information',
    },
  };
  
  // Get the style for the selected status type
  const statusStyle = statusTypes[type] || statusTypes.info;
  
  const containerStyle = [
    styles.container,
    {
      backgroundColor: statusStyle.backgroundColor,
      borderColor: statusStyle.borderColor,
      borderRadius: theme.roundness,
    },
    style,
  ];
  
  const Content = () => (
    <>
      <View style={styles.iconContainer}>
        <MaterialCommunityIcons 
          name={statusStyle.icon} 
          size={28} 
          color={statusStyle.iconColor} 
        />
      </View>
      <View style={styles.textContainer}>
        <View style={styles.headerRow}>
          <Text style={[styles.statusText, { color: theme.colors.text }]}>
            {status}
          </Text>
          {count !== undefined && (
            <View style={[styles.countBadge, { backgroundColor: statusStyle.iconColor }]}>
              <Text style={styles.countText}>{count}</Text>
            </View>
          )}
        </View>
        {subtitle && (
          <Text style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
            {subtitle}
          </Text>
        )}
      </View>
    </>
  );

  if (onPress) {
    return (
      <TouchableOpacity
        style={containerStyle}
        onPress={onPress}
        activeOpacity={0.7}
      >
        <Content />
      </TouchableOpacity>
    );
  }

  return (
    <Animatable.View 
      animation="fadeIn" 
      duration={500} 
      style={containerStyle}
    >
      <Content />
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderLeftWidth: 4,
    marginVertical: 8,
  },
  iconContainer: {
    marginRight: 12,
  },
  textContainer: {
    flex: 1,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statusText: {
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
  },
  subtitle: {
    fontSize: 12,
    marginTop: 4,
  },
  countBadge: {
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
    minWidth: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  countText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
});

export default StatusCard;
