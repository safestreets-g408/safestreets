import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const StatsCard = ({ 
  title,
  value,
  subtitle,
  icon,
  trend,
  trendValue,
  color = 'primary',
  onPress,
  animated = true,
  style
}) => {
  const getColorConfig = (color) => {
    const configs = {
      primary: {
        gradient: theme.colors.gradients.primary,
        iconBg: theme.colors.primaryLight,
        trendColor: theme.colors.success,
      },
      success: {
        gradient: theme.colors.gradients.success,
        iconBg: theme.colors.successLight,
        trendColor: theme.colors.success,
      },
      warning: {
        gradient: theme.colors.gradients.warning,
        iconBg: theme.colors.warningLight,
        trendColor: theme.colors.warning,
      },
      error: {
        gradient: theme.colors.gradients.error,
        iconBg: theme.colors.errorLight,
        trendColor: theme.colors.error,
      },
      info: {
        gradient: theme.colors.gradients.info,
        iconBg: theme.colors.infoLight,
        trendColor: theme.colors.info,
      },
    };
    
    return configs[color] || configs.primary;
  };

  const colorConfig = getColorConfig(color);

  const CardContent = () => (
    <View style={[styles.container, style]}>
      <LinearGradient
        colors={[...colorConfig.gradient, colorConfig.gradient[0] + '20']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.gradient}
      >
        <View style={styles.content}>
          <View style={styles.header}>
            <View style={[styles.iconContainer, { backgroundColor: 'rgba(255,255,255,0.2)' }]}>
              <Ionicons name={icon} size={24} color="white" />
            </View>
            {trend && trendValue && (
              <View style={styles.trendContainer}>
                <Ionicons 
                  name={trend === 'up' ? 'trending-up' : trend === 'down' ? 'trending-down' : 'remove'} 
                  size={16} 
                  color="rgba(255,255,255,0.8)" 
                />
                <Text style={styles.trendText}>{trendValue}</Text>
              </View>
            )}
          </View>
          
          <View style={styles.body}>
            <Text style={styles.value}>{value}</Text>
            <Text style={styles.title}>{title}</Text>
            {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
          </View>
        </View>
      </LinearGradient>
      
      {/* Decorative elements */}
      <View style={styles.decorativeElements}>
        <View style={[styles.circle, styles.circle1]} />
        <View style={[styles.circle, styles.circle2]} />
        <View style={[styles.circle, styles.circle3]} />
      </View>
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.9}>
        {animated ? (
          <Animatable.View animation="fadeInUp" duration={600} useNativeDriver>
            <CardContent />
          </Animatable.View>
        ) : (
          <CardContent />
        )}
      </TouchableOpacity>
    );
  }

  return animated ? (
    <Animatable.View animation="fadeInUp" duration={600} useNativeDriver>
      <CardContent />
    </Animatable.View>
  ) : (
    <CardContent />
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: theme.shadows.large.shadowColor,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 20,
    elevation: 8,
  },
  gradient: {
    padding: 20,
    minHeight: 120,
    position: 'relative',
  },
  content: {
    flex: 1,
    zIndex: 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  trendContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  trendText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  body: {
    flex: 1,
  },
  value: {
    color: 'white',
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 4,
    letterSpacing: -0.5,
  },
  title: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    fontWeight: '500',
  },
  decorativeElements: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    zIndex: 1,
  },
  circle: {
    position: 'absolute',
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 50,
  },
  circle1: {
    width: 100,
    height: 100,
    top: -50,
    right: -50,
  },
  circle2: {
    width: 60,
    height: 60,
    bottom: -30,
    left: -30,
  },
  circle3: {
    width: 40,
    height: 40,
    top: 20,
    right: 20,
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
});

export default StatsCard;
