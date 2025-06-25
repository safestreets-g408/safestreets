import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Card } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const QuickActionCard = ({ 
  title, 
  subtitle, 
  icon, 
  gradient, 
  onPress, 
  badge,
  index = 0 
}) => {
  return (
    <Animatable.View
      animation="fadeInUp"
      duration={600}
      delay={index * 100}
    >
      <TouchableOpacity onPress={onPress} activeOpacity={0.8}>
        <Card style={styles.actionCard}>
          <LinearGradient
            colors={gradient || theme.gradients.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.cardGradient}
          >
            <View style={styles.cardContent}>
              <View style={styles.iconContainer}>
                <MaterialCommunityIcons
                  name={icon}
                  size={32}
                  color={theme.colors.surface}
                />
                {badge && badge > 0 && (
                  <View style={styles.badge}>
                    <Text style={styles.badgeText}>
                      {badge > 99 ? '99+' : badge}
                    </Text>
                  </View>
                )}
              </View>
              
              <View style={styles.textContainer}>
                <Text style={styles.actionTitle}>{title}</Text>
                <Text style={styles.actionSubtitle}>{subtitle}</Text>
              </View>
              
              <MaterialCommunityIcons
                name="chevron-right"
                size={24}
                color={theme.colors.surface}
                style={styles.chevron}
              />
            </View>
          </LinearGradient>
        </Card>
      </TouchableOpacity>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  actionCard: {
    marginHorizontal: 4,
    marginVertical: 6,
    borderRadius: theme.borderRadius.large,
    overflow: 'hidden',
    ...theme.shadows.medium,
  },
  cardGradient: {
    borderRadius: theme.borderRadius.large,
  },
  cardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
  },
  iconContainer: {
    position: 'relative',
    marginRight: 16,
  },
  badge: {
    position: 'absolute',
    top: -6,
    right: -6,
    backgroundColor: theme.colors.error,
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: theme.colors.surface,
  },
  badgeText: {
    color: theme.colors.surface,
    fontSize: 11,
    fontWeight: 'bold',
  },
  textContainer: {
    flex: 1,
  },
  actionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: theme.colors.surface,
    marginBottom: 4,
    ...theme.typography.titleMedium,
  },
  actionSubtitle: {
    fontSize: 14,
    color: theme.colors.surfaceVariant,
    opacity: 0.9,
    fontWeight: '500',
  },
  chevron: {
    opacity: 0.8,
  },
});

export default QuickActionCard;
