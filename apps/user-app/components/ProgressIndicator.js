import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const ProgressIndicator = ({ 
  steps = [], 
  currentStep = 0,
  title,
  index = 0 
}) => {
  const getStepStatus = (stepIndex) => {
    if (stepIndex < currentStep) return 'completed';
    if (stepIndex === currentStep) return 'current';
    return 'upcoming';
  };

  const getStepIcon = (status, stepIcon) => {
    switch (status) {
      case 'completed':
        return 'check-circle';
      case 'current':
        return stepIcon || 'circle-outline';
      case 'upcoming':
        return 'circle-outline';
      default:
        return 'circle-outline';
    }
  };

  const getStepColor = (status) => {
    switch (status) {
      case 'completed':
        return theme.colors.success;
      case 'current':
        return theme.colors.primary;
      case 'upcoming':
        return theme.colors.outline;
      default:
        return theme.colors.outline;
    }
  };

  return (
    <Animatable.View
      animation="fadeInUp"
      duration={600}
      delay={index * 100}
      style={styles.container}
    >
      {title && (
        <View style={styles.headerContainer}>
          <MaterialCommunityIcons
            name="progress-check"
            size={24}
            color={theme.colors.primary}
          />
          <Text style={styles.title}>{title}</Text>
        </View>
      )}
      
      <View style={styles.progressContainer}>
        {steps.map((step, stepIndex) => {
          const status = getStepStatus(stepIndex);
          const isLast = stepIndex === steps.length - 1;
          
          return (
            <View key={stepIndex} style={styles.stepContainer}>
              <View style={styles.stepIndicator}>
                <View style={[
                  styles.stepIcon,
                  { backgroundColor: getStepColor(status) }
                ]}>
                  <MaterialCommunityIcons
                    name={getStepIcon(status, step.icon)}
                    size={16}
                    color={status === 'upcoming' ? theme.colors.onSurfaceVariant : theme.colors.surface}
                  />
                </View>
                
                {!isLast && (
                  <View style={[
                    styles.stepLine,
                    { 
                      backgroundColor: stepIndex < currentStep 
                        ? theme.colors.success 
                        : theme.colors.outline 
                    }
                  ]} />
                )}
              </View>
              
              <View style={styles.stepContent}>
                <Text style={[
                  styles.stepTitle,
                  { 
                    color: status === 'upcoming' 
                      ? theme.colors.onSurfaceVariant 
                      : theme.colors.onSurface,
                    fontWeight: status === 'current' ? '700' : '600'
                  }
                ]}>
                  {step.title}
                </Text>
                
                {step.description && (
                  <Text style={styles.stepDescription}>
                    {step.description}
                  </Text>
                )}
                
                {step.timestamp && status === 'completed' && (
                  <Text style={styles.stepTimestamp}>
                    Completed: {new Date(step.timestamp).toLocaleDateString()}
                  </Text>
                )}
                
                {status === 'current' && step.estimatedTime && (
                  <View style={styles.estimatedTimeContainer}>
                    <MaterialCommunityIcons
                      name="clock-outline"
                      size={14}
                      color={theme.colors.primary}
                    />
                    <Text style={styles.estimatedTime}>
                      Est. {step.estimatedTime}
                    </Text>
                  </View>
                )}
              </View>
            </View>
          );
        })}
      </View>
      
      {/* Progress Bar */}
      <View style={styles.progressBarContainer}>
        <View style={styles.progressBarTrack}>
          <LinearGradient
            colors={[theme.colors.primary, theme.colors.secondary]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={[
              styles.progressBarFill,
              { width: `${(currentStep / (steps.length - 1)) * 100}%` }
            ]}
          />
        </View>
        <Text style={styles.progressText}>
          {Math.round((currentStep / (steps.length - 1)) * 100)}% Complete
        </Text>
      </View>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius.large,
    padding: 20,
    marginHorizontal: 16,
    marginVertical: 8,
    ...theme.shadows.small,
  },
  headerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    gap: 12,
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: theme.colors.onSurface,
    ...theme.typography.titleMedium,
  },
  progressContainer: {
    marginBottom: 20,
  },
  stepContainer: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  stepIndicator: {
    alignItems: 'center',
    marginRight: 16,
  },
  stepIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  stepLine: {
    width: 2,
    height: 24,
    marginTop: 4,
  },
  stepContent: {
    flex: 1,
    paddingTop: 4,
  },
  stepTitle: {
    fontSize: 16,
    marginBottom: 4,
  },
  stepDescription: {
    fontSize: 14,
    color: theme.colors.onSurfaceVariant,
    lineHeight: 18,
    marginBottom: 4,
  },
  stepTimestamp: {
    fontSize: 12,
    color: theme.colors.success,
    fontWeight: '500',
  },
  estimatedTimeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  estimatedTime: {
    fontSize: 12,
    color: theme.colors.primary,
    fontWeight: '500',
  },
  progressBarContainer: {
    alignItems: 'center',
  },
  progressBarTrack: {
    width: '100%',
    height: 8,
    backgroundColor: theme.colors.surfaceVariant,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: theme.colors.primary,
    fontWeight: '600',
  },
});

export default ProgressIndicator;
