import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { ModernCard } from '../ui';

const QuickActionsComponent = ({ actions, navigation }) => {
  const theme = useTheme();
  
  return (
    <>
      <View style={styles.sectionHeader}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Quick Actions</Text>
      </View>

      <View style={styles.quickActionsContainer}>
        {actions.map((action, index) => (
          <Animatable.View 
            key={action.id} 
            animation={action.animation} 
            duration={500} 
            delay={index * 100}
            style={styles.quickActionWrapper}
          >
            <TouchableOpacity 
              activeOpacity={0.8}
              onPress={() => navigation.navigate(action.screen)}
            >
              <ModernCard style={styles.quickAction} elevation="medium">
                <LinearGradient
                  colors={action.gradientColors || [theme.colors.primary, theme.colors.primaryDark]}
                  style={styles.quickActionGradient}
                  start={{x: 0, y: 0}}
                  end={{x: 1, y: 1}}
                >
                  <View style={styles.quickActionIconContainer}>
                    <MaterialCommunityIcons name={action.icon} size={28} color="#fff" />
                  </View>
                  <Text style={styles.quickActionTitle}>{action.title}</Text>
                </LinearGradient>
              </ModernCard>
            </TouchableOpacity>
          </Animatable.View>
        ))}
      </View>
    </>
  );
};

const styles = StyleSheet.create({
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 16,
    marginBottom: 12,
    paddingHorizontal: 4
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold'
  },
  quickActionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -8,
    marginBottom: 16
  },
  quickActionWrapper: {
    width: '50%',
    padding: 8
  },
  quickAction: {
    borderRadius: 16,
    overflow: 'hidden',
    margin: 0,
    padding: 0
  },
  quickActionGradient: {
    padding: 16,
    alignItems: 'center',
    justifyContent: 'center',
    height: 100
  },
  quickActionIconContainer: {
    marginBottom: 12
  },
  quickActionTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500'
  }
});

export default QuickActionsComponent;
