import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { ModernCard } from '../ui';

const StatsComponent = ({ cityStats }) => {
  const theme = useTheme();
  
  return (
    <Animatable.View animation="fadeIn" duration={800} delay={200}>
      <ModernCard style={styles.cityStatsCard} elevation="medium">
        <View style={styles.cityStatsHeader}>
          <Text style={[styles.cityStatsTitle, { color: theme.colors.text }]}>
            City Statistics
          </Text>
          <Text style={[styles.cityStatsSubtitle, { color: theme.colors.textSecondary }]}>
            Weekly Performance
          </Text>
        </View>
        
        <View style={styles.statsGrid}>
          <View style={[
            styles.statBox, 
            { backgroundColor: theme.colors.info + '15', borderColor: theme.colors.info + '30' }
          ]}>
            <View style={[styles.statIconContainer, { backgroundColor: theme.colors.info + '20' }]}> 
              <MaterialCommunityIcons name="chart-line" size={18} color={theme.colors.info} />
            </View>
            <Text style={[styles.statValue, { color: theme.colors.text }]}>
              {cityStats.reportsThisWeek}
            </Text>
            <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>
              New Reports
            </Text>
          </View>
          
          <View style={[
            styles.statBox, 
            { backgroundColor: theme.colors.success + '15', borderColor: theme.colors.success + '30' }
          ]}>
            <View style={[styles.statIconContainer, { backgroundColor: theme.colors.success + '20' }]}>
              <MaterialCommunityIcons name="check-all" size={18} color={theme.colors.success} />
            </View>
            <Text style={[styles.statValue, { color: theme.colors.text }]}>
              {cityStats.repairsCompleted}
            </Text>
            <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>
              Completed
            </Text>
          </View>
          
          <View style={[
            styles.statBox, 
            { backgroundColor: theme.colors.warning + '15', borderColor: theme.colors.warning + '30' }
          ]}>
            <View style={[styles.statIconContainer, { backgroundColor: theme.colors.warning + '20' }]}>
              <MaterialCommunityIcons name="clock-outline" size={18} color={theme.colors.warning} />
            </View>
            <Text style={[styles.statValue, { color: theme.colors.text }]}>
              {cityStats.pendingIssues}
            </Text>
            <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>
              Pending
            </Text>
          </View>
        </View>
      </ModernCard>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  cityStatsCard: {
    padding: 16,
    borderRadius: 16,
    marginBottom: 16
  },
  cityStatsHeader: {
    marginBottom: 16
  },
  cityStatsTitle: {
    fontSize: 18,
    fontWeight: 'bold'
  },
  cityStatsSubtitle: {
    fontSize: 14,
    marginTop: 4
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginHorizontal: -8
  },
  statBox: {
    flex: 1,
    borderRadius: 12,
    padding: 12,
    marginHorizontal: 8,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.05)'
  },
  statIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    marginVertical: 4
  },
  statLabel: {
    fontSize: 12
  }
});

export default StatsComponent;
