import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Card, Title, Paragraph, ProgressBar } from 'react-native-paper';
import * as Animatable from 'react-native-animatable';

const StatsCard = ({ stats }) => {
  return (
    <Animatable.View animation="fadeInUp" duration={1000} delay={300}>
      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>City Overview</Title>
          
          <View style={styles.statRow}>
            <View style={styles.statItem}>
              <Title style={styles.statValue}>{stats.reportsThisWeek}</Title>
              <Paragraph style={styles.statLabel}>New Reports</Paragraph>
            </View>
            <View style={styles.statItem}>
              <Title style={styles.statValue}>{stats.repairsCompleted}</Title>
              <Paragraph style={styles.statLabel}>Completed</Paragraph>
            </View>
            <View style={styles.statItem}>
              <Title style={styles.statValue}>{stats.pendingIssues}</Title>
              <Paragraph style={styles.statLabel}>Pending</Paragraph>
            </View>
          </View>
          
          <View style={styles.progressContainer}>
            <Paragraph style={styles.progressLabel}>
              Completion Rate: {stats.completionRate}%
            </Paragraph>
            <ProgressBar
              progress={stats.completionRate / 100}
              color="#4CAF50"
              style={styles.progressBar}
            />
          </View>
        </Card.Content>
      </Card>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  card: {
    marginHorizontal: 16,
    marginVertical: 10,
    borderRadius: 10,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 10,
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statValue: {
    fontSize: 20,
    color: '#003366',
    fontWeight: 'bold',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
  progressContainer: {
    marginTop: 15,
  },
  progressLabel: {
    fontSize: 14,
    marginBottom: 5,
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
});

export default StatsCard;
