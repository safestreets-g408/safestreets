import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Card, useTheme } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';

const LocationInfoComponent = ({ locationName }) => {
  const theme = useTheme();
  
  return (
    <Animatable.View animation="fadeInUp" duration={800} delay={300}>
      <View style={styles.cardWrapper}>
        <Card style={styles.locationCard} elevation={3}>
          <View style={styles.locationCardContent}>
            <View style={[styles.locationIconContainer, { backgroundColor: theme.colors.primaryLight + '20' }]}>
              <MaterialCommunityIcons name="map-marker" size={24} color={theme.colors.primary} />
            </View>
            <View style={styles.locationTextContainer}>
              <Text style={[styles.locationTitle, { color: theme.colors.text }]}>{locationName}</Text>
              <Text style={[styles.locationDate, { color: theme.colors.textSecondary }]}>
                {new Date().toLocaleDateString('en-US', {
                  weekday: 'long', 
                  month: 'long', 
                  day: 'numeric',
                  year: 'numeric'
                })}
              </Text>
            </View>
            <View style={[styles.statusBadge, { backgroundColor: theme.colors.success }]}>
              <Text style={styles.statusText}>Active</Text>
            </View>
          </View>
        </Card>
      </View>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  cardWrapper: {
    marginBottom: 16,
    width: '100%'
  },
  locationCard: {
    borderRadius: 16,
    margin: 0,
    padding: 0,
    overflow: 'hidden'
  },
  locationCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  locationIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12
  },
  locationTextContainer: {
    flex: 1
  },
  locationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  locationDate: {
    fontSize: 12,
    marginTop: 2
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500'
  }
});

export default LocationInfoComponent;
