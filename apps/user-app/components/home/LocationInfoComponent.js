import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Card } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';

const LocationInfoComponent = ({ locationName }) => {
  return (
    <Animatable.View animation="fadeInUp" duration={800} delay={300}>
      <View style={styles.cardWrapper}>
        <Card style={styles.locationCard} elevation={3}>
          <View style={styles.locationCardContent}>
            <View style={styles.locationIconContainer}>
              <MaterialCommunityIcons name="map-marker" size={24} color="#003366" />
            </View>
            <View style={styles.locationTextContainer}>
              <Text style={styles.locationTitle}>{locationName}</Text>
              <Text style={styles.locationDate}>
                {new Date().toLocaleDateString('en-US', {
                  weekday: 'long', 
                  month: 'long', 
                  day: 'numeric',
                  year: 'numeric'
                })}
              </Text>
            </View>
            <View style={styles.statusBadge}>
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
    backgroundColor: 'rgba(0, 51, 102, 0.1)',
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
    color: '#333'
  },
  locationDate: {
    fontSize: 12,
    color: '#666',
    marginTop: 2
  },
  statusBadge: {
    backgroundColor: '#4CAF50',
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
