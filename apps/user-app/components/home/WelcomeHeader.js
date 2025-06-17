import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import * as Animatable from 'react-native-animatable';
import { Card, Title, Paragraph } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const WelcomeHeader = ({ workerName, locationName }) => {
  return (
    <Animatable.View 
      animation="fadeIn" 
      duration={800} 
      style={styles.headerContainer}
    >
      <View>
        <Text style={styles.welcomeText}>Welcome back,</Text>
        <Text style={styles.nameText}>{workerName || 'Worker'}</Text>
      </View>
      <View style={styles.locationContainer}>
        <MaterialCommunityIcons name="map-marker" size={18} color="#003366" />
        <Text style={styles.locationText}>{locationName}</Text>
      </View>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    marginVertical: 16,
    paddingHorizontal: 16,
  },
  welcomeText: {
    fontSize: 16,
    color: '#555',
  },
  nameText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#003366',
  },
  locationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  locationText: {
    fontSize: 14,
    color: '#555',
    marginLeft: 4,
  },
});

export default WelcomeHeader;
