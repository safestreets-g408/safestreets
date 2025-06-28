import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import * as Animatable from 'react-native-animatable';
import { Card, Title, Paragraph, useTheme } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const WelcomeHeader = ({ workerName, locationName }) => {
  const theme = useTheme();
  
  return (
    <Animatable.View 
      animation="fadeIn" 
      duration={800} 
      style={styles.headerContainer}
    >
      <View>
        <Text style={[styles.welcomeText, { color: theme.colors.textSecondary }]}>Welcome back,</Text>
        <Text style={[styles.nameText, { color: theme.colors.primary }]}>{workerName || 'Worker'}</Text>
      </View>
      <View style={styles.locationContainer}>
        <MaterialCommunityIcons name="map-marker" size={18} color={theme.colors.primary} />
        <Text style={[styles.locationText, { color: theme.colors.textSecondary }]}>{locationName}</Text>
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
  },
  nameText: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  locationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  locationText: {
    fontSize: 14,
    marginLeft: 4,
  },
});

export default WelcomeHeader;
