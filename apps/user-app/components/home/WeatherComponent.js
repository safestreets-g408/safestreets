import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ModernCard } from '../ui';

const WeatherComponent = ({ weatherData, locationName }) => {
  const theme = useTheme();
  
  return (
    <View style={styles.cardWrapper}>
      <ModernCard style={styles.weatherCard} elevation="medium">
        <LinearGradient
          colors={[theme.colors.info, theme.colors.infoLight]}
          style={styles.weatherGradient}
          start={{x: 0, y: 0}}
          end={{x: 1, y: 0}}
        >
          <View style={styles.weatherLeftSection}>
            <View style={styles.weatherLocationRow}>
              <MaterialCommunityIcons name="map-marker" size={16} color="#fff" />
              <Text style={styles.weatherCity}>{locationName}</Text>
            </View>
            <View style={styles.weatherMain}>
              <Text style={styles.weatherTemp}>
                {weatherData ? `${weatherData.temperature}°` : '72°'}
              </Text>
              <Text style={styles.weatherDesc}>
                {weatherData?.condition || 'Sunny'}
              </Text>
            </View>
            <View style={styles.weatherDetailsRow}>
              <View style={styles.weatherDetail}>
                <MaterialCommunityIcons name="water-percent" size={16} color="#fff" />
                <Text style={styles.weatherDetailText}>
                  {weatherData?.humidity || '10'}%
                </Text>
              </View>
              <View style={styles.weatherDetail}>
                <MaterialCommunityIcons name="weather-windy" size={16} color="#fff" />
                <Text style={styles.weatherDetailText}>
                  {weatherData?.windSpeed || '8'} mph
                </Text>
              </View>
              <View style={styles.weatherDetail}>
                <MaterialCommunityIcons name="calendar-today" size={16} color="#fff" />
                <Text style={styles.weatherDetailText}>
                  Today
                </Text>
              </View>
            </View>
          </View>
          
          <View style={styles.weatherIconSection}>
            <MaterialCommunityIcons 
              name={weatherData?.icon || "weather-sunny"} 
              size={80} 
              color="#fff"
              style={{opacity: 0.9}}
            />
          </View>
        </LinearGradient>
      </ModernCard>
    </View>
  );
};

const styles = StyleSheet.create({
  cardWrapper: {
    marginBottom: 16,
    width: '100%'
  },
  weatherCard: {
    margin: 0,
    padding: 0,
    overflow: 'hidden',
    borderRadius: 16
  },
  weatherGradient: {
    flexDirection: 'row',
    padding: 16,
    justifyContent: 'space-between',
    alignItems: 'center',
    borderRadius: 16
  },
  weatherLeftSection: {
    flex: 1,
  },
  weatherLocationRow: {
    flexDirection: 'row',
    alignItems: 'center'
  },
  weatherCity: {
    fontSize: 14,
    color: '#fff',
    marginLeft: 4
  },
  weatherMain: {
    marginVertical: 8
  },
  weatherTemp: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#fff'
  },
  weatherDesc: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.9)'
  },
  weatherDetailsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8
  },
  weatherDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16
  },
  weatherDetailText: {
    marginLeft: 4,
    color: 'rgba(255,255,255,0.9)',
    fontSize: 12
  },
  weatherIconSection: {
    alignItems: 'center',
    justifyContent: 'center'
  }
});

export default WeatherComponent;
