import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Card } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { theme } from '../theme';

const WeatherCard = ({ 
  weatherData, 
  locationName,
  index = 0 
}) => {
  const getWeatherIcon = (condition) => {
    switch (condition?.toLowerCase()) {
      case 'sunny':
      case 'clear':
        return 'weather-sunny';
      case 'cloudy':
      case 'overcast':
        return 'weather-cloudy';
      case 'partly cloudy':
        return 'weather-partly-cloudy';
      case 'rainy':
      case 'rain':
        return 'weather-rainy';
      case 'stormy':
        return 'weather-lightning-rainy';
      case 'snowy':
      case 'snow':
        return 'weather-snowy';
      case 'windy':
        return 'weather-windy';
      case 'foggy':
      case 'mist':
        return 'weather-fog';
      default:
        return 'weather-partly-cloudy';
    }
  };

  const getWeatherGradient = (condition) => {
    switch (condition?.toLowerCase()) {
      case 'sunny':
      case 'clear':
        return ['#FFD54F', '#FF8F00'];
      case 'cloudy':
      case 'overcast':
        return ['#90A4AE', '#546E7A'];
      case 'partly cloudy':
        return ['#81C784', '#388E3C'];
      case 'rainy':
      case 'rain':
        return ['#64B5F6', '#1976D2'];
      case 'stormy':
        return ['#7986CB', '#303F9F'];
      case 'snowy':
      case 'snow':
        return ['#E1F5FE', '#0277BD'];
      default:
        return theme.gradients.primary;
    }
  };

  if (!weatherData) {
    return null;
  }

  return (
    <Animatable.View
      animation="fadeInRight"
      duration={800}
      delay={index * 100}
    >
      <Card style={styles.weatherCard}>
        <LinearGradient
          colors={getWeatherGradient(weatherData.condition)}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.weatherGradient}
        >
          <View style={styles.weatherContent}>
            <View style={styles.locationContainer}>
              <MaterialCommunityIcons
                name="map-marker"
                size={16}
                color={theme.colors.surface}
              />
              <Text style={styles.locationText} numberOfLines={1}>
                {locationName || 'Current Location'}
              </Text>
            </View>
            
            <View style={styles.mainWeather}>
              <MaterialCommunityIcons
                name={getWeatherIcon(weatherData.condition)}
                size={48}
                color={theme.colors.surface}
              />
              <View style={styles.temperatureContainer}>
                <Text style={styles.temperature}>
                  {Math.round(weatherData.temperature)}°
                </Text>
                <Text style={styles.condition}>
                  {weatherData.condition}
                </Text>
              </View>
            </View>
            
            <View style={styles.weatherDetails}>
              <View style={styles.detailItem}>
                <MaterialCommunityIcons
                  name="water-percent"
                  size={16}
                  color={theme.colors.surface}
                />
                <Text style={styles.detailText}>
                  {weatherData.humidity}%
                </Text>
              </View>
              
              <View style={styles.detailItem}>
                <MaterialCommunityIcons
                  name="weather-windy"
                  size={16}
                  color={theme.colors.surface}
                />
                <Text style={styles.detailText}>
                  {weatherData.windSpeed} mph
                </Text>
              </View>
              
              <View style={styles.detailItem}>
                <MaterialCommunityIcons
                  name="thermometer"
                  size={16}
                  color={theme.colors.surface}
                />
                <Text style={styles.detailText}>
                  Feels like {Math.round(weatherData.feelsLike || weatherData.temperature)}°
                </Text>
              </View>
            </View>
          </View>
        </LinearGradient>
      </Card>
    </Animatable.View>
  );
};

const styles = StyleSheet.create({
  weatherCard: {
    marginHorizontal: 16,
    marginVertical: 8,
    borderRadius: theme.borderRadius.large,
    overflow: 'hidden',
    ...theme.shadows.medium,
  },
  weatherGradient: {
    borderRadius: theme.borderRadius.large,
  },
  weatherContent: {
    padding: 20,
  },
  locationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  locationText: {
    color: theme.colors.surface,
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 4,
    opacity: 0.9,
    flex: 1,
  },
  mainWeather: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  temperatureContainer: {
    marginLeft: 16,
    flex: 1,
  },
  temperature: {
    fontSize: 36,
    fontWeight: 'bold',
    color: theme.colors.surface,
    ...theme.typography.headlineLarge,
  },
  condition: {
    fontSize: 16,
    color: theme.colors.surface,
    fontWeight: '500',
    opacity: 0.9,
    textTransform: 'capitalize',
  },
  weatherDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  detailText: {
    color: theme.colors.surface,
    fontSize: 12,
    fontWeight: '500',
    marginLeft: 4,
    opacity: 0.9,
  },
});

export default WeatherCard;
