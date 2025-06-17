import { useState, useEffect } from 'react';
import * as Location from 'expo-location';

/**
 * Custom hook for handling location services
 */
export const useLocation = () => {
  const [location, setLocation] = useState(null);
  const [locationName, setLocationName] = useState('Loading...');
  const [errorMsg, setErrorMsg] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getCurrentLocation();
  }, []);

  const getCurrentLocation = async () => {
    try {
      setLoading(true);
      let { status } = await Location.requestForegroundPermissionsAsync();
      
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        setLocationName('Location unavailable');
        setLoading(false);
        return;
      }

      let location = await Location.getCurrentPositionAsync({});
      setLocation(location);
      
      // Get location name from coordinates
      let geocode = await Location.reverseGeocodeAsync({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude
      });
      
      if (geocode && geocode.length > 0) {
        const { city, region } = geocode[0];
        setLocationName(city || region || 'Unknown location');
      }
    } catch (error) {
      setErrorMsg('Could not fetch location');
      setLocationName('Location unavailable');
    } finally {
      setLoading(false);
    }
  };

  return {
    location,
    locationName,
    errorMsg,
    loading,
    getCurrentLocation
  };
};
