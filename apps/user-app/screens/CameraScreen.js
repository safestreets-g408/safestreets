// screens/CameraScreen.js
import React, { useState, useRef, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  Image, 
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import { MaterialIcons, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { 
  Button, 
  Card, 
  Title, 
  Paragraph, 
  Surface,
  useTheme,
  IconButton,
  Chip,
  ActivityIndicator as PaperActivityIndicator
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { ConsistentHeader } from '../components/ui';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const CameraScreen = ({ navigation }) => {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [hasLocationPermission, setHasLocationPermission] = useState(null);
  const theme = useTheme();
  const [capturedImage, setCapturedImage] = useState(null);
  const [location, setLocation] = useState(null);
  const [locationAddress, setLocationAddress] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [cameraFacing, setCameraFacing] = useState('back');
  const [fetchingLocation, setFetchingLocation] = useState(false);
  
  const cameraRef = useRef(null);

  const toggleCameraFacing = () => {
    setCameraFacing(
      cameraFacing === 'back'
        ? 'front'
        : 'back'
    );
  };

  useEffect(() => {
    const requestPermissions = async () => {
      try {
        if (!cameraPermission?.granted) {
          await requestCameraPermission();
        }
        
        // Request foreground location permission first
        const { status: foregroundStatus } = await Location.requestForegroundPermissionsAsync();
        
        if (foregroundStatus === 'granted') {
          // Try to get background location permission for better accuracy
          try {
            const { status: backgroundStatus } = await Location.requestBackgroundPermissionsAsync();
            console.log('Background location status:', backgroundStatus);
          } catch (bgError) {
            console.log('Background location not available:', bgError);
          }
          
          // Pre-fetch location for faster response when user takes photo
          setHasLocationPermission(true);
          getCurrentLocation();
        } else {
          setHasLocationPermission(false);
        }

        if (!cameraPermission?.granted) {
          Alert.alert('Permission required', 'Camera permission is needed to take photos');
        }

        if (foregroundStatus !== 'granted') {
          Alert.alert('Permission required', 'Location permission is needed to tag your reports with accurate location details');
        }
      } catch (error) {
        console.error('Error requesting permissions:', error);
        Alert.alert('Error', 'Failed to request necessary permissions');
      }
    };

    requestPermissions();
  }, [cameraPermission, requestCameraPermission]);

  const getCurrentLocation = async () => {
    if (hasLocationPermission) {
      try {
        setFetchingLocation(true);
        
        // First, try to get high accuracy location with timeout
        let position;
        try {
          position = await Promise.race([
            Location.getCurrentPositionAsync({
              accuracy: Location.Accuracy.BestForNavigation,
              timeInterval: 5000,
              distanceInterval: 1,
            }),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('High accuracy timeout')), 8000)
            )
          ]);
        } catch (highAccuracyError) {
          console.log('High accuracy failed, trying balanced accuracy:', highAccuracyError.message);
          // Fallback to balanced accuracy if high accuracy fails or times out
          position = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.Balanced,
            timeInterval: 3000,
          });
        }
        
        setLocation(position);
        
        // Get detailed address from coordinates with multiple attempts
        await getDetailedAddress(position.coords.latitude, position.coords.longitude);
        
        setFetchingLocation(false);
      } catch (error) {
        console.error('Error getting location:', error);
        Alert.alert(
          'Location Error',
          'Could not get your current location. You can still submit the report.',
          [{ text: 'OK' }]
        );
        setFetchingLocation(false);
      }
    }
  };

  const getDetailedAddress = async (latitude, longitude) => {
    try {
      // Get multiple address responses for better accuracy
      const [primaryAddress, fallbackAddress] = await Promise.allSettled([
        Location.reverseGeocodeAsync({
          latitude,
          longitude
        }),
        // Try with slightly offset coordinates for better results
        Location.reverseGeocodeAsync({
          latitude: latitude + 0.00001,
          longitude: longitude + 0.00001
        })
      ]);

      let bestAddress = null;
      
      // Use primary address if available
      if (primaryAddress.status === 'fulfilled' && primaryAddress.value?.length > 0) {
        bestAddress = primaryAddress.value[0];
      } 
      // Fallback to secondary address
      else if (fallbackAddress.status === 'fulfilled' && fallbackAddress.value?.length > 0) {
        bestAddress = fallbackAddress.value[0];
      }

      if (bestAddress) {
        // Format detailed address with all available components
        const addressComponents = [];
        
        // Door number and street
        if (bestAddress.streetNumber) {
          addressComponents.push(bestAddress.streetNumber);
        }
        if (bestAddress.street) {
          addressComponents.push(bestAddress.street);
        }
        
        // Area and locality
        if (bestAddress.district) {
          addressComponents.push(bestAddress.district);
        }
        if (bestAddress.subregion) {
          addressComponents.push(bestAddress.subregion);
        }
        
        // City and region
        if (bestAddress.city) {
          addressComponents.push(bestAddress.city);
        }
        if (bestAddress.region) {
          addressComponents.push(bestAddress.region);
        }
        
        // Postal code
        if (bestAddress.postalCode) {
          addressComponents.push(bestAddress.postalCode);
        }
        
        // Country
        if (bestAddress.country) {
          addressComponents.push(bestAddress.country);
        }

        const formattedAddress = addressComponents
          .filter(Boolean)
          .filter((component, index, arr) => arr.indexOf(component) === index) // Remove duplicates
          .join(', ');
          
        setLocationAddress(formattedAddress || 'Address not available');
        
        // Store detailed location info for submission
        setLocation(prev => ({
          ...prev,
          addressDetails: {
            streetNumber: bestAddress.streetNumber,
            street: bestAddress.street,
            district: bestAddress.district,
            subregion: bestAddress.subregion,
            city: bestAddress.city,
            region: bestAddress.region,
            postalCode: bestAddress.postalCode,
            country: bestAddress.country,
            formattedAddress: formattedAddress
          }
        }));
      } else {
        setLocationAddress('Detailed address not available');
      }
      
    } catch (error) {
      console.error('Error getting detailed address:', error);
      setLocationAddress('Address lookup failed');
    }
  };

  const takePicture = async () => {
    if (!cameraRef.current) {
      Alert.alert('Error', 'Camera is not ready. Please wait or restart the app.');
      return;
    }

    try {
      // Start location fetching immediately when user starts taking picture
      const locationPromise = hasLocationPermission ? getCurrentLocation() : null;
      
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        skipProcessing: false,
        exif: true, // Include EXIF data which may contain GPS info
      });

      if (photo && photo.uri) {
        setCapturedImage(photo);
        
        // If location wasn't started above, start it now
        if (!locationPromise && hasLocationPermission) {
          getCurrentLocation();
        }
      } else {
        throw new Error('Invalid photo data');
      }
    } catch (error) {
      console.error('Error taking picture:', error);

      Alert.alert(
        'Camera Error',
        'Failed to capture image with camera. Would you like to select an image from your gallery instead?',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Select from Gallery', onPress: pickImage }
        ]
      );
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [16, 9],
        quality: 0.7,
      });

      if (!result.canceled && result.assets && result.assets[0]) {
        setCapturedImage(result.assets[0]);
        getCurrentLocation();
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to select image from gallery');
    }
  };

  const retakePicture = () => {
    setCapturedImage(null);
    setLocation(null);
    setLocationAddress(null);
  };

  const submitReport = async () => {
    if (!capturedImage) {
      Alert.alert('Error', 'Image is required');
      return;
    }

    if (!location) {
      Alert.alert('Error', 'Location is required');
      return;
    }

    setIsSubmitting(true);

    try {
      setTimeout(() => {
        Alert.alert(
          'Success',
          'Your report has been submitted',
          [
            {
              text: 'OK',
              onPress: () => {
                retakePicture();
                navigation.navigate('Reports');
              }
            }
          ]
        );
        setIsSubmitting(false);
      }, 1500);
    } catch (error) {
      console.error('Error submitting report:', error);
      Alert.alert('Error', 'Network error while submitting report');
      setIsSubmitting(false);
    }
  };

  if (cameraPermission === undefined || hasLocationPermission === null) {
    return (
      <SafeAreaView style={[styles.container, styles.centered, { backgroundColor: theme.colors.background }]}>
        <StatusBar barStyle="dark-content" backgroundColor={theme.colors.background} />
        <PaperActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>
          Requesting permissions...
        </Text>
      </SafeAreaView>
    );
  }

  if (!cameraPermission.granted) {
    return (
      <SafeAreaView style={[styles.container, styles.centered, { backgroundColor: theme.colors.background }]}>
        <StatusBar barStyle="dark-content" backgroundColor={theme.colors.background} />
        <Surface style={[styles.permissionCard, { backgroundColor: theme.colors.surface }]} elevation={4}>
          <MaterialCommunityIcons 
            name="camera-off" 
            size={64} 
            color={theme.colors.outline} 
            style={styles.permissionIcon}
          />
          <Title style={[styles.permissionTitle, { color: theme.colors.text }]}>
            Camera Access Required
          </Title>
          <Paragraph style={[styles.noPermissionText, { color: theme.colors.textSecondary }]}>
            Camera permission is required to use this feature. Please enable camera access in your device settings.
          </Paragraph>
          <Button
            mode="contained"
            onPress={() => navigation.navigate('Reports')}
            style={[styles.permissionButton, { backgroundColor: theme.colors.primary }]}
            contentStyle={styles.buttonContent}
          >
            Go to Reports
          </Button>
        </Surface>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: capturedImage ? theme.colors.background : '#000' }]} edges={capturedImage ? ['top'] : []}>
      <StatusBar barStyle={capturedImage ? "light-content" : "light-content"} backgroundColor={capturedImage ? theme.colors.primary : "#000"} />
      
      {capturedImage ? (
        <View style={[styles.capturedContainer, { backgroundColor: theme.colors.background }]}>
          {/* Header */}
          <ConsistentHeader
            title="Review & Submit"
            useGradient={true}
            elevated={true}
            centered={true}
            back={{
              visible: true,
              onPress: () => navigation.goBack()
            }}
          />

          <View style={styles.contentContainer}>
            <Animatable.View animation="fadeInUp" delay={200}>
              <Surface style={[styles.imageCard, { backgroundColor: theme.colors.surface }]} elevation={4}>
                <Image
                  source={{ uri: capturedImage.uri }}
                  style={styles.capturedImage}
                  resizeMode="cover"
                />
              </Surface>
            </Animatable.View>

            <Animatable.View animation="fadeInUp" delay={400}>
              {fetchingLocation ? (
                <Surface style={[styles.locationCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.locationContent}>
                    <PaperActivityIndicator size="small" color={theme.colors.primary} />
                    <Text style={[styles.locationText, { color: theme.colors.text }]}>
                      Getting precise location...
                    </Text>
                  </View>
                </Surface>
              ) : location ? (
                <Surface style={[styles.locationCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.locationHeader}>
                    <View style={styles.locationContent}>
                      <MaterialCommunityIcons name="map-marker" size={20} color={theme.colors.success} />
                      <Text style={[styles.locationText, { color: theme.colors.text }]}>
                        {locationAddress || 'Address not available'}
                      </Text>
                    </View>
                    <IconButton
                      icon="refresh"
                      size={20}
                      iconColor={theme.colors.primary}
                      onPress={getCurrentLocation}
                      disabled={fetchingLocation}
                    />
                  </View>
                  {location.addressDetails && (
                    <View style={styles.coordinatesContainer}>
                      <Text style={[styles.coordinatesText, { color: theme.colors.outline }]}>
                        {`${location.coords.latitude.toFixed(6)}, ${location.coords.longitude.toFixed(6)}`}
                      </Text>
                      <Chip 
                        icon="crosshairs-gps" 
                        textStyle={{ fontSize: 12 }}
                        style={{ backgroundColor: theme.colors.primaryContainer }}
                      >
                        Accuracy: ±{Math.round(location.coords.accuracy)}m
                      </Chip>
                    </View>
                  )}
                </Surface>
              ) : (
                <Surface style={[styles.locationCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.locationHeader}>
                    <View style={styles.locationContent}>
                      <MaterialCommunityIcons name="alert-circle" size={20} color={theme.colors.error} />
                      <Text style={[styles.locationText, { color: theme.colors.text }]}>
                        Failed to get location
                      </Text>
                    </View>
                    <IconButton
                      icon="refresh"
                      size={20}
                      iconColor={theme.colors.primary}
                      onPress={getCurrentLocation}
                      disabled={fetchingLocation}
                    />
                  </View>
                </Surface>
              )}
            </Animatable.View>

            <Animatable.View animation="fadeInUp" delay={600}>
              <View style={styles.buttonRow}>
                <Button
                  mode="outlined"
                  onPress={retakePicture}
                  style={[styles.retakeButton, { borderColor: theme.colors.primary }]}
                  textColor={theme.colors.primary}
                  disabled={isSubmitting}
                  icon="camera-retake"
                  contentStyle={styles.buttonContent}
                >
                  Retake
                </Button>

                <Button
                  mode="contained"
                  onPress={submitReport}
                  style={[styles.submitButton, { backgroundColor: theme.colors.primary }]}
                  loading={isSubmitting}
                  disabled={isSubmitting || fetchingLocation || !location}
                  icon="send"
                  contentStyle={styles.buttonContent}
                >
                  Submit Report
                </Button>
              </View>
            </Animatable.View>
          </View>
        </View>
      ) : (
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={cameraFacing}
            ratio="16:9"
          >
            {/* Instructions overlay */}
            <LinearGradient
              colors={['rgba(0,0,0,0.6)', 'transparent']}
              style={styles.instructions}
            >
              <Surface style={[styles.instructionSurface, { backgroundColor: theme.colors.surface + 'E6' }]} elevation={2}>
                <MaterialCommunityIcons name="camera" size={20} color={theme.colors.primary} />
                <Text style={[styles.instructionText, { color: theme.colors.text }]}>
                  Take a clear photo of the road damage
                </Text>
              </Surface>
            </LinearGradient>
            
            {/* Camera controls */}
            <View style={styles.cameraControls}>
              <TouchableOpacity
                style={[styles.flipButton, { backgroundColor: theme.colors.surface + 'CC' }]}
                onPress={toggleCameraFacing}
              >
                <MaterialCommunityIcons name="camera-flip" size={28} color={theme.colors.primary} />
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.captureButton, { borderColor: theme.colors.surface }]}
                onPress={takePicture}
              >
                <View style={[styles.captureButtonInner, { backgroundColor: theme.colors.surface }]} />
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.galleryButton, { backgroundColor: theme.colors.surface + 'CC' }]}
                onPress={pickImage}
              >
                <MaterialCommunityIcons name="image" size={28} color={theme.colors.primary} />
              </TouchableOpacity>
            </View>
          </CameraView>
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
  },
  permissionCard: {
    padding: 24,
    borderRadius: 12,
    width: '90%',
    alignItems: 'center',
  },
  permissionIcon: {
    marginBottom: 16,
  },
  permissionTitle: {
    marginBottom: 12,
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  noPermissionText: {
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 20,
  },
  permissionButton: {
    borderRadius: 12,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  capturedContainer: {
    flex: 1,
  },
  capturedHeader: {
    paddingTop: StatusBar.currentHeight || 44,
    paddingBottom: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 16,
  },
  contentContainer: {
    flex: 1,
    padding: 16,
  },
  imageCard: {
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
  },
  capturedImage: {
    width: '100%',
    height: 300,
    borderRadius: 12,
  },
  locationCard: {
    marginBottom: 20,
    borderRadius: 12,
    padding: 16,
  },
  locationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  locationContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  locationText: {
    fontSize: 14,
    flex: 1,
  },
  coordinatesContainer: {
    marginTop: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  coordinatesText: {
    fontSize: 12,
    fontFamily: 'monospace',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  retakeButton: {
    flex: 1,
    borderRadius: 12,
  },
  submitButton: {
    flex: 2,
    borderRadius: 12,
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  instructions: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 120,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 60,
  },
  instructionSurface: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 8,
  },
  instructionText: {
    fontSize: 14,
    fontWeight: '500',
  },
  cameraControls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  flipButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  galleryButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
});

export default CameraScreen;

