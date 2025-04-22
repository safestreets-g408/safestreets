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
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { Button, Card, Title, Paragraph, Surface } from 'react-native-paper';

const CameraScreen = ({ navigation }) => {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [hasLocationPermission, setHasLocationPermission] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [location, setLocation] = useState(null);
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
        
        const { status: locationStatus } = await Location.requestForegroundPermissionsAsync();
        setHasLocationPermission(locationStatus === 'granted');

        if (!cameraPermission?.granted) {
          Alert.alert('Permission required', 'Camera permission is needed to take photos');
        }

        if (locationStatus !== 'granted') {
          Alert.alert('Permission required', 'Location permission is needed to tag your reports');
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
        const position = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.High
        });
        setLocation(position);
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

  const takePicture = async () => {
    if (!cameraRef.current) {
      Alert.alert('Error', 'Camera is not ready. Please wait or restart the app.');
      return;
    }

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        skipProcessing: false,
      });

      if (photo && photo.uri) {
        setCapturedImage(photo);
        getCurrentLocation();
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
      // Here we would normally upload the image and data to a server
      // For now, we're simulating a successful submission
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
      <SafeAreaView style={[styles.container, styles.centered]}>
        <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Requesting permissions...</Text>
      </SafeAreaView>
    );
  }

  if (!cameraPermission.granted) {
    return (
      <SafeAreaView style={[styles.container, styles.centered]}>
        <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
        <Surface style={styles.permissionCard}>
          <Title style={styles.permissionTitle}>Camera Access Required</Title>
          <Paragraph style={styles.noPermissionText}>
            Camera permission is required to use this feature. Please enable camera access in your device settings.
          </Paragraph>
          <Button
            mode="contained"
            onPress={() => navigation.navigate('Reports')}
            style={styles.permissionButton}
          >
            Go to Reports
          </Button>
        </Surface>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle={capturedImage ? "dark-content" : "light-content"} backgroundColor={capturedImage ? "#f5f5f5" : "#000"} />
      {capturedImage ? (
        <View style={styles.capturedContainer}>
          <Card style={styles.imageCard}>
            <Card.Cover
              source={{ uri: capturedImage.uri }}
              style={styles.capturedImage}
            />
          </Card>

          {fetchingLocation ? (
            <Card style={styles.locationCard}>
              <Card.Content>
                <View style={styles.locationContent}>
                  <ActivityIndicator size="small" color="#3498db" style={{marginRight: 10}} />
                  <Paragraph style={styles.locationText}>
                    Fetching location...
                  </Paragraph>
                </View>
              </Card.Content>
            </Card>
          ) : location ? (
            <Card style={styles.locationCard}>
              <Card.Content>
                <View style={styles.locationContent}>
                  <Ionicons name="location" size={20} color="#3498db" />
                  <Paragraph style={styles.locationText}>
                    {location.coords.latitude.toFixed(6)}, {location.coords.longitude.toFixed(6)}
                  </Paragraph>
                </View>
              </Card.Content>
            </Card>
          ) : (
            <Card style={styles.locationCard}>
              <Card.Content>
                <View style={styles.locationContent}>
                  <Ionicons name="warning" size={20} color="#e74c3c" />
                  <Paragraph style={styles.locationText}>
                    Failed to get location
                  </Paragraph>
                </View>
              </Card.Content>
            </Card>
          )}

          <View style={styles.buttonRow}>
            <Button
              mode="outlined"
              onPress={retakePicture}
              style={styles.retakeButton}
              disabled={isSubmitting}
              icon="camera-retake"
            >
              Retake
            </Button>

            <Button
              mode="contained"
              onPress={submitReport}
              style={styles.submitButton}
              loading={isSubmitting}
              disabled={isSubmitting || fetchingLocation || !location}
              icon="send"
            >
              Submit Report
            </Button>
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
            <View style={styles.instructions}>
              <Surface style={styles.instructionSurface}>
                <Text style={styles.instructionText}>
                  Take a clear photo of the road damage
                </Text>
              </Surface>
            </View>
            
            <View style={styles.cameraControls}>
              <TouchableOpacity
                style={styles.flipButton}
                onPress={toggleCameraFacing}
              >
                <MaterialIcons name="flip-camera-android" size={28} color="white" />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.captureButton}
                onPress={takePicture}
              >
                <View style={styles.captureButtonInner} />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.galleryButton}
                onPress={pickImage}
              >
                <MaterialIcons name="photo-library" size={28} color="white" />
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
    backgroundColor: '#f5f5f5',
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#555',
  },
  permissionCard: {
    padding: 24,
    borderRadius: 12,
    elevation: 4,
    width: '90%',
    alignItems: 'center',
  },
  permissionTitle: {
    marginBottom: 12,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  permissionButton: {
    marginTop: 20,
    paddingVertical: 8,
    width: '100%',
    backgroundColor: '#3498db',
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  cameraControls: {
    position: 'absolute',
    bottom: 30,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    borderColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 65,
    height: 65,
    borderRadius: 32.5,
    backgroundColor: 'white',
  },
  flipButton: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 12,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  galleryButton: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 12,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  capturedContainer: {
    flex: 1,
    padding: 16,
  },
  imageCard: {
    elevation: 4,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
  },
  capturedImage: {
    height: 250,
    borderRadius: 0,
  },
  locationCard: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 8,
  },
  locationContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  locationText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#555',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  retakeButton: {
    flex: 1,
    marginRight: 8,
    borderColor: '#3498db',
    borderWidth: 1,
  },
  submitButton: {
    flex: 1.5,
    marginLeft: 8,
    backgroundColor: '#3498db',
  },
  instructions: {
    position: 'absolute',
    top: 20,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  instructionSurface: {
    padding: 12,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  instructionText: {
    color: 'white',
    fontWeight: '500',
  },
  noPermissionText: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
    color: '#555',
    lineHeight: 22,
  },
});

export default CameraScreen;