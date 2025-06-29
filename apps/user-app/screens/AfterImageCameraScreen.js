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
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { 
  Button, 
  useTheme,
  IconButton,
  Snackbar
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { ConsistentHeader } from '../components/ui';
import { uploadAfterImage } from '../utils/taskAPI';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const AfterImageCameraScreen = ({ navigation, route }) => {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const theme = useTheme();
  const cameraRef = useRef(null);
  
  const { reportId, taskTitle } = route.params;
  
  const [capturedImage, setCapturedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [cameraFacing, setCameraFacing] = useState('back');
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // Request camera permissions on mount
  useEffect(() => {
    requestCameraPermissions();
  }, []);

  const requestCameraPermissions = async () => {
    if (!cameraPermission?.granted) {
      await requestCameraPermission();
    }
  };

  const takePicture = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
        skipProcessing: false,
      });
      
      setCapturedImage(photo.uri);
    } catch (error) {
      console.error('Error taking picture:', error);
      Alert.alert('Error', 'Failed to take picture. Please try again.');
    }
  };

  const selectFromGallery = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (!permissionResult.granted) {
        Alert.alert('Permission Required', 'Please grant permission to access photo library.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setCapturedImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select image from gallery.');
    }
  };

  const retakeImage = () => {
    setCapturedImage(null);
  };

  const uploadImage = async () => {
    if (!capturedImage) return;

    setIsUploading(true);
    try {
      await uploadAfterImage(reportId, capturedImage);
      
      setSnackbarMessage('After image uploaded successfully!');
      setSnackbarVisible(true);
      
      // Call completion callback if provided
      if (route.params?.onComplete) {
        route.params.onComplete();
      }
      
      // Navigate back after successful upload
      setTimeout(() => {
        navigation.goBack();
      }, 2000);
    } catch (error) {
      console.error('Error uploading after image:', error);
      Alert.alert('Upload Error', error.message || 'Failed to upload after image. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const toggleCameraFacing = () => {
    setCameraFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  // Show permission request screen if camera permission not granted
  if (!cameraPermission?.granted) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />
        
        <ConsistentHeader
          title="Camera Permission"
          subtitle="Permission required to capture after image"
          useGradient={true}
          back={{
            visible: true,
            onPress: () => navigation.goBack()
          }}
        />

        <View style={styles.permissionContainer}>
          <MaterialIcons name="camera-alt" size={80} color={theme.colors.primary} />
          <Text style={[styles.permissionTitle, { color: theme.colors.text }]}>
            Camera Access Required
          </Text>
          <Text style={[styles.permissionMessage, { color: theme.colors.textSecondary }]}>
            We need access to your camera to capture the after image of the completed repair.
          </Text>
          <Button
            mode="contained"
            onPress={requestCameraPermission}
            style={[styles.permissionButton, { backgroundColor: theme.colors.primary }]}
          >
            Grant Camera Permission
          </Button>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />
      
      <ConsistentHeader
        title="Capture After Image"
        subtitle={`Task: ${taskTitle}`}
        useGradient={true}
        back={{
          visible: true,
          onPress: () => navigation.goBack()
        }}
      />

      {capturedImage ? (
        // Image Preview Screen
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.previewImage} />
          
          <View style={styles.previewActions}>
            <Animatable.View animation="slideInUp" duration={600}>
              <Button
                mode="outlined"
                onPress={retakeImage}
                style={[styles.actionButton, { borderColor: theme.colors.primary }]}
                labelStyle={{ color: theme.colors.primary }}
                icon="camera-retake"
              >
                Retake
              </Button>
            </Animatable.View>
            
            <Animatable.View animation="slideInUp" duration={600} delay={100}>
              <Button
                mode="contained"
                onPress={uploadImage}
                loading={isUploading}
                disabled={isUploading}
                style={[styles.actionButton, { backgroundColor: theme.colors.success }]}
                labelStyle={{ color: '#ffffff' }}
                icon="upload"
              >
                {isUploading ? 'Uploading...' : 'Upload Image'}
              </Button>
            </Animatable.View>
          </View>
        </View>
      ) : (
        // Camera Screen
        <View style={styles.cameraContainer}>
          <CameraView 
            ref={cameraRef}
            style={styles.camera}
            facing={cameraFacing}
          />
          
          {/* Camera overlay */}
          <View style={styles.cameraOverlay}>
            <View style={styles.topControls}>
              <Text style={styles.instructionText}>
                Capture the repaired area to complete the task
              </Text>
            </View>
            
            <View style={styles.bottomControls}>
              <TouchableOpacity
                style={[styles.galleryButton, { backgroundColor: theme.colors.surface }]}
                onPress={selectFromGallery}
              >
                <Ionicons name="images" size={24} color={theme.colors.primary} />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.captureButton, { borderColor: '#ffffff' }]}
                onPress={takePicture}
              >
                <View style={[styles.captureButtonInner, { backgroundColor: '#ffffff' }]} />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.flipButton, { backgroundColor: theme.colors.surface }]}
                onPress={toggleCameraFacing}
              >
                <Ionicons name="camera-reverse" size={24} color={theme.colors.primary} />
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}

      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => setSnackbarVisible(false)}
        duration={3000}
        style={{ backgroundColor: theme.colors.success }}
      >
        {snackbarMessage}
      </Snackbar>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: '700',
    textAlign: 'center',
    marginTop: 24,
    marginBottom: 16,
  },
  permissionMessage: {
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  permissionButton: {
    paddingHorizontal: 24,
    paddingVertical: 8,
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    paddingVertical: 50,
  },
  topControls: {
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  instructionText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 20,
    overflow: 'hidden',
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  galleryButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
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
    width: 64,
    height: 64,
    borderRadius: 32,
  },
  flipButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewContainer: {
    flex: 1,
    justifyContent: 'space-between',
  },
  previewImage: {
    flex: 1,
    width: '100%',
    resizeMode: 'cover',
  },
  previewActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
  },
  actionButton: {
    minWidth: 140,
    paddingVertical: 8,
  },
});

export default AfterImageCameraScreen;
