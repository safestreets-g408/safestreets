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
  ScrollView,
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
import { aiServices } from '../utils/aiServices';
import { useAuth } from '../context/AuthContext';
import { submitNewReport, submitAiReport } from '../utils/reports';
// Road validation now uses the AI server

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const CameraScreen = ({ navigation }) => {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [hasLocationPermission, setHasLocationPermission] = useState(null);
  const theme = useTheme();
  const { fieldWorker } = useAuth();
  const [capturedImage, setCapturedImage] = useState(null);
  const [location, setLocation] = useState(null);
  const [locationAddress, setLocationAddress] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [cameraFacing, setCameraFacing] = useState('back');
  const [fetchingLocation, setFetchingLocation] = useState(false);
  
  // AI Analysis states
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [showAnnotatedImage, setShowAnnotatedImage] = useState(false);
  
  // Road classification states
  const [roadValidation, setRoadValidation] = useState(null);
  const [isValidatingRoad, setIsValidatingRoad] = useState(false);
  const [roadValidationError, setRoadValidationError] = useState(null);
  
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
          
          // Immediately fetch location when app opens
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

  const validateRoadImage = async (imageUri) => {
    try {
      setIsValidatingRoad(true);
      setRoadValidationError(null);
      
      console.log('Starting road validation for image:', imageUri);
      
      // Use the AI server's road classification endpoint
      const base64Image = await aiServices.imageToBase64(imageUri);
      
      const response = await fetch(`${await aiServices.getAiServerUrl()}/classify-road`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          confidenceThreshold: 0.6
        })
      });
      
      if (!response.ok) {
        throw new Error(`AI server error: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Road classification failed');
      }
      
      const validation = {
        isValid: result.isRoad,
        confidence: result.confidence,
        message: result.message,
        classification: result.features
      };
      
      setRoadValidation(validation);
      
      console.log('Road validation result:', {
        isValid: validation.isValid,
        confidence: validation.confidence,
        message: validation.message
      });
      
      return validation;
    } catch (error) {
      console.error('Road validation failed:', error);
      setRoadValidationError(error.message);
      
      // Return a permissive result to avoid blocking the user
      const fallbackValidation = {
        isValid: true,
        confidence: 0.5,
        message: 'Unable to validate image, proceeding with analysis',
        classification: { error: error.message }
      };
      
      setRoadValidation(fallbackValidation);
      return fallbackValidation;
    } finally {
      setIsValidatingRoad(false);
    }
  };

  // Analysis flow now happens directly in pickImage and takePicture
  // First road validation, then damage analysis if valid

  const proceedWithDamageAnalysis = async (imageUri, locationDetails) => {
    try {
      // Ensure a valid location string that isn't "Unknown location"
      let formattedAddress = 'Unspecified road location';

      // Try to get location from different sources in order of preference
      if (locationDetails?.formattedAddress && 
          locationDetails.formattedAddress !== 'Unknown location' &&
          locationDetails.formattedAddress !== 'Address not available' &&
          locationDetails.formattedAddress !== 'Address pending...') {
        formattedAddress = locationDetails.formattedAddress;
      } else if (locationAddress && 
                locationAddress !== 'Unknown location' && 
                locationAddress !== 'Address not available') {
        formattedAddress = locationAddress;
      } else if (location?.addressDetails?.city) {
        // Construct a basic location from city/region if available
        formattedAddress = [
          location.addressDetails.city,
          location.addressDetails.region,
          location.addressDetails.country
        ].filter(Boolean).join(', ');
      }

      console.log('Starting AI analysis with location:', formattedAddress);
      
      // Create a safe location object with fallbacks
      const locationToUse = {
        formattedAddress: formattedAddress,
        coordinates: locationDetails?.coordinates || 
          (location?.coords ? [location.coords.longitude, location.coords.latitude] : undefined)
      };
      
      // Use YOLOv8 directly through the analyzeRoadDamage function
      // which now uses YOLOv8 internally instead of the VIT model
      const analysis = await aiServices.analyzeRoadDamage(imageUri, locationToUse);
      
      // Store the analysis result
      setAiAnalysis(analysis);
      
      console.log('YOLOv8 AI analysis completed:', {
        damageClass: analysis?.classification?.damageClass,
        damageType: analysis?.classification?.damageType,
        severity: analysis?.classification?.severity,
        hasAnnotatedImage: !!analysis?.classification?.annotatedImage,
        summary: analysis?.summary?.substring(0, 50) + '...'
      });
      
    } catch (error) {
      console.error('Damage analysis failed:', error);
      throw error; // Re-throw to be handled by the calling function
    }
  };

  const takePicture = async () => {
    if (!cameraRef.current) {
      Alert.alert('Error', 'Camera is not ready. Please wait or restart the app.');
      return;
    }

    try {
      // Start location fetching immediately when user takes picture
      if (hasLocationPermission && (!location || !locationAddress)) {
        try {
          await getCurrentLocation();
        } catch (err) {
          console.log('Location fetch error:', err);
        }
      }
      
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        skipProcessing: false,
        exif: true, // Include EXIF data which may contain GPS info
      });

      if (photo && photo.uri) {
        setCapturedImage(photo);
        
        // First validate if the image contains a road
        try {
          setIsValidatingRoad(true);
          const roadValidationResult = await validateRoadImage(photo.uri);
          setIsValidatingRoad(false);
          
          // If road validation fails, show warning and ask user if they want to proceed
          if (!roadValidationResult.isValid) {
            Alert.alert(
              'Road Surface Not Detected',
              roadValidationResult.message + '\n\nWould you like to proceed anyway or take a new photo?',
              [
                { text: 'Take New Photo', style: 'cancel', onPress: () => {
                  retakePicture();
                  return;
                }},
                { text: 'Proceed Anyway', onPress: () => {
                  // Continue with analysis using whatever location we have
                  const locationData = location?.addressDetails || {
                    formattedAddress: locationAddress || 'Address being determined...',
                    coordinates: location?.coords ? 
                      [location.coords.longitude, location.coords.latitude] : undefined
                  };
                  proceedWithDamageAnalysis(photo.uri, locationData);
                }}
              ]
            );
            return;
          }
          
          // If road validation passes, proceed with damage analysis
          const locationData = location?.addressDetails || {
            formattedAddress: locationAddress || 'Address being determined...',
            coordinates: location?.coords ? 
              [location.coords.longitude, location.coords.latitude] : undefined
          };
          proceedWithDamageAnalysis(photo.uri, locationData);
          
        } catch (error) {
          console.error('Error in road validation:', error);
          // If road validation fails, still proceed with damage analysis
          const locationData = location?.addressDetails || {
            formattedAddress: locationAddress || 'Address being determined...',
            coordinates: location?.coords ? 
              [location.coords.longitude, location.coords.latitude] : undefined
          };
          proceedWithDamageAnalysis(photo.uri, locationData);
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
        
        // Start location fetch if needed
        if (hasLocationPermission && (!location || !locationAddress)) {
          try {
            await getCurrentLocation();
          } catch (err) {
            console.log('Location fetch error:', err);
          }
        }

        // First validate if the image contains a road
        try {
          setIsValidatingRoad(true);
          const roadValidationResult = await validateRoadImage(result.assets[0].uri);
          setIsValidatingRoad(false);
          
          // If road validation fails, show warning and ask user if they want to proceed
          if (!roadValidationResult.isValid) {
            Alert.alert(
              'Road Surface Not Detected',
              roadValidationResult.message + '\n\nWould you like to proceed anyway or take a new photo?',
              [
                { text: 'Take New Photo', style: 'cancel', onPress: () => {
                  retakePicture();
                  return;
                }},
                { text: 'Proceed Anyway', onPress: () => {
                  // Continue with analysis using whatever location we have
                  const locationData = location?.addressDetails || {
                    formattedAddress: locationAddress || 'Address being determined...',
                    coordinates: location?.coords ? 
                      [location.coords.longitude, location.coords.latitude] : undefined
                  };
                  proceedWithDamageAnalysis(result.assets[0].uri, locationData);
                }}
              ]
            );
            return;
          }
          
          // If road validation passes, proceed with damage analysis
          const locationData = location?.addressDetails || {
            formattedAddress: locationAddress || 'Address being determined...',
            coordinates: location?.coords ? 
              [location.coords.longitude, location.coords.latitude] : undefined
          };
          proceedWithDamageAnalysis(result.assets[0].uri, locationData);
          
        } catch (error) {
          console.error('Error in road validation:', error);
          // If road validation fails, still proceed with damage analysis
          const locationData = location?.addressDetails || {
            formattedAddress: locationAddress || 'Address being determined...',
            coordinates: location?.coords ? 
              [location.coords.longitude, location.coords.latitude] : undefined
          };
          proceedWithDamageAnalysis(result.assets[0].uri, locationData);
        }
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to select image from gallery');
    }
  };

  const retakePicture = () => {
    // Clear image and analysis data
    setCapturedImage(null);
    setAiAnalysis(null);
    setIsAnalyzing(false);
    setAnalysisError(null);
    setShowAnnotatedImage(false);
    
    // Clear road validation data
    setRoadValidation(null);
    setIsValidatingRoad(false);
    setRoadValidationError(null);
    
    // Optionally clear location - only if user wants fresh location
    // Could keep location data for faster re-submission
    if (fetchingLocation) {
      // If a location fetch is in progress, cancel it by resetting state
      setFetchingLocation(false);
      setLocation(null);
      setLocationAddress(null);
    }
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

    if (!fieldWorker) {
      Alert.alert('Error', 'User authentication required');
      return;
    }

    // If AI is still analyzing, ask user if they want to wait
    if (isAnalyzing) {
      Alert.alert(
        'AI Analysis in Progress',
        'AI is still analyzing your image. Would you like to wait for the analysis to complete?',
        [
          { text: 'Wait', style: 'cancel' },
          { 
            text: 'Submit Without AI', 
            onPress: () => {
              setIsAnalyzing(false); 
              submitReportInternal();
            } 
          }
        ]
      );
      return;
    }

    submitReportInternal();
  };
  
  const submitReportInternal = async () => {
    setIsSubmitting(true);

    try {
      // Ensure we have the most up-to-date location data
      let currentAddress = locationAddress || 'Address not available';
      
      // CRITICAL: Ensure currentAddress never contains AI analysis text
      if (currentAddress && (currentAddress.includes('Road damage report') || 
                           currentAddress.includes('observed on') || 
                           currentAddress.includes('damage') ||
                           currentAddress.length > 200)) {
        console.warn('Detected AI analysis text in currentAddress, using fallback');
        currentAddress = location?.addressDetails?.formattedAddress || 'Address lookup failed';
      }
      
      const coordString = location?.coords ? 
        `${location.coords.latitude},${location.coords.longitude}` : '';

      console.log('Report submission debug:', {
        locationAddress,
        currentAddress,
        locationAddressDetails: location?.addressDetails?.formattedAddress,
        aiSummary: aiAnalysis?.summary?.substring(0, 50) + '...'
      });

      // Clean up AI data to ensure it's valid
      let cleanAiData = null;
      if (aiAnalysis) {
        try {
          // Create a clean version of the classification without the image
          // The annotated image will be stored separately in the AI report
          const cleanClassification = {
            damageClass: aiAnalysis.classification.damageClass,
            damageType: aiAnalysis.classification.damageType,
            severity: aiAnalysis.classification.severity,
            // Don't include the annotated image here to keep the JSON smaller
          };
          
          cleanAiData = JSON.stringify({
            classification: cleanClassification,
            summary: aiAnalysis.summary,
            analysisTimestamp: new Date().toISOString()
          });
        } catch (aiDataError) {
          console.error('Error preparing AI data:', aiDataError);
        }
      }

      // Prepare submission data with AI analysis results and tenant information
      const reportData = {
        // Send proper location object structure that backend expects
        location: {
          coordinates: location?.coords ? [location.coords.longitude, location.coords.latitude] : undefined,
          address: currentAddress
        },
        damageType: aiAnalysis?.classification?.damageType || 'Road Damage',
        severity: aiAnalysis?.classification?.severity || 'Medium',
        priority: aiAnalysis?.classification?.severity === 'High' ? 'High' : 
                 aiAnalysis?.classification?.severity === 'Medium' ? 'Medium' : 'Low',
        description: aiAnalysis?.summary || 'Damage report submitted via mobile app',
        // Include AI analysis metadata
        aiGenerated: !!aiAnalysis,
        aiData: cleanAiData
      };

      console.log('Submitting report with AI analysis and tenant context:', {
        hasAiAnalysis: !!aiAnalysis,
        damageType: aiAnalysis?.classification?.damageType,
        severity: aiAnalysis?.classification?.severity, 
        location: currentAddress,
        tenantId: fieldWorker.tenant
      });

      // Prepare final submission data based on whether we have AI analysis or not
      let submittedReport = null;
      let retryCount = 0;
      const maxRetries = 2;
      
      // If we have AI analysis, submit directly as an AI report with the annotated image
      if (aiAnalysis) {
        try {
          // Validate that annotated image is a proper base64 string
          let annotatedImage = null;
          if (aiAnalysis?.classification?.annotatedImage) {
            // Check if it's a valid base64 string (at least somewhat long and doesn't contain invalid chars)
            const isValidBase64 = aiAnalysis.classification.annotatedImage.length > 100 && 
                                  !/[^A-Za-z0-9+/=]/.test(aiAnalysis.classification.annotatedImage);
            
            if (isValidBase64) {
              annotatedImage = aiAnalysis.classification.annotatedImage;
              console.log('Valid annotated image detected, length:', annotatedImage.length);
              
              if (annotatedImage.length > 900000) {
                console.warn('Annotated image is large:', Math.round(annotatedImage.length/1024), 'KB, may cause issues');
              }
            } else {
              console.log('Invalid annotated image detected, using original image instead');
            }
          }
          
          // Prepare location data carefully - use the actual address from location data
          let locationData = {};
          if (location?.coords) {
            // Use the proper formatted address from location.addressDetails
            let actualAddress = location?.addressDetails?.formattedAddress || locationAddress || 'Location data available without address';
            
            // Safety check: ensure the address doesn't contain AI analysis text
            if (actualAddress && (actualAddress.includes('Road damage report') || actualAddress.includes('observed on') || actualAddress.length > 200)) {
              console.warn('Detected AI analysis text in address, using fallback');
              actualAddress = location?.addressDetails?.formattedAddress || 'Address data corrupted, using coordinates only';
            }
            
            locationData = {
              coordinates: location?.coords ? [location.coords.longitude, location.coords.latitude] : undefined, // MongoDB uses [longitude, latitude] order
              address: actualAddress
            };
          } else {
            // Fallback to locationAddress (the reverse geocoded address)
            let fallbackAddress = locationAddress || 'Address not available';
            
            // Safety check for fallback address too
            if (fallbackAddress && (fallbackAddress.includes('Road damage report') || fallbackAddress.includes('observed on') || fallbackAddress.length > 200)) {
              console.warn('Detected AI analysis text in fallback address');
              fallbackAddress = 'Address lookup failed';
            }
            
            locationData = {
              address: fallbackAddress
            };
          }
          
          console.log('AI Report location data:', {
            locationAddress,
            formattedAddress: location?.addressDetails?.formattedAddress,
            finalAddress: locationData.address,
            hasCoordinates: !!locationData.coordinates
          });
          
          // Create AI report object with all the necessary data
          const aiReport = {
            // Required core fields for AiReport model
            tenant: fieldWorker.tenant,
            predictionClass: aiAnalysis?.classification?.damageClass || 'Unknown',
            damageType: aiAnalysis?.classification?.damageType || 'Road Damage',
            severity: aiAnalysis?.classification?.severity || 'Medium',
            priority: aiAnalysis?.classification?.severity === 'High' ? 8 : aiAnalysis?.classification?.severity === 'Medium' ? 5 : 3,
            
            // Use the annotated image if available, otherwise use the original image
            annotatedImageBase64: annotatedImage || imageBase64 || '',
            
            // Location data structured as expected by the model
            location: locationData,
            
            // Additional fields for compatibility
            description: reportData.description || '',
            region: reportData.region || 'Default Region',
            createdAt: new Date().toISOString()
          };
          
          console.log('Report data prepared for submission:', {
            damageType: aiReport.damageType,
            severity: aiReport.severity,
            hasAnnotatedImage: !!(aiReport.annotatedImageBase64),
            imageSize: aiReport.annotatedImageBase64 ? `${Math.round(aiReport.annotatedImageBase64.length / 1024)}KB` : 'N/A'
          });
          
          // Submit the AI report with retries
          while (retryCount <= maxRetries) {
            try {
              // Use the submitAiReport function which properly formats the data for the /ai-reports endpoint
              submittedReport = await submitAiReport(aiReport);
              
              if (submittedReport && (submittedReport._id || submittedReport.id)) {
                console.log('AI report submitted successfully with ID:', submittedReport._id || submittedReport.id);
                break; // Success, exit the retry loop
              } else {
                throw new Error('Server returned empty response');
              }
            } catch (submitError) {
              console.error(`AI report submission attempt ${retryCount + 1} failed:`, submitError.message);
              
              if (retryCount === maxRetries) {
                throw new Error(`Failed to submit report after ${maxRetries + 1} attempts: ${submitError.message}`);
              }
              
              // Wait before retrying (exponential backoff)
              await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retryCount)));
              retryCount++;
            }
          }
        } catch (aiError) {
          console.error('Error submitting AI report, falling back to standard report:', aiError.message);
          // Reset retry count to try standard submission
          retryCount = 0;
          // Fall through to standard report submission as backup
        }
      } 
      
      // If no AI analysis or AI submission failed, submit as a standard damage report
      if (!submittedReport) {
        while (retryCount <= maxRetries) {
          try {
            submittedReport = await submitNewReport(reportData, capturedImage.uri);
            
            if (submittedReport && (submittedReport._id || submittedReport.id)) {
              console.log('Standard report submitted successfully with ID:', submittedReport._id || submittedReport.id);
              break; // Success, exit the retry loop
            } else {
              throw new Error('Server returned empty response');
            }
          } catch (submitError) {
            console.error(`Standard report submission attempt ${retryCount + 1} failed:`, submitError.message);
            
            if (retryCount === maxRetries) {
              throw new Error(`Failed to submit report after ${maxRetries + 1} attempts: ${submitError.message}`);
            }
            
            // Wait before retrying (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retryCount)));
            retryCount++;
          }
        }
      }

      Alert.alert(
        'Success',
        aiAnalysis 
          ? `Your report has been submitted with YOLOv8 AI analysis. Detected: ${aiAnalysis.classification.damageType} (${aiAnalysis.classification.severity} severity)`
          : 'Your report has been submitted successfully',
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
    } catch (error) {
      console.error('Error submitting report:', error);
      Alert.alert('Error', 'Network error while submitting report: ' + error.message);
    } finally {
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

          <ScrollView 
            style={styles.contentContainer}
            contentContainerStyle={styles.scrollContentContainer}
            showsVerticalScrollIndicator={false}
          >
            <Animatable.View animation="fadeInUp" delay={200}>
              <Surface style={[styles.imageCard, { backgroundColor: theme.colors.surface }]} elevation={4}>
                {aiAnalysis?.classification?.annotatedImage && (
                  <View style={styles.imageToggleContainer}>
                    <Button
                      mode="contained"
                      compact
                      onPress={() => setShowAnnotatedImage(!showAnnotatedImage)}
                      style={styles.imageToggleButton}
                      icon={showAnnotatedImage ? "image-outline" : "image-edit"}
                    >
                      {showAnnotatedImage ? 'Original Image' : 'AI Detection'}
                    </Button>
                  </View>
                )}
                {isAnalyzing && (
                  <View style={styles.imageAnalyzingOverlay}>
                    <PaperActivityIndicator size={40} color={theme.colors.primary} />
                    <Text style={styles.imageAnalyzingText}>Analyzing damage...</Text>
                  </View>
                )}
                <Image
                  source={{ 
                    uri: showAnnotatedImage && aiAnalysis?.classification?.annotatedImage 
                         ? `data:image/jpeg;base64,${aiAnalysis.classification.annotatedImage}`
                         : capturedImage.uri 
                  }}
                  style={styles.capturedImage}
                  resizeMode="cover"
                  onError={(error) => {
                    console.error('Image loading error:', error.nativeEvent?.error || 'Unknown image error');
                    // Detect base64 issues that could break rendering
                    const hasInvalidBase64 = showAnnotatedImage && 
                      (!aiAnalysis?.classification?.annotatedImage || 
                       aiAnalysis.classification.annotatedImage.length < 100);
                       
                    // If annotated image fails to load, fall back to original
                    if (showAnnotatedImage) {
                      setShowAnnotatedImage(false);
                      Alert.alert(
                        'Image Display Issue',
                        hasInvalidBase64 
                          ? 'The AI detection image data is incomplete or invalid. Showing original image instead.'
                          : 'Could not load AI detection overlay. Showing original image instead.'
                      );
                    }
                  }}
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

            {/* Road Validation Section */}
            <Animatable.View animation="fadeInUp" delay={450}>
              {isValidatingRoad ? (
                <Surface style={[styles.roadValidationCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.roadValidationHeader}>
                    <MaterialCommunityIcons name="road" size={24} color={theme.colors.primary} />
                    <Text style={[styles.roadValidationTitle, { color: theme.colors.text }]}>
                      Road Detection
                    </Text>
                  </View>
                  <View style={styles.roadValidationContent}>
                    <PaperActivityIndicator size="small" color={theme.colors.primary} />
                    <Text style={[styles.roadValidationText, { color: theme.colors.text }]}>
                      Checking if image contains a road...
                    </Text>
                  </View>
                </Surface>
              ) : roadValidation ? (
                <Surface style={[styles.roadValidationCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.roadValidationHeader}>
                    <MaterialCommunityIcons 
                      name={roadValidation.isValid ? "road" : "road-variant"} 
                      size={24} 
                      color={roadValidation.isValid ? theme.colors.success : theme.colors.warning} 
                    />
                    <Text style={[styles.roadValidationTitle, { color: theme.colors.text }]}>
                      Road Detection
                    </Text>
                    <Chip 
                      icon={roadValidation.isValid ? "check-circle" : "alert-circle"}
                      style={{ 
                        backgroundColor: roadValidation.isValid 
                          ? theme.colors.tertiaryContainer 
                          : theme.colors.errorContainer 
                      }}
                      textStyle={{ fontSize: 10 }}
                    >
                      {(roadValidation.confidence * 100).toFixed(0)}% confidence
                    </Chip>
                  </View>
                  
                  <Text style={[styles.roadValidationMessage, { 
                    color: roadValidation.isValid ? theme.colors.text : theme.colors.error 
                  }]}>
                    {roadValidation.message}
                  </Text>
                  
                  {!roadValidation.isValid && (
                    <Button
                      mode="outlined"
                      onPress={() => validateRoadImage(capturedImage.uri)}
                      style={{ marginTop: 8 }}
                      compact
                      icon="refresh"
                    >
                      Re-analyze
                    </Button>
                  )}
                </Surface>
              ) : null}
            </Animatable.View>

            {/* AI Analysis Section */}
            <Animatable.View animation="fadeInUp" delay={500}>
              {isAnalyzing ? (
                <Surface style={[styles.aiAnalysisCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.aiAnalysisHeader}>
                    <MaterialCommunityIcons name="brain" size={24} color={theme.colors.primary} />
                    <Text style={[styles.aiAnalysisTitle, { color: theme.colors.text }]}>
                      AI Analysis
                    </Text>
                  </View>
                  <View style={styles.aiAnalysisContent}>
                    <PaperActivityIndicator size="small" color={theme.colors.primary} />
                    <Text style={[styles.aiAnalysisText, { color: theme.colors.text }]}>
                      Analyzing damage...
                    </Text>
                  </View>
                </Surface>
              ) : aiAnalysis ? (
                <Surface style={[styles.aiAnalysisCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.aiAnalysisHeader}>
                    <MaterialCommunityIcons name="brain" size={24} color={theme.colors.success} />
                    <Text style={[styles.aiAnalysisTitle, { color: theme.colors.text }]}>
                      AI Analysis Complete
                    </Text>
                  </View>
                  
                  <View style={styles.aiResultsContainer}>
                    <View style={styles.damageChipsContainer}>
                      <Chip 
                        icon="road-variant" 
                        style={{ backgroundColor: theme.colors.primaryContainer }}
                        textStyle={{ fontSize: 12 }}
                      >
                        {aiAnalysis.classification.damageType}
                      </Chip>
                      <Chip 
                        icon="alert-circle" 
                        style={{ 
                          backgroundColor: aiAnalysis.classification.severity === 'High' 
                            ? theme.colors.errorContainer 
                            : aiAnalysis.classification.severity === 'Medium'
                            ? theme.colors.onSurfaceVariant + '20'
                            : theme.colors.tertiaryContainer 
                        }}
                        textStyle={{ fontSize: 12 }}
                      >
                        {aiAnalysis.classification.severity} Severity
                      </Chip>
                    </View>
                    
                    <Text style={[styles.aiSummaryText, { color: theme.colors.text }]}>
                      {aiAnalysis.summary}
                    </Text>
                  </View>
                </Surface>
              ) : analysisError ? (
                <Surface style={[styles.aiAnalysisCard, { backgroundColor: theme.colors.surface }]} elevation={2}>
                  <View style={styles.aiAnalysisHeader}>
                    <MaterialCommunityIcons name="brain" size={24} color={theme.colors.error} />
                    <Text style={[styles.aiAnalysisTitle, { color: theme.colors.text }]}>
                      AI Analysis Failed
                    </Text>
                  </View>
                  <View style={styles.aiAnalysisContent}>
                    <Text style={[styles.aiAnalysisText, { color: theme.colors.error }]}>
                      {analysisError}
                    </Text>
                    <Button
                      mode="outlined"
                      onPress={() => {
                        // First validate the road, then proceed with damage analysis
                        validateRoadImage(capturedImage.uri).then(result => {
                          const locationData = location?.addressDetails || {
                            formattedAddress: locationAddress || 'Address being determined...',
                            coordinates: location?.coords ? 
                              [location.coords.longitude, location.coords.latitude] : undefined
                          };
                          proceedWithDamageAnalysis(capturedImage.uri, locationData);
                        });
                      }}
                      style={{ marginTop: 8 }}
                      compact
                    >
                      Retry Analysis
                    </Button>
                  </View>
                </Surface>
              ) : null}
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
                  disabled={isSubmitting || fetchingLocation || !location || isValidatingRoad}
                  icon="send"
                  contentStyle={styles.buttonContent}
                >
                  {isValidatingRoad ? 'Validating...' : isAnalyzing ? 'Analyzing...' : 'Submit Report'}
                </Button>
              </View>
            </Animatable.View>
          </ScrollView>
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
                  Take a clear photo - AI will analyze the damage automatically
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
  },
  scrollContentContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  imageCard: {
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
    position: 'relative',
  },
  capturedImage: {
    width: '100%',
    height: 300,
    borderRadius: 12,
  },
  imageToggleContainer: {
    position: 'absolute',
    top: 10,
    right: 10,
    zIndex: 10,
  },
  imageToggleButton: {
    borderRadius: 8,
    opacity: 0.85,
  },
  imageAnalyzingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    zIndex: 5,
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageAnalyzingText: {
    color: 'white',
    marginTop: 10,
    fontWeight: '600',
    textShadowColor: 'rgba(0,0,0,0.75)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
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
  aiAnalysisCard: {
    marginBottom: 20,
    borderRadius: 12,
    padding: 16,
  },
  aiAnalysisHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  aiAnalysisTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  aiAnalysisContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  aiAnalysisText: {
    fontSize: 14,
    flex: 1,
  },
  aiResultsContainer: {
    gap: 12,
  },
  damageChipsContainer: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  aiSummaryText: {
    fontSize: 14,
    lineHeight: 20,
    fontStyle: 'italic',
  },
  roadValidationCard: {
    marginBottom: 16,
    borderRadius: 12,
    padding: 16,
  },
  roadValidationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  roadValidationTitle: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  roadValidationContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  roadValidationText: {
    fontSize: 14,
    flex: 1,
  },
  roadValidationMessage: {
    fontSize: 14,
    lineHeight: 18,
    marginBottom: 4,
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

