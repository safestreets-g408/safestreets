import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Typography,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Paper,
  Grid,
  Container,
  Chip,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import AssessmentIcon from '@mui/icons-material/Assessment';
import { api } from '../utils/api';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const AiAnalysis = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [reports, setReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [locationStatus, setLocationStatus] = useState('');
  const [locationData, setLocationData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('vit'); // Default to ViT model
  const [modelInfo, setModelInfo] = useState(null);
  const [loadingModelInfo, setLoadingModelInfo] = useState(false);

  // Fetch YOLO model info when component mounts
  useEffect(() => {
    const fetchYoloModelInfo = async () => {
      setLoadingModelInfo(true);
      try {
        // Fetch model info through our backend API
        const data = await api.get('/ai/model-info?type=yolo');
        setModelInfo(data.modelInfo);
        console.log('YOLO model info:', data.modelInfo);
      } catch (err) {
        console.error('Error fetching YOLO model info:', err);
      } finally {
        setLoadingModelInfo(false);
      }
    };

    fetchYoloModelInfo();
  }, []);

  // Check AI server status when the model selection changes
  useEffect(() => {
    if (selectedModel === 'yolo') {
      const checkServer = async () => {
        try {
          console.log('Checking AI server status for YOLO model...');
          const status = await api.checkAiServer();
          console.log('AI server status check result:', status);
          
          if (!status.success) {
            setError('AI server may not be accessible. YOLO model might not work correctly.');
          } else {
            console.log('AI server is accessible');
            // Clear any previous errors about the server
            if (error && error.includes('AI server')) {
              setError(null);
            }
          }
        } catch (err) {
          console.warn('Could not check AI server status:', err);
          setError('Could not verify AI server status. YOLO detection may not work properly.');
        }
      };
      
      checkServer();
    }
  }, [selectedModel, error]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    if (newValue === 1) {
      fetchReports();
    }
  };

  const fetchReports = async () => {
    setReportsLoading(true);
    try {
      const data = await api.get('/images/reports');
      setReports(data.reports);
    } catch (err) {
      setError(err.message);
    } finally {
      setReportsLoading(false);
    }
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const getLocation = () => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        setLocationStatus('Geolocation is not supported by your browser');
        reject('Geolocation not supported');
        return;
      }

      setLocationStatus('Fetching location...');
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const { latitude, longitude } = position.coords;
            
            // Get address using reverse geocoding
            try {
              const response = await fetch(
                `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
              );
              const data = await response.json();
              
              const locationInfo = {
                latitude,
                longitude,
                address: data.display_name || `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`
              };
              
              setLocationData(locationInfo);
              setLocationStatus('Location fetched successfully');
              resolve(locationInfo);
            } catch (error) {
              console.error('Error getting address:', error);
              const locationInfo = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                address: `${position.coords.latitude.toFixed(6)}, ${position.coords.longitude.toFixed(6)}`
              };
              setLocationData(locationInfo);
              setLocationStatus('Location coordinates fetched (address lookup failed)');
              resolve(locationInfo);
            }
          } catch (error) {
            console.error('Error processing location:', error);
            setLocationStatus('Error processing location data');
            reject(error);
          }
        },
        (error) => {
          let errorMsg = 'Unable to fetch location';
          switch (error.code) {
            case error.PERMISSION_DENIED:
              errorMsg = 'Location access denied. Please enable location permissions.';
              break;
            case error.POSITION_UNAVAILABLE:
              errorMsg = 'Location information is unavailable.';
              break;
            case error.TIMEOUT:
              errorMsg = 'Location request timed out.';
              break;
            default:
              errorMsg = 'An unknown error occurred while fetching location.';
          }
          setLocationStatus(errorMsg);
          console.error('Error getting location:', error);
          reject(error);
        },
        { 
          maximumAge: 60000,        // Use cached position if less than 1 minute old
          timeout: 10000,           // Wait up to 10 seconds
          enableHighAccuracy: false // Don't need high accuracy, save battery
        }
      );
    });
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    
    // Check if AI server is accessible first
    if (selectedModel === 'yolo') {
      try {
        console.log('Verifying AI server connection before YOLO analysis...');
        const serverStatus = await api.checkAiServer();
        if (!serverStatus.success) {
          setLoading(false);
          setError('AI server is not accessible. Please ensure the AI server is running before analyzing images with YOLO.');
          return;
        }
        console.log('AI server is online, proceeding with analysis');
      } catch (err) {
        console.warn('Could not check AI server status:', err);
        // We'll continue anyway and let the main operation handle errors
      }
    }

    try {
      // Get location first
      let locationInfo = null;
      try {
        locationInfo = await getLocation();
      } catch (error) {
        console.warn('Could not get location:', error);
      }

      console.log('Processing image:', {
        name: selectedImage.name,
        type: selectedImage.type,
        size: selectedImage.size,
        model: selectedModel
      });

      if (selectedModel === 'yolo') {
        // Create form data for YOLO model
        // Check if the image is too large
        const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB
        if (selectedImage.size > MAX_IMAGE_SIZE) {
          throw new Error(`Image size (${(selectedImage.size / 1024 / 1024).toFixed(2)}MB) exceeds the 5MB limit. Please select a smaller image.`);
        }

        // Check if the image is in a supported format
        const supportedFormats = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!supportedFormats.includes(selectedImage.type)) {
          throw new Error(`Image format ${selectedImage.type} is not supported. Please use JPEG or PNG images.`);
        }

        console.log('Creating form data for YOLO analysis with image:', {
          name: selectedImage.name,
          type: selectedImage.type,
          size: `${(selectedImage.size / 1024).toFixed(2)} KB`
        });
        
        const formData = new FormData();
        formData.append('image', selectedImage);
        
        // Add location data if available
        if (locationInfo) {
          formData.append('latitude', locationInfo.latitude.toString());
          formData.append('longitude', locationInfo.longitude.toString());
          if (locationInfo.address) {
            formData.append('address', locationInfo.address);
          }
        }

        try {
          console.log('Sending image for YOLO analysis...');
          // Use the backend API to process the image with YOLO model
          const response = await api.postFormData('/images/analyze-yolo', formData);
          
          console.log('Response from YOLO analysis:', response);
          
          // Check if we have any detections, even if success is false
          const hasDetections = response.detections && Array.isArray(response.detections);
          const hasAnnotatedImage = response.annotated_image || response.annotatedImage;
          
          if (response.success || hasDetections || hasAnnotatedImage) {
            // Format YOLO detection results for display
            const detectionCount = response.detections?.length || 0;
            
            // Determine severity and priority based on number and types of detections
            let severity = 'LOW';
            let priority = 3;
            
            if (detectionCount > 5) {
              severity = 'HIGH';
              priority = 8;
            } else if (detectionCount > 0) {
              severity = 'MEDIUM';
              priority = 5;
            }
            
            // Find the most significant detection (highest confidence)
            let primaryDetection = 'None';
            let highestConfidence = 0;
            
            if (response.detections && response.detections.length > 0) {
              response.detections.forEach(detection => {
                const confidence = detection.confidence || detection.score || 0;
                if (confidence > highestConfidence) {
                  highestConfidence = confidence;
                  primaryDetection = detection.class || detection.name;
                }
              });
            }
            
            setPrediction({
              damageType: primaryDetection !== 'None' ? `Object: ${primaryDetection}` : 'YOLO Detection',
              severity: severity,
              priority: priority,
              annotatedImage: response.annotated_image || response.annotatedImage,
              detections: response.detections || [],
              detectionCount: detectionCount,
              model: 'yolo',
              fallback: !!response.fallback
            });
            
            // Show warning if fallback was used
            if (response.fallback) {
              setError('Using fallback detection. Some features may be limited.');
            } else {
              console.log('YOLO Response processed successfully:', response);
            }
          } else {
            console.error('YOLO API returned error:', response);
            throw new Error(response.message || 'Failed to process image with YOLO');
          }
        } catch (yoloErr) {
          console.error('Error with YOLO analysis:', yoloErr);
          
          // Check if the error response contains any usable data
          if (yoloErr.data && (yoloErr.data.detections || yoloErr.data.annotated_image)) {
            console.log('Found usable data in error response, attempting to display it');
            
            // Determine severity and priority based on number and types of detections
            const detectionCount = yoloErr.data.detections?.length || 0;
            let severity = 'LOW';
            let priority = 3;
            
            if (detectionCount > 5) {
              severity = 'HIGH';
              priority = 8;
            } else if (detectionCount > 0) {
              severity = 'MEDIUM';
              priority = 5;
            }
            
            // Find the most significant detection
            let primaryDetection = 'None';
            let highestConfidence = 0;
            
            if (yoloErr.data.detections && yoloErr.data.detections.length > 0) {
              yoloErr.data.detections.forEach(detection => {
                const confidence = detection.confidence || detection.score || 0;
                if (confidence > highestConfidence) {
                  highestConfidence = confidence;
                  primaryDetection = detection.class || detection.name;
                }
              });
            }
            
            setPrediction({
              damageType: primaryDetection !== 'None' ? `Object: ${primaryDetection}` : 'YOLO Detection (Partial)',
              severity: severity,
              priority: priority,
              annotatedImage: yoloErr.data.annotated_image || yoloErr.data.annotatedImage,
              detections: yoloErr.data.detections || [],
              detectionCount: detectionCount,
              model: 'yolo',
              fallback: true
            });
            
            // Show warning but don't throw error
            setError(`Warning: ${yoloErr.message}. Displaying partial results.`);
            return; // Don't throw error since we have something to show
          }
          
          // Check if the error might be related to the AI server being down
          if (yoloErr.message.includes('connect') || 
              yoloErr.message.includes('ECONNREFUSED') ||
              yoloErr.message.includes('Network Error') ||
              yoloErr.message.includes('timeout')) {
            throw new Error('Could not connect to the AI server. Please ensure it is running and try again.');
          } else if (yoloErr.status === 404) {
            throw new Error('YOLO API endpoint not found. Please check server configuration.');
          } else if (yoloErr.message.includes('413') || yoloErr.status === 413) {
            throw new Error('Image is too large. Please try a smaller image (under 5MB).');
          } else {
            throw new Error(`YOLO processing error: ${yoloErr.message}`);
          }
        }
      } else {
        // Default ViT model using our backend
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('model', selectedModel); // Add selected model to form data
        
        // Add location data if available
        if (locationInfo) {
          formData.append('latitude', locationInfo.latitude.toString());
          formData.append('longitude', locationInfo.longitude.toString());
          if (locationInfo.address) {
            formData.append('address', locationInfo.address);
          }
        }

        // Log the form data to ensure it's properly set
        console.log('Form data entries:');
        for (let [key, value] of formData.entries()) {
          console.log(`${key}:`, typeof value === 'object' ? 'File object' : value);
        }

        // Use the new postFormData method instead of post
        const response = await api.postFormData('/images/upload', formData);

        if (response.success) {
          setPrediction({
            ...response.prediction,
            model: 'vit'
          });
          console.log('Response:', response);
        } else {
          throw new Error(response.message || 'Failed to process image');
        }
      }
    } catch (err) {
      console.error('Error analyzing image:', err);
      setError(err.message || 'An error occurred while analyzing the image');
    } finally {
      setLoading(false);
    }
  };

  // Reports are now rendered directly in the JSX of TabPanel with index 1

  // Add saveReport function for both ViT and YOLO models
  const saveReport = async () => {
    if (!prediction) return;
    
    setLoading(true);
    try {
      let reportData = {};
      
      if (prediction.model === 'yolo') {
        // YOLO report data
        reportData = {
          damageType: prediction.damageType,
          severity: prediction.severity,
          priority: prediction.priority,
          annotatedImage: prediction.annotatedImage,
          detections: prediction.detections,
          model: 'yolo',
          location: locationData
        };
      } else {
        // ViT report data
        reportData = {
          damageType: prediction.damageType,
          severity: prediction.severity,
          priority: prediction.priority,
          location: locationData,
          model: 'vit'
        };
      }
      
      const response = await api.post('/images/save-report', reportData);
      if (response.success) {
        setError(null);
        alert('Report saved successfully!');
        // Refresh reports list if we're on that tab
        if (tabValue === 1) {
          fetchReports();
        }
      } else {
        throw new Error(response.message || 'Failed to save report');
      }
    } catch (err) {
      console.error('Error saving report:', err);
      setError(`Failed to save report: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ width: '100%', py: 3 }}>
        
        {locationStatus && (
          <Alert 
            severity={locationStatus.includes('successfully') ? 'success' : 'info'}
            variant="filled"
            sx={{ 
              mb: 2.5,
              borderRadius: 2,
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
            }}
          >
            {locationStatus}
          </Alert>
        )}

        <Box sx={{ 
          mb: 3,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap'
        }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            aria-label="AI analysis tabs"
            sx={{
              '& .MuiTab-root': { 
                py: 1,
                textTransform: 'none', 
                fontSize: '0.85rem',
                fontWeight: 500,
                minHeight: '40px',
                letterSpacing: '0.01em'
              },
              '& .Mui-selected': {
                fontWeight: 600
              },
              '& .MuiTabs-indicator': {
                height: 3,
                borderTopLeftRadius: 3,
                borderTopRightRadius: 3
              }
            }}
          >
            <Tab 
              label="Upload Image" 
              icon={<CloudUploadIcon sx={{ fontSize: '1.2rem' }} />} 
              iconPosition="start"
            />
            <Tab 
              label="View Reports" 
              icon={<AssessmentIcon sx={{ fontSize: '1.2rem' }} />} 
              iconPosition="start"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Paper 
            elevation={0} 
            sx={{ 
              borderRadius: 3, 
              overflow: 'hidden',
              boxShadow: theme => theme.palette.mode === 'dark' 
                ? '0 4px 20px rgba(0,0,0,0.4)' 
                : '0 4px 20px rgba(0,0,0,0.08)'
            }}
          >
            <Box sx={{ p: 4 }}>
              {/* Image Upload Section */}
              <Box 
                sx={{ 
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 3
                }}
              >
                <Button
                  component="label"
                  role={undefined}
                  variant="contained"
                  tabIndex={-1}
                  startIcon={<ImageIcon />}
                  size="large"
                  sx={{
                    borderRadius: 2,
                    py: 1.5,
                    px: 4,
                    textTransform: 'none',
                    fontSize: '1rem',
                    fontWeight: 600,
                    boxShadow: '0 4px 12px rgba(25, 118, 210, 0.3)',
                    '&:hover': {
                      boxShadow: '0 6px 16px rgba(25, 118, 210, 0.4)'
                    }
                  }}
                >
                  Choose Image
                  <VisuallyHiddenInput 
                    type="file" 
                    accept="image/*"
                    onChange={handleImageSelect}
                  />
                </Button>

                {error && (
                  <Alert 
                    severity="error" 
                    variant="filled"
                    sx={{ 
                      width: '100%', 
                      mt: 2,
                      borderRadius: 2,
                      boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
                    }}
                  >
                    {error}
                  </Alert>
                )}

                {selectedImage && (
                  <Box sx={{ mt: 3, width: '100%' }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        mb: 2,
                        fontWeight: 600,
                        fontSize: '1.1rem',
                        color: theme => theme.palette.primary.main
                      }}
                    >
                      Selected Image
                    </Typography>
                    <Box
                      sx={{
                        width: '100%',
                        height: 350,
                        backgroundImage: `url(${previewUrl})`,
                        backgroundSize: 'contain',
                        backgroundPosition: 'center',
                        backgroundRepeat: 'no-repeat',
                        borderRadius: 3,
                        overflow: 'hidden',
                        boxShadow: theme => theme.palette.mode === 'dark' 
                          ? '0 4px 20px rgba(0,0,0,0.3)' 
                          : '0 4px 20px rgba(0,0,0,0.1)',
                        position: 'relative',
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          background: 'rgba(0,0,0,0.03)',
                          pointerEvents: 'none'
                        }
                      }}
                    />
                    
                    {/* Model Selection */}
                    <Box sx={{ 
                      mt: 4, 
                      mb: 3,
                      p: 3,
                      borderRadius: 2,
                      bgcolor: theme => theme.palette.mode === 'dark' 
                        ? 'rgba(255,255,255,0.03)' 
                        : 'rgba(0,0,0,0.02)',
                      border: '1px solid',
                      borderColor: theme => theme.palette.mode === 'dark' 
                        ? 'rgba(255,255,255,0.1)' 
                        : 'rgba(0,0,0,0.05)'
                    }}>
                      <FormControl component="fieldset" sx={{ width: '100%' }}>
                        <FormLabel 
                          component="legend"
                          sx={{
                            fontWeight: 600,
                            fontSize: '0.9rem',
                            mb: 1.5,
                            color: theme => theme.palette.mode === 'dark' 
                              ? theme.palette.primary.light 
                              : theme.palette.primary.dark
                          }}
                        >
                          Select AI Model
                        </FormLabel>
                        <RadioGroup
                          row
                          aria-label="ai-model"
                          name="ai-model"
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                        >
                          <FormControlLabel 
                            value="vit" 
                            control={
                              <Radio 
                                sx={{
                                  '&.Mui-checked': {
                                    color: theme => theme.palette.primary.main
                                  }
                                }}
                              />
                            } 
                            label={
                              <Typography sx={{ fontWeight: 500 }}>
                                Vision Transformer (ViT)
                              </Typography>
                            } 
                            sx={{ mr: 4 }}
                          />
                          <FormControlLabel 
                            value="yolo" 
                            control={
                              <Radio 
                                sx={{
                                  '&.Mui-checked': {
                                    color: theme => theme.palette.primary.main
                                  }
                                }}
                              />
                            } 
                            label={
                              <Typography sx={{ fontWeight: 500 }}>
                                YOLO v8 (Object Detection)
                              </Typography>
                            } 
                          />
                        </RadioGroup>
                        
                        {selectedModel === 'yolo' && (
                          <Box sx={{ mt: 1.5, ml: 0.5 }}>
                            {loadingModelInfo ? (
                              <Box sx={{ 
                                display: 'flex', 
                                alignItems: 'center',
                                p: 1,
                                borderRadius: 1,
                                bgcolor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(255,255,255,0.05)' 
                                  : 'rgba(0,0,0,0.02)'
                              }}>
                                <CircularProgress size={16} sx={{ mr: 1.5 }} />
                                <Typography variant="body2">Loading model information...</Typography>
                              </Box>
                            ) : modelInfo ? (
                              <Box sx={{ 
                                p: 1.5, 
                                borderRadius: 2,
                                bgcolor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(25, 118, 210, 0.15)' 
                                  : 'rgba(25, 118, 210, 0.08)',
                              }}>
                                <Typography variant="body2" sx={{ fontWeight: 500, color: theme => theme.palette.primary.main }}>
                                  YOLO model: {modelInfo.classes?.length || '0'} classes, 
                                  Confidence threshold: {modelInfo.confidence_threshold || 'N/A'}
                                </Typography>
                              </Box>
                            ) : (
                              <Typography 
                                variant="body2" 
                                color="error" 
                                sx={{ 
                                  display: 'block',
                                  p: 1,
                                  borderRadius: 1,
                                  bgcolor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(255,0,0,0.1)' 
                                    : 'rgba(255,0,0,0.05)'
                                }}
                              >
                                Failed to load model information
                              </Typography>
                            )}
                          </Box>
                        )}
                      </FormControl>
                    </Box>
                    
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={analyzeImage}
                      disabled={loading}
                      size="large"
                      sx={{ 
                        mt: 2,
                        py: 1.5,
                        px: 4,
                        borderRadius: 2,
                        textTransform: 'none',
                        fontSize: '1rem',
                        fontWeight: 600,
                        boxShadow: '0 4px 12px rgba(25, 118, 210, 0.2)',
                        '&:hover': {
                          boxShadow: '0 6px 16px rgba(25, 118, 210, 0.3)'
                        }
                      }}
                    >
                      {loading ? (
                        <>
                          <CircularProgress size={24} sx={{ mr: 1.5 }} color="inherit" />
                          Analyzing Image...
                        </>
                      ) : (
                        'Analyze Image'
                      )}
                    </Button>
                  </Box>
                )}

                {locationData && (
                  <Box sx={{ 
                    mt: 4, 
                    width: '100%',
                    p: 2.5,
                    borderRadius: 3,
                    bgcolor: theme => theme.palette.mode === 'dark' 
                      ? 'rgba(0,150,136,0.08)' 
                      : 'rgba(0,150,136,0.04)',
                    border: '1px solid',
                    borderColor: theme => theme.palette.mode === 'dark' 
                      ? 'rgba(0,150,136,0.15)' 
                      : 'rgba(0,150,136,0.15)',
                    boxShadow: theme => theme.palette.mode === 'dark'
                      ? 'none'
                      : '0 2px 8px rgba(0,0,0,0.04)'
                  }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        mb: 2, 
                        color: 'success.main',
                        fontWeight: 600,
                        fontSize: '1rem',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                    >
                      <Box 
                        component="span" 
                        sx={{ 
                          display: 'inline-block',
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: 'success.main',
                          mr: 1.5
                        }}
                      /> Location Information
                    </Typography>
                    <Grid container spacing={2.5}>
                      <Grid item xs={12} sm={6}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            color: 'text.secondary', 
                            fontWeight: 500,
                            mb: 0.5,
                            fontSize: '0.75rem'
                          }}
                        >
                          LATITUDE
                        </Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {locationData.latitude.toFixed(6)}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            color: 'text.secondary', 
                            fontWeight: 500,
                            mb: 0.5,
                            fontSize: '0.75rem'
                          }}
                        >
                          LONGITUDE
                        </Typography>
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {locationData.longitude.toFixed(6)}
                        </Typography>
                      </Grid>
                      {locationData.address && (
                        <Grid item xs={12}>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              color: 'text.secondary', 
                              fontWeight: 500,
                              mb: 0.5,
                              fontSize: '0.75rem'
                            }}
                          >
                            ADDRESS
                          </Typography>
                          <Typography 
                            variant="body1" 
                            sx={{ 
                              fontWeight: 500, 
                              lineHeight: 1.5,
                              p: 1.5,
                              borderRadius: 1,
                              bgcolor: theme => theme.palette.mode === 'dark' 
                                ? 'rgba(255,255,255,0.03)' 
                                : 'rgba(0,0,0,0.03)'
                            }}
                          >
                            {locationData.address}
                          </Typography>
                        </Grid>
                      )}
                    </Grid>
                  </Box>
                )}

                {prediction && (
                  <Box sx={{ 
                    mt: 5, 
                    width: '100%',
                    bgcolor: theme => theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.8)',
                    borderRadius: 4,
                    overflow: 'hidden',
                    boxShadow: '0 6px 24px rgba(0,0,0,0.08)',
                    border: '1px solid',
                    borderColor: theme => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'
                  }}>
                    <Box sx={{ 
                      p: 2.5, 
                      pb: 2,
                      bgcolor: theme => theme.palette.mode === 'dark' 
                        ? 'rgba(25, 118, 210, 0.2)' 
                        : 'rgba(25, 118, 210, 0.05)',
                      borderBottom: '1px solid',
                      borderColor: theme => theme.palette.mode === 'dark' 
                        ? 'rgba(255,255,255,0.1)' 
                        : 'rgba(25, 118, 210, 0.15)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}>
                      <Typography 
                        variant="h5" 
                        sx={{ 
                          fontWeight: 600, 
                          color: theme => theme.palette.mode === 'dark' 
                            ? theme.palette.primary.light 
                            : theme.palette.primary.main,
                          fontSize: '1.25rem'
                        }}
                      >
                        Analysis Results
                      </Typography>
                      <Chip 
                        label={prediction.model === 'yolo' 
                          ? (prediction.fallback ? 'YOLO Fallback Detection' : 'YOLO Detection') 
                          : 'ViT Classification'} 
                        color={prediction.fallback ? "warning" : "primary"}
                        size="small" 
                        sx={{ 
                          fontWeight: 600,
                          px: 1,
                          borderRadius: '12px',
                          '& .MuiChip-label': { px: 1 }
                        }}
                      />
                    </Box>
                    
                    {/* Display annotated image if available */}
                    <Box sx={{ px: 3, py: 3 }}>
                      {prediction.annotatedImage && (
                        <Box sx={{ mb: 4 }}>
                          <Typography 
                            variant="subtitle1" 
                            sx={{ 
                              mb: 2, 
                              fontWeight: 600, 
                              color: theme => theme.palette.mode === 'dark' 
                                ? theme.palette.primary.light 
                                : theme.palette.primary.main,
                              display: 'flex',
                              alignItems: 'center'
                            }}
                          >
                            <Box 
                              component="span" 
                              sx={{ 
                                display: 'inline-block',
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: 'primary.main',
                                mr: 1.5
                              }}
                            /> Annotated Image
                          </Typography>
                          <Box
                            sx={{
                              borderRadius: 3,
                              overflow: 'hidden',
                              boxShadow: theme => theme.palette.mode === 'dark' 
                                ? '0 8px 24px rgba(0,0,0,0.3)' 
                                : '0 8px 24px rgba(0,0,0,0.1)',
                              position: 'relative',
                              '&::after': {
                                content: '""',
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.1)',
                                borderRadius: 3,
                                pointerEvents: 'none'
                              }
                            }}
                          >
                            <Box
                              component="img"
                              src={typeof prediction.annotatedImage === 'string' && prediction.annotatedImage.startsWith('data:') 
                                ? prediction.annotatedImage 
                                : `data:image/jpeg;base64,${prediction.annotatedImage}`}
                              alt="Annotated"
                              sx={{
                                width: '100%',
                                maxHeight: '500px',
                                objectFit: 'contain',
                                display: 'block'
                              }}
                            />
                          </Box>
                        </Box>
                      )}
                      
                      {/* YOLO specific results */}
                      {prediction.model === 'yolo' && (
                        <>
                          <Box sx={{
                            mb: 4,
                            p: 3,
                            borderRadius: 3,
                            bgcolor: theme => theme.palette.mode === 'dark' 
                              ? 'rgba(0,0,0,0.2)' 
                              : 'rgba(0,0,0,0.02)',
                            border: '1px solid',
                            borderColor: theme => theme.palette.mode === 'dark' 
                              ? 'rgba(255,255,255,0.05)' 
                              : 'rgba(0,0,0,0.05)'
                          }}>
                            <Grid container spacing={3}>
                              <Grid item xs={12} sm={6} md={3}>
                                <Box sx={{ 
                                  p: 2.5, 
                                  bgcolor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(25, 118, 210, 0.15)' 
                                    : 'rgba(255,255,255,1)',
                                  borderRadius: 3,
                                  boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                  border: '1px solid',
                                  borderColor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(255,255,255,0.05)' 
                                    : 'rgba(25, 118, 210, 0.1)',
                                  height: '100%',
                                  display: 'flex',
                                  flexDirection: 'column'
                                }}>
                                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                    DAMAGE TYPE
                                  </Typography>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Chip 
                                      label={prediction.damageType} 
                                      color="primary" 
                                      size="small" 
                                      sx={{ 
                                        fontWeight: 600,
                                        px: 0.5,
                                        borderRadius: '8px',
                                        '& .MuiChip-label': { px: 1 }
                                      }} 
                                    />
                                  </Box>
                                </Box>
                              </Grid>
                              <Grid item xs={12} sm={6} md={3}>
                                <Box sx={{ 
                                  p: 2.5, 
                                  bgcolor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(211, 47, 47, 0.15)' 
                                    : 'rgba(255,255,255,1)',
                                  borderRadius: 3,
                                  boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                  border: '1px solid',
                                  borderColor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(255,255,255,0.05)' 
                                    : `rgba(${prediction.severity === 'HIGH' ? '211, 47, 47' : prediction.severity === 'MEDIUM' ? '237, 108, 2' : '46, 125, 50'}, 0.1)`,
                                  height: '100%',
                                  display: 'flex',
                                  flexDirection: 'column'
                                }}>
                                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                    SEVERITY
                                  </Typography>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Chip 
                                      label={prediction.severity} 
                                      color={prediction.severity === 'HIGH' ? "error" : prediction.severity === 'MEDIUM' ? "warning" : "success"} 
                                      size="small" 
                                      sx={{ 
                                        fontWeight: 600,
                                        px: 0.5,
                                        borderRadius: '8px',
                                        '& .MuiChip-label': { px: 1 }
                                      }} 
                                    />
                                  </Box>
                                </Box>
                              </Grid>
                              <Grid item xs={12} sm={6} md={3}>
                                <Box sx={{ 
                                  p: 2.5, 
                                  bgcolor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(237, 108, 2, 0.15)' 
                                    : 'rgba(255,255,255,1)',
                                  borderRadius: 3,
                                  boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                  border: '1px solid',
                                  borderColor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(255,255,255,0.05)' 
                                    : 'rgba(237, 108, 2, 0.1)',
                                  height: '100%',
                                  display: 'flex',
                                  flexDirection: 'column'
                                }}>
                                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                    PRIORITY
                                  </Typography>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Chip 
                                      label={prediction.priority} 
                                      color="warning" 
                                      size="small" 
                                      sx={{ 
                                        fontWeight: 600,
                                        px: 0.5,
                                        borderRadius: '8px',
                                        '& .MuiChip-label': { px: 1 }
                                      }} 
                                    />
                                  </Box>
                                </Box>
                              </Grid>
                              <Grid item xs={12} sm={6} md={3}>
                                <Box sx={{ 
                                  p: 2.5, 
                                  bgcolor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(3, 169, 244, 0.15)' 
                                    : 'rgba(255,255,255,1)',
                                  borderRadius: 3,
                                  boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                  border: '1px solid',
                                  borderColor: theme => theme.palette.mode === 'dark' 
                                    ? 'rgba(255,255,255,0.05)' 
                                    : 'rgba(3, 169, 244, 0.1)',
                                  height: '100%',
                                  display: 'flex',
                                  flexDirection: 'column'
                                }}>
                                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                    OBJECTS DETECTED
                                  </Typography>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Chip 
                                      label={prediction.detectionCount || 0} 
                                      color="info" 
                                      size="small" 
                                      sx={{ 
                                        fontWeight: 600,
                                        px: 0.5,
                                        borderRadius: '8px',
                                        '& .MuiChip-label': { px: 1 }
                                      }} 
                                    />
                                  </Box>
                                </Box>
                              </Grid>
                            </Grid>
                          </Box>
                          
                          {prediction.detections && prediction.detections.length > 0 && (
                            <Box sx={{ mb: 3 }}>
                              <Typography 
                                variant="subtitle1" 
                                sx={{ 
                                  mb: 2, 
                                  fontWeight: 600, 
                                  color: theme => theme.palette.mode === 'dark' 
                                    ? theme.palette.info.light 
                                    : theme.palette.info.dark,
                                  display: 'flex',
                                  alignItems: 'center'
                                }}
                              >
                                <Box 
                                  component="span" 
                                  sx={{ 
                                    display: 'inline-block',
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    bgcolor: 'info.main',
                                    mr: 1.5
                                  }}
                                /> Detection Details
                              </Typography>
                              <Paper 
                                variant="outlined" 
                                sx={{ 
                                  borderRadius: 3,
                                  overflow: 'hidden',
                                  maxHeight: '320px'
                                }}
                              >
                                <Box sx={{ 
                                  maxHeight: '320px', 
                                  overflow: 'auto',
                                  p: 0.5
                                }}>
                                  {prediction.detections.map((detection, index) => (
                                    <Box 
                                      key={index} 
                                      sx={{ 
                                        p: 2,
                                        mb: 1,
                                        borderRadius: 2,
                                        bgcolor: theme => theme.palette.mode === 'dark' 
                                          ? 'rgba(255,255,255,0.03)' 
                                          : 'rgba(0,0,0,0.01)',
                                        border: '1px solid',
                                        borderColor: theme => theme.palette.mode === 'dark' 
                                          ? 'rgba(255,255,255,0.05)' 
                                          : 'rgba(0,0,0,0.05)',
                                        '&:last-child': {
                                          mb: 0
                                        }
                                      }}
                                    >
                                      <Grid container spacing={2} alignItems="center">
                                        <Grid item xs={7}>
                                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mr: 1 }}>
                                              CLASS:
                                            </Typography>
                                            <Chip 
                                              label={detection.class || detection.name} 
                                              size="small" 
                                              color="primary" 
                                              sx={{ 
                                                fontWeight: 500,
                                                borderRadius: '6px',
                                                '& .MuiChip-label': { px: 1 }
                                              }}
                                            />
                                          </Box>
                                        </Grid>
                                        <Grid item xs={5}>
                                          <Box sx={{ 
                                            display: 'flex', 
                                            alignItems: 'center', 
                                            justifyContent: 'flex-end' 
                                          }}>
                                            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mr: 1 }}>
                                              CONFIDENCE:
                                            </Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                              {((detection.confidence || detection.score || 0) * 100).toFixed(1)}%
                                            </Typography>
                                          </Box>
                                        </Grid>
                                      </Grid>
                                    </Box>
                                  ))}
                                </Box>
                              </Paper>
                            </Box>
                          )}
                        </>
                      )}
                      
                      {/* ViT specific results */}
                      {prediction.model === 'vit' && (
                        <Box sx={{
                          mb: 4,
                          p: 3,
                          borderRadius: 3,
                          bgcolor: theme => theme.palette.mode === 'dark' 
                            ? 'rgba(0,0,0,0.2)' 
                            : 'rgba(0,0,0,0.02)',
                          border: '1px solid',
                          borderColor: theme => theme.palette.mode === 'dark' 
                            ? 'rgba(255,255,255,0.05)' 
                            : 'rgba(0,0,0,0.05)'
                        }}>
                          <Grid container spacing={3}>
                            <Grid item xs={12} sm={6} md={4}>
                              <Box sx={{ 
                                p: 2.5, 
                                bgcolor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(25, 118, 210, 0.15)' 
                                  : 'rgba(255,255,255,1)',
                                borderRadius: 3,
                                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                border: '1px solid',
                                borderColor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(255,255,255,0.05)' 
                                  : 'rgba(25, 118, 210, 0.1)',
                                height: '100%',
                                display: 'flex',
                                flexDirection: 'column'
                              }}>
                                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                  DAMAGE TYPE
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Chip 
                                    label={prediction.damageType} 
                                    color="primary" 
                                    size="small" 
                                    sx={{ 
                                      fontWeight: 600,
                                      px: 0.5,
                                      borderRadius: '8px',
                                      '& .MuiChip-label': { px: 1 }
                                    }} 
                                  />
                                </Box>
                              </Box>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4}>
                              <Box sx={{ 
                                p: 2.5, 
                                bgcolor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(211, 47, 47, 0.15)' 
                                  : 'rgba(255,255,255,1)',
                                borderRadius: 3,
                                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                border: '1px solid',
                                borderColor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(255,255,255,0.05)' 
                                  : 'rgba(211, 47, 47, 0.1)',
                                height: '100%',
                                display: 'flex',
                                flexDirection: 'column'
                              }}>
                                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                  SEVERITY
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Chip 
                                    label={prediction.severity} 
                                    color="error" 
                                    size="small" 
                                    sx={{ 
                                      fontWeight: 600,
                                      px: 0.5,
                                      borderRadius: '8px',
                                      '& .MuiChip-label': { px: 1 }
                                    }} 
                                  />
                                </Box>
                              </Box>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4}>
                              <Box sx={{ 
                                p: 2.5, 
                                bgcolor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(237, 108, 2, 0.15)' 
                                  : 'rgba(255,255,255,1)',
                                borderRadius: 3,
                                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                                border: '1px solid',
                                borderColor: theme => theme.palette.mode === 'dark' 
                                  ? 'rgba(255,255,255,0.05)' 
                                  : 'rgba(237, 108, 2, 0.1)',
                                height: '100%',
                                display: 'flex',
                                flexDirection: 'column'
                              }}>
                                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500, mb: 1 }}>
                                  PRIORITY
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Chip 
                                    label={prediction.priority} 
                                    color="warning" 
                                    size="small" 
                                    sx={{ 
                                      fontWeight: 600,
                                      px: 0.5,
                                      borderRadius: '8px',
                                      '& .MuiChip-label': { px: 1 }
                                    }} 
                                  />
                                </Box>
                              </Box>
                            </Grid>
                          </Grid>
                        </Box>
                      )}
                      
                      <Box sx={{ 
                        mt: 3,
                        display: 'flex', 
                        gap: 2.5,
                        justifyContent: 'flex-end'
                      }}>
                        <Button
                          variant="outlined"
                          color="secondary"
                          onClick={() => {
                            handleTabChange(null, 1);
                          }}
                          sx={{
                            borderRadius: 2,
                            py: 1.25,
                            px: 3,
                            textTransform: 'none',
                            fontWeight: 600,
                            fontSize: '0.95rem'
                          }}
                        >
                          View All Reports
                        </Button>
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={saveReport}
                          disabled={loading}
                          sx={{
                            borderRadius: 2,
                            py: 1.25,
                            px: 3,
                            textTransform: 'none',
                            fontWeight: 600,
                            fontSize: '0.95rem',
                            boxShadow: '0 4px 12px rgba(25, 118, 210, 0.2)',
                            '&:hover': {
                              boxShadow: '0 6px 14px rgba(25, 118, 210, 0.3)'
                            }
                          }}
                        >
                          {loading ? (
                            <>
                              <CircularProgress size={20} sx={{ mr: 1.5 }} color="inherit" />
                              Saving...
                            </>
                          ) : (
                            'Save Report'
                          )}
                        </Button>
                      </Box>
                    </Box>
                  </Box>
                )}
              </Box>
            </Box>
          </Paper>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ px: { xs: 1, sm: 3 }, py: 3 }}>
            {reportsLoading ? (
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center', 
                py: 8 
              }}>
                <CircularProgress color="primary" size={42} thickness={4} />
                <Typography 
                  variant="body1" 
                  sx={{ 
                    mt: 3, 
                    fontWeight: 500,
                    color: 'text.secondary'
                  }}
                >
                  Loading reports...
                </Typography>
              </Box>
            ) : reports.length === 0 ? (
              <Paper 
                elevation={0}
                sx={{ 
                  p: 5, 
                  py: 8,
                  borderRadius: 4, 
                  textAlign: 'center', 
                  bgcolor: theme => theme.palette.mode === 'dark' 
                    ? 'rgba(255,255,255,0.02)' 
                    : 'rgba(240,242,245,0.6)',
                  border: '1px dashed',
                  borderColor: theme => theme.palette.mode === 'dark' 
                    ? 'rgba(255,255,255,0.1)' 
                    : 'rgba(0,0,0,0.1)',
                }}
              >
                <AssessmentIcon 
                  sx={{ 
                    fontSize: 80, 
                    color: theme => theme.palette.mode === 'dark' 
                      ? 'rgba(255,255,255,0.2)' 
                      : 'rgba(0,0,0,0.1)', 
                    mb: 3 
                  }} 
                />
                <Typography 
                  variant="h5" 
                  gutterBottom 
                  sx={{ 
                    fontWeight: 600,
                    mb: 1.5 
                  }}
                >
                  No Reports Available
                </Typography>
                <Typography 
                  variant="body1" 
                  color="text.secondary"
                  sx={{ 
                    maxWidth: 450, 
                    mx: 'auto',
                    lineHeight: 1.6 
                  }}
                >
                  Upload and analyze images to create damage reports. Reports will be displayed here once created.
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => handleTabChange(null, 0)}
                  sx={{ 
                    mt: 4,
                    borderRadius: 2,
                    py: 1.5,
                    px: 4,
                    textTransform: 'none',
                    fontWeight: 600,
                    fontSize: '1rem',
                    boxShadow: '0 4px 12px rgba(25, 118, 210, 0.2)',
                    '&:hover': {
                      boxShadow: '0 6px 16px rgba(25, 118, 210, 0.3)'
                    }
                  }}
                >
                  Create New Report
                </Button>
              </Paper>
            ) : (
              <>
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center', 
                  mb: 4 
                }}>
                  <Typography 
                    variant="h5" 
                    sx={{ 
                      fontWeight: 600,
                      color: theme => theme.palette.mode === 'dark' 
                        ? theme.palette.primary.light 
                        : theme.palette.primary.dark
                    }}
                  >
                    Damage Reports
                  </Typography>
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={() => handleTabChange(null, 0)}
                    startIcon={<CloudUploadIcon />}
                    sx={{ 
                      borderRadius: 2,
                      py: 1,
                      px: 2.5,
                      textTransform: 'none',
                      fontWeight: 600
                    }}
                  >
                    New Analysis
                  </Button>
                </Box>
                
                <Grid container spacing={3}>
                  {reports.map((report) => (
                    <Grid item xs={12} md={6} lg={4} key={report.id}>
                      <Paper 
                        elevation={0}
                        sx={{ 
                          p: 0,
                          borderRadius: 3,
                          overflow: 'hidden',
                          transition: 'transform 0.3s, box-shadow 0.3s',
                          border: '1px solid',
                          borderColor: theme => theme.palette.mode === 'dark' 
                            ? 'rgba(255,255,255,0.05)' 
                            : 'rgba(0,0,0,0.04)',
                          boxShadow: theme => theme.palette.mode === 'dark'
                            ? '0 4px 20px rgba(0,0,0,0.2)'
                            : '0 4px 20px rgba(0,0,0,0.05)',
                          '&:hover': {
                            transform: 'translateY(-6px)',
                            boxShadow: theme => theme.palette.mode === 'dark'
                              ? '0 12px 28px rgba(0,0,0,0.3)'
                              : '0 12px 28px rgba(0,0,0,0.1)'
                          }
                        }}
                      >
                        {report.annotatedImage && (
                          <Box sx={{ 
                            position: 'relative',
                            width: '100%',
                            height: 220, 
                            overflow: 'hidden',
                            bgcolor: theme => theme.palette.mode === 'dark' 
                              ? '#121212' 
                              : '#f0f0f0'
                          }}>
                            <Box
                              component="img"
                              src={report.annotatedImage}
                              alt="Annotated damage"
                              sx={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                transition: 'transform 0.6s',
                                '&:hover': {
                                  transform: 'scale(1.05)'
                                }
                              }}
                            />
                            <Box sx={{ 
                              position: 'absolute',
                              top: 12,
                              right: 12,
                              zIndex: 2
                            }}>
                              <Chip 
                                size="small" 
                                label={report.severity} 
                                color={report.severity === 'High' ? 'error' : report.severity === 'Medium' ? 'warning' : 'success'} 
                                sx={{ 
                                  fontWeight: 600,
                                  boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
                                  borderRadius: '8px',
                                  '& .MuiChip-label': { px: 1.5 }
                                }}
                              />
                            </Box>
                          </Box>
                        )}
                        
                        <Box sx={{ 
                          p: 2.5,
                          bgcolor: theme => theme.palette.mode === 'dark' 
                            ? 'rgba(0,0,0,0.2)' 
                            : 'rgba(255,255,255,1)'
                        }}>
                          <Typography 
                            variant="subtitle1" 
                            sx={{ 
                              fontWeight: 600, 
                              fontSize: '1.1rem',
                              mb: 2,
                              color: theme => theme.palette.mode === 'dark' 
                                ? theme.palette.primary.light 
                                : theme.palette.primary.dark
                            }}
                          >
                            Damage Report
                          </Typography>
                          
                          <Box sx={{ 
                            display: 'flex', 
                            flexDirection: 'column', 
                            gap: 1.5,
                            mb: 1
                          }}>
                            <Box sx={{ 
                              display: 'flex', 
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              p: 1.5,
                              borderRadius: 2,
                              bgcolor: theme => theme.palette.mode === 'dark' 
                                ? 'rgba(255,255,255,0.03)' 
                                : 'rgba(0,0,0,0.02)'
                            }}>
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: 'text.secondary',
                                  fontWeight: 500,
                                  fontSize: '0.75rem',
                                  textTransform: 'uppercase',
                                  letterSpacing: 0.5
                                }}
                              >
                                Type
                              </Typography>
                              <Typography 
                                variant="body2" 
                                sx={{ 
                                  fontWeight: 600,
                                  color: theme => theme.palette.mode === 'dark' 
                                    ? theme.palette.primary.light 
                                    : theme.palette.primary.main
                                }}
                              >
                                {report.damageType}
                              </Typography>
                            </Box>
                            
                            <Box sx={{ 
                              display: 'flex', 
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              p: 1.5,
                              borderRadius: 2,
                              bgcolor: theme => theme.palette.mode === 'dark' 
                                ? 'rgba(255,255,255,0.03)' 
                                : 'rgba(0,0,0,0.02)'
                            }}>
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: 'text.secondary',
                                  fontWeight: 500,
                                  fontSize: '0.75rem',
                                  textTransform: 'uppercase',
                                  letterSpacing: 0.5
                                }}
                              >
                                Priority
                              </Typography>
                              <Chip 
                                label={report.priority} 
                                size="small" 
                                color="warning"
                                sx={{ 
                                  fontWeight: 600,
                                  borderRadius: '6px',
                                  height: 22,
                                  '& .MuiChip-label': { px: 1 }
                                }}
                              />
                            </Box>
                            
                            <Box sx={{ 
                              display: 'flex', 
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              p: 1.5,
                              borderRadius: 2,
                              bgcolor: theme => theme.palette.mode === 'dark' 
                                ? 'rgba(255,255,255,0.03)' 
                                : 'rgba(0,0,0,0.02)'
                            }}>
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: 'text.secondary',
                                  fontWeight: 500,
                                  fontSize: '0.75rem',
                                  textTransform: 'uppercase',
                                  letterSpacing: 0.5
                                }}
                              >
                                Date
                              </Typography>
                              <Typography 
                                variant="body2" 
                                sx={{ 
                                  fontWeight: 500,
                                  color: 'text.primary'
                                }}
                              >
                                {new Date(report.createdAt).toLocaleDateString('en-US', {
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric'
                                })}
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </>
            )}
          </Box>
        </TabPanel>
      </Box>
    </Container>
  );
};

export default AiAnalysis;