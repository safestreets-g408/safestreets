import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Paper,
  Grid,
  Divider,
  Container,
  Chip,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Tooltip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import AssessmentIcon from '@mui/icons-material/Assessment';
import { api } from '../utils/api';
import { TOKEN_KEY } from '../config/constants';

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
        <Typography variant="h4" gutterBottom>
          AI Analysis
        </Typography>
        
        {locationStatus && (
          <Alert 
            severity={locationStatus.includes('successfully') ? 'success' : 'info'} 
            sx={{ mb: 2 }}
          >
            {locationStatus}
          </Alert>
        )}

        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            aria-label="AI analysis tabs"
          >
            <Tab 
              label="Upload Image" 
              icon={<CloudUploadIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="View Reports" 
              icon={<AssessmentIcon />} 
              iconPosition="start"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Paper elevation={3}>
            <Box sx={{ p: 3 }}>
              {/* Image Upload Section */}
              <Box 
                sx={{ 
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 2
                }}
              >
                <Button
                  component="label"
                  role={undefined}
                  variant="contained"
                  tabIndex={-1}
                  startIcon={<ImageIcon />}
                >
                  Choose Image
                  <VisuallyHiddenInput 
                    type="file" 
                    accept="image/*"
                    onChange={handleImageSelect}
                  />
                </Button>

                {error && (
                  <Alert severity="error" sx={{ width: '100%', mt: 2 }}>
                    {error}
                  </Alert>
                )}

                {selectedImage && (
                  <Box sx={{ mt: 2, width: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Selected Image
                    </Typography>
                    <Box
                      sx={{
                        width: '100%',
                        height: 300,
                        backgroundImage: `url(${previewUrl})`,
                        backgroundSize: 'contain',
                        backgroundPosition: 'center',
                        backgroundRepeat: 'no-repeat',
                        border: '1px solid #e0e0e0',
                        borderRadius: 1
                      }}
                    />
                    
                    {/* Model Selection */}
                    <Box sx={{ mt: 2, mb: 2 }}>
                      <FormControl component="fieldset">
                        <FormLabel component="legend">Select AI Model</FormLabel>
                        <RadioGroup
                          row
                          aria-label="ai-model"
                          name="ai-model"
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                        >
                          <FormControlLabel 
                            value="vit" 
                            control={<Radio />} 
                            label="ViT" 
                          />
                          <FormControlLabel 
                            value="yolo" 
                            control={<Radio />} 
                            label="YOLO v8" 
                          />
                        </RadioGroup>
                        
                        {selectedModel === 'yolo' && (
                          <Box sx={{ mt: 1 }}>
                            {loadingModelInfo ? (
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <CircularProgress size={16} sx={{ mr: 1 }} />
                                <Typography variant="caption">Loading model info...</Typography>
                              </Box>
                            ) : modelInfo ? (
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                                YOLO model: {modelInfo.classes?.length || '0'} classes, 
                                Confidence threshold: {modelInfo.confidence_threshold || 'N/A'}
                              </Typography>
                            ) : (
                              <Typography variant="caption" color="error" sx={{ display: 'block' }}>
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
                      sx={{ mt: 2 }}
                    >
                      {loading ? (
                        <>
                          <CircularProgress size={24} sx={{ mr: 1 }} />
                          Analyzing...
                        </>
                      ) : (
                        'Analyze Image'
                      )}
                    </Button>
                  </Box>
                )}

                {locationData && (
                  <Box sx={{ mt: 2, width: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Location Information
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Latitude: {locationData.latitude}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Longitude: {locationData.longitude}
                        </Typography>
                      </Grid>
                      {locationData.address && (
                        <Grid item xs={12}>
                          <Typography variant="body2" color="text.secondary">
                            Address: {locationData.address}
                          </Typography>
                        </Grid>
                      )}
                    </Grid>
                  </Box>
                )}

                {prediction && (
                  <Box sx={{ mt: 3, width: '100%' }}>
                    <Typography variant="h6" gutterBottom>
                      Analysis Results 
                      <Chip 
                        label={prediction.model === 'yolo' 
                          ? (prediction.fallback ? 'YOLO Fallback Detection' : 'YOLO Detection') 
                          : 'ViT Classification'} 
                        color={prediction.fallback ? "warning" : "info"}
                        size="small" 
                        sx={{ ml: 1 }}
                      />
                    </Typography>
                    
                    {/* Display annotated image if available */}
                    {prediction.annotatedImage && (
                      <Box sx={{ my: 2, textAlign: 'center' }}>
                        <Typography variant="subtitle2" gutterBottom>Annotated Image</Typography>
                        <Box
                          component="img"
                          src={typeof prediction.annotatedImage === 'string' && prediction.annotatedImage.startsWith('data:') 
                            ? prediction.annotatedImage 
                            : `data:image/jpeg;base64,${prediction.annotatedImage}`}
                          alt="Annotated"
                          sx={{
                            maxWidth: '100%',
                            maxHeight: '400px',
                            objectFit: 'contain',
                            border: '1px solid #e0e0e0',
                            borderRadius: 1
                          }}
                        />
                      </Box>
                    )}
                    
                    {/* YOLO specific results */}
                    {prediction.model === 'yolo' && (
                      <>
                        <Grid container spacing={2} sx={{ mb: 2 }}>
                          <Grid item xs={12} sm={6}>
                            <Typography variant="body1">
                              Damage Type: <Chip label={prediction.damageType} color="primary" size="small" />
                            </Typography>
                          </Grid>
                          <Grid item xs={12} sm={6}>
                            <Typography variant="body1">
                              Severity: <Chip label={prediction.severity} color={prediction.severity === 'HIGH' ? "error" : prediction.severity === 'MEDIUM' ? "warning" : "success"} size="small" />
                            </Typography>
                          </Grid>
                          <Grid item xs={12} sm={6}>
                            <Typography variant="body1">
                              Priority: <Chip label={prediction.priority} color="warning" size="small" />
                            </Typography>
                          </Grid>
                          <Grid item xs={12} sm={6}>
                            <Typography variant="body1">
                              Objects Detected: <Chip label={prediction.detectionCount || 0} color="info" size="small" />
                            </Typography>
                          </Grid>
                        </Grid>
                        
                        {prediction.detections && prediction.detections.length > 0 && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle1" gutterBottom>
                              Detection Details
                            </Typography>
                            <Paper variant="outlined" sx={{ p: 1, maxHeight: '200px', overflow: 'auto' }}>
                              {prediction.detections.map((detection, index) => (
                                <Box key={index} sx={{ p: 1, mb: 1, borderBottom: '1px solid #eee' }}>
                                  <Grid container spacing={1}>
                                    <Grid item xs={6}>
                                      <Typography variant="body2">Class: <Chip label={detection.class || detection.name} size="small" color="primary" /></Typography>
                                    </Grid>
                                    <Grid item xs={6}>
                                      <Typography variant="body2">Confidence: {((detection.confidence || detection.score || 0) * 100).toFixed(1)}%</Typography>
                                    </Grid>
                                  </Grid>
                                </Box>
                              ))}
                            </Paper>
                          </Box>
                        )}
                      </>
                    )}
                    
                    {/* ViT specific results */}
                    {prediction.model === 'vit' && (
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <Typography variant="body1">
                            Damage Type: <Chip label={prediction.damageType} color="primary" size="small" />
                          </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Typography variant="body1">
                            Severity: <Chip label={prediction.severity} color="error" size="small" />
                          </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Typography variant="body1">
                            Priority: <Chip label={prediction.priority} color="warning" size="small" />
                          </Typography>
                        </Grid>
                      </Grid>
                    )}
                    
                    <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={saveReport}
                        disabled={loading}
                      >
                        {loading ? (
                          <>
                            <CircularProgress size={20} sx={{ mr: 1 }} />
                            Saving...
                          </>
                        ) : (
                          'Save Report'
                        )}
                      </Button>
                      <Button
                        variant="outlined"
                        color="secondary"
                        onClick={() => {
                          handleTabChange(null, 1);
                        }}
                      >
                        View All Reports
                      </Button>
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
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : reports.length === 0 ? (
              <Paper sx={{ p: 4, borderRadius: 2, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                <AssessmentIcon sx={{ fontSize: 60, color: 'text.secondary', opacity: 0.5, mb: 2 }} />
                <Typography variant="h6" gutterBottom>No Reports Available</Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload and analyze images to create damage reports
                </Typography>
              </Paper>
            ) : (
              <Grid container spacing={3}>
                {reports.map((report) => (
                  <Grid item xs={12} md={6} lg={4} key={report.id}>
                    <Paper 
                      elevation={2} 
                      sx={{ 
                        p: 0,
                        borderRadius: 2,
                        overflow: 'hidden',
                        transition: 'transform 0.2s, box-shadow 0.2s',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 4
                        }
                      }}
                    >
                      {report.annotatedImage && (
                        <Box sx={{ 
                          width: '100%',
                          height: 200, 
                          overflow: 'hidden',
                          backgroundColor: '#f0f0f0'
                        }}>
                          <img
                            src={report.annotatedImage}
                            alt="Annotated damage"
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover'
                            }}
                          />
                        </Box>
                      )}
                      
                      <Box sx={{ p: 2.5 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Typography variant="h6" sx={{ fontWeight: 500 }}>
                            Damage Report
                          </Typography>
                          <Chip 
                            size="small" 
                            label={report.severity} 
                            color={report.severity === 'High' ? 'error' : report.severity === 'Medium' ? 'warning' : 'success'} 
                          />
                        </Box>
                        
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" color="text.secondary">Type</Typography>
                            <Typography variant="body2" fontWeight="medium">{report.damageType}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" color="text.secondary">Priority</Typography>
                            <Typography variant="body2" fontWeight="medium">{report.priority}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" color="text.secondary">Date</Typography>
                            <Typography variant="body2">{new Date(report.createdAt).toLocaleDateString()}</Typography>
                          </Box>
                        </Box>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            )}
          </Box>
        </TabPanel>
      </Box>
    </Container>
  );
};

export default AiAnalysis;