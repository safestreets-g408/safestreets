import React, { useEffect, useState, useCallback } from 'react';
import { useTheme } from '@mui/material/styles';
import { 
  Box, 
  Typography, 
  Grid, 
  Chip, 
  Divider,
  Paper,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';
// API constants are imported directly
import { API_BASE_URL, API_ENDPOINTS, TOKEN_KEY } from '../../config/constants';
import { formatLocation } from '../../utils/formatters';

const ViewDamageReport = ({ report }) => {
  const theme = useTheme();
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Function to handle reading out the report description
  const handleSpeak = useCallback(() => {
    // If already speaking, stop the speech
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    // If no description, nothing to read
    if (!report.description) {
      return;
    }

    // Create the text to speech
    const speech = new SpeechSynthesisUtterance(report.description);
    speech.rate = 0.9; // slightly slower than default
    speech.pitch = 1;
    speech.volume = 1;

    // When speech ends, reset the speaking state
    speech.onend = () => {
      setIsSpeaking(false);
    };

    // If speech is interrupted or errors out
    speech.onerror = () => {
      setIsSpeaking(false);
    };

    // Start speaking
    setIsSpeaking(true);
    window.speechSynthesis.speak(speech);
  }, [report.description, isSpeaking]);

  // Get image URL with authentication token
  const getAuthenticatedImageUrl = (reportId, type) => {
    const token = localStorage.getItem(TOKEN_KEY);
    return `${API_BASE_URL}${API_ENDPOINTS.DAMAGE_REPORTS}/report/${reportId}/image/${type}?token=${token}`;
  };

  // Function to parse location if it's stored as a string
  const parseLocation = (locationData) => {
    if (!locationData) return null;
    
    // If already an object, return it
    if (typeof locationData === 'object' && !Array.isArray(locationData)) return locationData;
    
    // If it's an array, try to use the first element (some backends might format it this way)
    if (Array.isArray(locationData) && locationData.length > 0) {
      console.log('Location is an array, using first element:', locationData[0]);
      return locationData[0];
    }
    
    // Try parsing if it's a JSON string
    if (typeof locationData === 'string') {
      try {
        // First try to parse as JSON
        const parsed = JSON.parse(locationData);
        // Handle case where parsed result is an array
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed[0];
        }
        return parsed;
      } catch (e) {
        // Not JSON, check if it's a string with coordinates pattern like "lat, long"
        const coordMatch = locationData.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
        if (coordMatch) {
          // Return as GeoJSON format object
          return {
            coordinates: [parseFloat(coordMatch[2]), parseFloat(coordMatch[1])],
            address: locationData
          };
        }
        
        // If it doesn't contain coordinates, it might be just an address
        return {
          address: locationData
        };
      }
    }
    
    return null;
  };
  
  // Extract coordinates from location data in any format
  const extractCoordinates = (location) => {
    // If location is not defined, return null
    if (!location) return null;
    
    // If it's a string, try to parse it
    if (typeof location === 'string') {
      try {
        // Try to parse as JSON
        const parsed = JSON.parse(location);
        return extractCoordinates(parsed);
      } catch (e) {
        // Not JSON, check for coordinate pattern in string
        const coordMatch = location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
        if (coordMatch) {
          return { 
            latitude: parseFloat(coordMatch[1]), 
            longitude: parseFloat(coordMatch[2])
          };
        }
        return null;
      }
    }
    
    // Handle GeoJSON format
    if (location.coordinates && Array.isArray(location.coordinates) && location.coordinates.length === 2) {
      // GeoJSON format is [longitude, latitude]
      return {
        latitude: location.coordinates[1],
        longitude: location.coordinates[0]
      };
    }
    
    // Handle lat/lng properties
    if ((location.lat !== undefined && location.lng !== undefined) ||
        (location.latitude !== undefined && location.longitude !== undefined)) {
      return {
        latitude: location.lat || location.latitude,
        longitude: location.lng || location.longitude
      };
    }
    
    // Handle nested location object common in MongoDB GeoJSON responses
    if (location.location && location.location.coordinates) {
      return {
        latitude: location.location.coordinates[1],
        longitude: location.location.coordinates[0]
      };
    }
    
    return null;
  };
  
  // Computed location object that handles various formats
  const locationObj = React.useMemo(() => {
    // Check if we have location data
    if (!report?.location) {
      console.log('No location data found in report');
      return null;
    }
    
    console.log('Processing location data:', report.location);
    
    // First check if we have a nested location object (common in MongoDB responses)
    if (report?.location?.location?.coordinates) {
      console.log('Found nested location object:', report.location.location);
      return report.location.location;
    }
    
    // Try to parse if it's a string
    const parsedLocation = parseLocation(report.location);
    console.log('Parsed location result:', parsedLocation);
    
    // If we successfully parsed it, return the parsed object
    if (parsedLocation) {
      return parsedLocation;
    }
    
    // If report has direct latitude/longitude properties
    if (report.latitude !== undefined && report.longitude !== undefined) {
      console.log('Using report direct coordinates:', report.latitude, report.longitude);
      return {
        coordinates: [report.longitude, report.latitude],
        address: typeof report.location === 'string' ? report.location : 'Location from coordinates'
      };
    }
    
    // Last resort: create a basic location object from the raw data
    if (typeof report.location === 'string') {
      return {
        address: report.location
      };
    }
    
    return report.location;
  }, [report]);
  
  useEffect(() => {
    // Log the report structure to help debug
    console.log('Report data:', report);
    console.log('Location data raw:', report?.location);
    console.log('Location data type:', typeof report?.location);
    console.log('Processed location:', locationObj);
    
    const loadImage = () => {
      // Check for both reportId and _id to ensure we can always find the report identifier
      const id = report?.reportId || report?._id;
      
      if (report && id) {
        setLoading(true);
        console.log('Loading image for report:', id);
        
        // Using direct URL with authentication token
        const directImageUrl = getAuthenticatedImageUrl(id, 'before');
        console.log('Image URL:', directImageUrl);
        
        // Create an image element to test loading
        const img = new Image();
        img.onload = () => {
          console.log('Image loaded successfully for report:', id);
          setImageUrl(directImageUrl);
          setLoading(false);
        };
        img.onerror = (e) => {
          console.error('Error loading image for report:', id, e);
          // Try the other ID format if the first one fails
          if (report.reportId && report._id && report.reportId !== report._id) {
            const altId = id === report.reportId ? report._id : report.reportId;
            const altImageUrl = getAuthenticatedImageUrl(altId, 'before');
            console.log('Trying alternative image URL:', altImageUrl);
            
            const altImg = new Image();
            altImg.onload = () => {
              console.log('Alternative image loaded successfully');
              setImageUrl(altImageUrl);
              setLoading(false);
            };
            altImg.onerror = () => {
              console.error('Both image URLs failed to load');
              setImageUrl(null);
              setLoading(false);
            };
            altImg.src = altImageUrl;
          } else {
            setImageUrl(null);
            setLoading(false);
          }
        };
        img.src = directImageUrl;
      } else {
        console.log('No report ID available, skipping image load');
        setLoading(false);
      }
    };

    loadImage();
    
    return () => {
      // Cancel any ongoing speech when component unmounts
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, [report, locationObj]);

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed': return 'success';
      case 'in progress': return 'info';
      case 'pending': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Report ID and Date */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h5" gutterBottom>
              Report #{report.reportId}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Submitted on {new Date(report.createdAt).toLocaleString()}
            </Typography>
          </Box>
          <Divider />
        </Grid>

        {/* Status and Severity */}
        <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" gutterBottom>Status</Typography>
            <Chip 
              label={report.status}
              color={getStatusColor(report.status)}
              sx={{ fontWeight: 600 }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" gutterBottom>Assignment</Typography>
            <Typography>{report.assignedTo ? report.assignedTo.name : 'Unassigned'}</Typography>
          </Paper>
        </Grid>

        {/* Location Details */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Location Details</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">Region</Typography>
                <Typography variant="body1">{report.region}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">Location</Typography>
                <Typography variant="body1">
                  {(() => {
                    // First check if locationObj has an address
                    if (locationObj && locationObj.address) {
                      return locationObj.address;
                    }
                    
                    // Check if report has a location string
                    if (typeof report.location === 'string') {
                      // Remove any coordinate-like patterns to display just the address part
                      const addressOnly = report.location.replace(/[-+]?\d+\.\d+[,\s]+[-+]?\d+\.\d+/, '').trim();
                      if (addressOnly) {
                        return addressOnly;
                      }
                      return report.location;
                    }
                    
                    // Use the formatter as fallback
                    return formatLocation(locationObj || (report.latitude && report.longitude ? { latitude: report.latitude, longitude: report.longitude } : null));
                  })()}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">Coordinates</Typography>
                <Typography variant="body1">
                  {(() => {
                    // Try to extract coordinates with our helper
                    const coords = extractCoordinates(locationObj || report.location);
                    if (coords) {
                      return `${coords.latitude.toFixed(6)}, ${coords.longitude.toFixed(6)}`;
                    }
                    
                    // If report has direct coordinates
                    if (report.latitude !== undefined && report.longitude !== undefined) {
                      return `${report.latitude.toFixed(6)}, ${report.longitude.toFixed(6)}`;
                    }
                    
                    // Fallback for older logic
                    if (locationObj) {
                      // Try to get coordinates from locationObj
                      if (locationObj.coordinates && Array.isArray(locationObj.coordinates) && locationObj.coordinates.length === 2) {
                        try {
                          // In GeoJSON format, coordinates are [longitude, latitude]
                          const longitude = locationObj.coordinates[0];
                          const latitude = locationObj.coordinates[1];
                          
                          if (typeof longitude === 'number' && typeof latitude === 'number') {
                            return `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
                          } else if (longitude !== undefined && latitude !== undefined) {
                            return `${latitude}, ${longitude}`;
                          }
                        } catch (e) {
                          console.error('Error formatting coordinates:', e);
                        }
                      }
                      
                      // Try lat/lng or latitude/longitude properties
                      if ((locationObj.lat !== undefined && locationObj.lng !== undefined) ||
                          (locationObj.latitude !== undefined && locationObj.longitude !== undefined)) {
                        const lat = locationObj.lat || locationObj.latitude;
                        const lng = locationObj.lng || locationObj.longitude;
                        
                        if (typeof lat === 'number' && typeof lng === 'number') {
                          return `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                        } else if (lat !== undefined && lng !== undefined) {
                          return `${lat}, ${lng}`;
                        }
                      }
                    }
                    
                    // Last fallback - no coordinates available
                    return 'Coordinates not available';
                  })()}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Damage Details */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Damage Information</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">Damage Type</Typography>
                <Typography variant="body1">{report.damageType}</Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">Severity</Typography>
                <Chip 
                  label={report.severity}
                  color={getSeverityColor(report.severity)}
                  size="small"
                />
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">Priority</Typography>
                <Typography variant="body1">{report.priority}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">Description</Typography>
                  <Tooltip title={isSpeaking ? "Stop reading" : "Read description aloud"}>
                    <IconButton 
                      size="small" 
                      onClick={handleSpeak} 
                      color={isSpeaking ? "primary" : "default"}
                      disabled={!report.description}
                    >
                      {isSpeaking ? <VolumeUpIcon /> : <VolumeOffIcon />}
                    </IconButton>
                  </Tooltip>
                </Box>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  {report.description}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">Action Required</Typography>
                <Typography variant="body1">{report.action}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Image */}
        {report && report.reportId && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Damage Image</Typography>
              <Box
                sx={{
                  position: 'relative',
                  width: '100%',
                  height: 300,
                  borderRadius: 1,
                  overflow: 'hidden',
                  backgroundColor: theme.palette.background.default
                }}
              >
                {loading ? (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%'
                    }}
                  >
                    <CircularProgress />
                  </Box>
                ) : imageUrl ? (
                  <img
                    src={imageUrl}
                    alt="Damage"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                    onError={(e) => {
                      console.error('Image failed to load in view');
                      e.target.onerror = null;
                      e.target.src = 'https://via.placeholder.com/400x300?text=Image+Not+Available';
                    }}
                  />
                ) : (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      backgroundColor: theme.palette.background.paper
                    }}
                  >
                    <Typography color="text.secondary">
                      No image available
                    </Typography>
                  </Box>
                )}
              </Box>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ViewDamageReport;
