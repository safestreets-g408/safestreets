/* eslint-disable no-unused-vars */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Chip, 
  Stack, 
  useTheme,
  TextField,
  Button,
  IconButton,
  CircularProgress,
  Drawer,
  Grid,
  Divider,
  Switch,
  FormGroup,
  FormControlLabel,
  Alert,
  Tooltip
} from '@mui/material';
import LayersIcon from '@mui/icons-material/Layers';
import WarningIcon from '@mui/icons-material/Warning';
import FilterListIcon from '@mui/icons-material/FilterList';
import RefreshIcon from '@mui/icons-material/Refresh';
import CloseIcon from '@mui/icons-material/Close';
import InfoIcon from '@mui/icons-material/Info';
import SearchIcon from '@mui/icons-material/Search';
import DashboardIcon from '@mui/icons-material/Dashboard';
import HeatmapIcon from '@mui/icons-material/Whatshot'; 
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import 'leaflet.markercluster';
import 'leaflet.heat';

import { api } from '../utils/api';
import { API_ENDPOINTS } from '../config/constants';
import ViewDamageReport from '../components/reports/ViewDamageReport';

function MapView() {
  const theme = useTheme();
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersLayerRef = useRef(null);
  const clusterLayerRef = useRef(null);
  const heatLayerRef = useRef(null);
  
  // State
  const [mapStyle, setMapStyle] = useState('streets');
  const [selectedReport, setSelectedReport] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [damageReports, setDamageReports] = useState([]);
  const [error, setError] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showDebug, setShowDebug] = useState(false); // Debug mode state
  const [filters, setFilters] = useState({
    severity: '',
    damageType: '',
    startDate: null,
    endDate: null,
    region: ''
  });
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);
  const [damageTypes, setDamageTypes] = useState([]);
  const [regions, setRegions] = useState([]);
  
  // Define the getSeverityColor function early
  const getSeverityColor = useCallback((severity) => {
    if (!severity) return theme.palette.primary.main;
    const severityUpper = String(severity).toUpperCase();
    switch(severityUpper) {
      case 'HIGH': return theme.palette.error.main;
      case 'MEDIUM': return theme.palette.warning.main;
      case 'LOW': return theme.palette.info.main;
      default: return theme.palette.primary.main;
    }
  }, [theme.palette.error.main, theme.palette.warning.main, theme.palette.info.main, theme.palette.primary.main]);

  // Map style options
  const mapStyles = [
    { value: 'streets', label: 'Streets', url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png' },
    { value: 'satellite', label: 'Satellite', url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}' },
    { value: 'topo', label: 'Topographic', url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png' },
    { value: 'dark', label: 'Dark', url: 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png' }
  ];
  
  // Enhanced helper function to extract coordinates from any location format
  const extractCoordinates = useCallback((location) => {
    // If location is not defined, return null
    if (!location) return null;
    
    console.log('Extracting coordinates from:', typeof location, location);
    
    try {
      // If it's already a valid pair of coordinates, return them
      if (Array.isArray(location) && location.length === 2 &&
          typeof location[0] === 'number' && typeof location[1] === 'number') {
        console.log('Found array coordinates:', location);
        // Most GeoJSON uses [longitude, latitude] format
        return { latitude: location[1], longitude: location[0] };
      }
      
      // Handle GeoJSON format directly in location object
      if (typeof location === 'object' && location !== null) {
        // Check for standard GeoJSON format
        if (location.type === 'Point' && Array.isArray(location.coordinates) && location.coordinates.length === 2) {
          console.log('Found GeoJSON Point coordinates:', location.coordinates);
          return {
            latitude: location.coordinates[1],
            longitude: location.coordinates[0]
          };
        }
        
        // Check for coordinates array directly
        if (location.coordinates && Array.isArray(location.coordinates) && location.coordinates.length === 2) {
          console.log('Found coordinates array in object:', location.coordinates);
          return {
            latitude: location.coordinates[1],
            longitude: location.coordinates[0]
          };
        }
        
        // Check for lat/lng or latitude/longitude properties
        if ((location.lat !== undefined && location.lng !== undefined) ||
            (location.latitude !== undefined && location.longitude !== undefined)) {
          const lat = location.lat !== undefined ? location.lat : location.latitude;
          const lng = location.lng !== undefined ? location.lng : location.longitude;
          
          // Validate the coordinates are actual numbers and in reasonable range
          if (typeof lat === 'number' && typeof lng === 'number' &&
              lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
            console.log('Found lat/lng properties:', lat, lng);
            return { latitude: lat, longitude: lng };
          }
          
          // Try parsing string values to numbers
          if ((typeof lat === 'string' || typeof lng === 'string')) {
            const numLat = parseFloat(lat);
            const numLng = parseFloat(lng);
            if (!isNaN(numLat) && !isNaN(numLng) &&
                numLat >= -90 && numLat <= 90 && numLng >= -180 && numLng <= 180) {
              console.log('Parsed lat/lng string values:', numLat, numLng);
              return { latitude: numLat, longitude: numLng };
            }
          }
        }
        
        // Handle nested location object common in MongoDB GeoJSON responses
        if (location.location) {
          // Try recursively with the nested location
          const nestedCoordinates = extractCoordinates(location.location);
          if (nestedCoordinates) {
            console.log('Found coordinates in nested location object');
            return nestedCoordinates;
          }
        }
        
        // Check for geo property which sometimes contains location data
        if (location.geo) {
          const geoCoordinates = extractCoordinates(location.geo);
          if (geoCoordinates) {
            console.log('Found coordinates in geo property');
            return geoCoordinates;
          }
        }
      }
      
      // If it's a string, try multiple parsing approaches
      if (typeof location === 'string') {
        // First try to parse as JSON
        try {
          const parsed = JSON.parse(location);
          const parsedCoordinates = extractCoordinates(parsed);
          if (parsedCoordinates) {
            console.log('Successfully parsed location JSON');
            return parsedCoordinates;
          }
        } catch (e) {
          // Not valid JSON, continue to other string parsing methods
        }
        
        // Check for coordinate patterns in string
        // Match patterns like:
        // - "17.3850, 78.4866" (lat, lng with comma)
        // - "17.3850 78.4866" (lat lng with space)
        // - "(17.3850, 78.4866)" (parentheses format)
        // - "latitude: 17.3850, longitude: 78.4866" (labeled format)
        
        // Try standard coordinate pattern with comma or space
        const coordMatch = location.match(/([-+]?\d+\.?\d*)[,\s]+([-+]?\d+\.?\d*)/);
        if (coordMatch) {
          const lat = parseFloat(coordMatch[1]);
          const lng = parseFloat(coordMatch[2]);
          
          // Validate coordinates are in reasonable range
          if (!isNaN(lat) && !isNaN(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
            console.log('Found coordinate pattern in string:', lat, lng);
            return { latitude: lat, longitude: lng };
          }
        }
        
        // Try pattern with "lat/latitude" and "lng/longitude" labels
        const labelMatch = location.match(
          /(?:lat|latitude)[^\d-+]*([-+]?\d+\.?\d*)[^\d-+]*(?:lon|lng|longitude)[^\d-+]*([-+]?\d+\.?\d*)|(?:lon|lng|longitude)[^\d-+]*([-+]?\d+\.?\d*)[^\d-+]*(?:lat|latitude)[^\d-+]*([-+]?\d+\.?\d*)/i
        );
        
        if (labelMatch) {
          // Check which pattern matched (lat-lng or lng-lat)
          let lat, lng;
          
          if (labelMatch[1] !== undefined && labelMatch[2] !== undefined) {
            // lat-lng pattern matched
            lat = parseFloat(labelMatch[1]);
            lng = parseFloat(labelMatch[2]);
          } else if (labelMatch[3] !== undefined && labelMatch[4] !== undefined) {
            // lng-lat pattern matched
            lng = parseFloat(labelMatch[3]);
            lat = parseFloat(labelMatch[4]);
          }
          
          if (!isNaN(lat) && !isNaN(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
            console.log('Found labeled coordinate pattern in string:', lat, lng);
            return { latitude: lat, longitude: lng };
          }
        }
      }
      
      // If we got this far, coordinates couldn't be extracted
      console.warn('Failed to extract coordinates from:', location);
      return null;
    } catch (error) {
      console.error('Error in extractCoordinates function:', error);
      return null;
    }
  }, []);
  
  // Enhanced helper function to geocode addresses using OpenStreetMap Nominatim API
  const geocodeAddress = useCallback(async (address) => {
    if (!address || typeof address !== 'string' || address.trim().length < 3) {
      console.log('Invalid or too short address for geocoding:', address);
      return null;
    }
    
    try {
      // Clean up the address - remove any coordinate-like patterns, JSON artifacts
      const cleanAddress = address
        .replace(/[{}[\]"']/g, ' ')  // Remove JSON/object syntax characters
        .replace(/\b(?:lat|latitude|lon|lng|longitude|coordinates|type|point)\b[:=]/gi, ' ')  // Remove coordinate-related keys
        .replace(/[-+]?\d+\.\d+,\s*[-+]?\d+\.\d+/g, ' ')  // Remove coordinate-like patterns
        .trim();
      
      if (cleanAddress.length < 3) {
        console.log('Address too short after cleaning:', cleanAddress);
        return null;
      }
      
      console.log('Geocoding address:', cleanAddress);
      
      // Add a small random delay to avoid hitting rate limits (between 100-300ms)
      await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));
      
      // Use Nominatim API to geocode the address with proper headers
      const encodedAddress = encodeURIComponent(cleanAddress);
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodedAddress}&format=json&limit=1&addressdetails=0`,
        {
          headers: {
            'User-Agent': 'SafeStreets Admin Portal Application',
            'Accept-Language': 'en-US,en',
            'Referer': 'https://safestreets.org'
          }
        }
      );
      
      if (!response.ok) {
        console.error('Geocoding API error:', response.status, response.statusText);
        return null;
      }
      
      const data = await response.json();
      
      if (data && data.length > 0) {
        const result = data[0];
        
        // Validate coordinates are in a reasonable range
        const lat = parseFloat(result.lat);
        const lon = parseFloat(result.lon);
        
        if (!isNaN(lat) && !isNaN(lon) && 
            lat >= -90 && lat <= 90 && 
            lon >= -180 && lon <= 180) {
          console.log('Successfully geocoded address:', cleanAddress, 'to coordinates:', lat, lon);
          return {
            latitude: lat,
            longitude: lon,
            confidence: parseFloat(result.importance || 0),
            displayName: result.display_name
          };
        } else {
          console.warn('Geocoding returned invalid coordinates for address:', cleanAddress);
          return null;
        }
      } else {
        console.warn('No geocoding results found for address:', cleanAddress);
        return null;
      }
    } catch (error) {
      console.error('Geocoding error for address:', address, error);
      return null;
    }
  }, []);
  
  // Fetch damage reports
  useEffect(() => {
    const fetchDamageReports = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Build query parameters
        const queryParams = new URLSearchParams();
        
        if (filters.severity) {
          queryParams.append('severity', filters.severity);
        }
        
        if (filters.damageType) {
          queryParams.append('damageType', filters.damageType);
        }
        
        if (filters.region) {
          queryParams.append('region', filters.region);
        }
        
        if (filters.startDate) {
          queryParams.append('startDate', filters.startDate.toISOString());
        }
        
        if (filters.endDate) {
          queryParams.append('endDate', filters.endDate.toISOString());
        }
        
        const query = queryParams.toString();
        const endpoint = `${API_ENDPOINTS.DAMAGE_REPORTS}/reports${query ? `?${query}` : ''}`;
        
        console.log('Fetching damage reports from:', endpoint);
        const response = await api.get(endpoint);
        
        // Process data from API
        const reports = Array.isArray(response) ? response : [];
        
        console.log(`Received ${reports.length} reports from API:`, reports.slice(0, 2));
        
        // Process and geocode reports with enhanced error handling and location extraction
        const processedReportsPromises = reports.map(async (report, index) => {
          try {
            // Skip processing if no report ID or already over 100 reports processed
            // This prevents excessive geocoding requests and performance issues
            if (!report || (!report.reportId && !report._id)) {
              console.warn('Skipping invalid report');
              return null;
            }
            
            const reportId = report.reportId || report._id;
            console.log(`Processing report ${index + 1}/${reports.length}: ${reportId}`);
            
            // Check for coordinates in multiple locations
            // 1. First try the location property
            let coordinates = null;
            if (report.location) {
              coordinates = extractCoordinates(report.location);
              if (coordinates) {
                console.log(`Found coordinates in location property for report ${reportId}:`, coordinates);
                return {
                  ...report,
                  latitude: coordinates.latitude,
                  longitude: coordinates.longitude,
                  originalLocation: report.location,
                  coordinateSource: 'extracted'
                };
              }
            }
            
            // 2. Try any geoLocation property
            if (report.geoLocation) {
              coordinates = extractCoordinates(report.geoLocation);
              if (coordinates) {
                console.log(`Found coordinates in geoLocation property for report ${reportId}:`, coordinates);
                return {
                  ...report,
                  latitude: coordinates.latitude,
                  longitude: coordinates.longitude,
                  originalLocation: report.geoLocation,
                  coordinateSource: 'extracted'
                };
              }
            }
            
            // 3. Try coordinates property
            if (report.coordinates) {
              coordinates = extractCoordinates(report.coordinates);
              if (coordinates) {
                console.log(`Found coordinates in coordinates property for report ${reportId}:`, coordinates);
                return {
                  ...report,
                  latitude: coordinates.latitude,
                  longitude: coordinates.longitude,
                  originalLocation: report.coordinates,
                  coordinateSource: 'extracted'
                };
              }
            }
            
            // 4. Try position property
            if (report.position) {
              coordinates = extractCoordinates(report.position);
              if (coordinates) {
                console.log(`Found coordinates in position property for report ${reportId}:`, coordinates);
                return {
                  ...report,
                  latitude: coordinates.latitude,
                  longitude: coordinates.longitude,
                  originalLocation: report.position,
                  coordinateSource: 'extracted'
                };
              }
            }
            
            // 5. Check if we have lat/lng directly on the report object
            if ((report.lat !== undefined && report.lng !== undefined) || 
                (report.latitude !== undefined && report.longitude !== undefined)) {
              
              const lat = report.lat !== undefined ? report.lat : report.latitude;
              const lng = report.lng !== undefined ? report.lng : report.longitude;
              
              // Convert to numbers if they're strings
              const latNum = typeof lat === 'string' ? parseFloat(lat) : lat;
              const lngNum = typeof lng === 'string' ? parseFloat(lng) : lng;
              
              if (!isNaN(latNum) && !isNaN(lngNum) && 
                  latNum >= -90 && latNum <= 90 && 
                  lngNum >= -180 && lngNum <= 180) {
                console.log(`Found direct lat/lng properties for report ${reportId}:`, latNum, lngNum);
                return {
                  ...report,
                  latitude: latNum,
                  longitude: lngNum,
                  originalLocation: { lat: latNum, lng: lngNum },
                  coordinateSource: 'extracted'
                };
              }
            }
            
            // 6. Try geocoding for address-like strings
            // For address lookup, check multiple fields
            const addressFields = [
              report.location, 
              report.address, 
              report.streetAddress,
              report.fullAddress,
              report.locationDescription
            ];
            
            // Find the best candidate for geocoding (longest non-empty string)
            const addressToGeocode = addressFields
              .filter(field => typeof field === 'string' && field.trim().length > 5)
              .sort((a, b) => b.length - a.length)[0];
            
            if (addressToGeocode) {
              // If we have an address field, try geocoding
              // But limit geocoding to first 50 reports to avoid rate limiting
              if (index < 50) {
                console.log(`Geocoding address for report ${reportId}:`, addressToGeocode);
                const geocoded = await geocodeAddress(addressToGeocode);
                
                if (geocoded) {
                  // If we have region info in the report, add it to the coordinates for better context
                  const regionInfo = report.region ? ` (${report.region})` : '';
                  
                  return {
                    ...report,
                    latitude: geocoded.latitude,
                    longitude: geocoded.longitude,
                    originalLocation: addressToGeocode,
                    geocodedAddress: geocoded.displayName,
                    geocodeConfidence: geocoded.confidence,
                    coordinateSource: 'geocoded',
                    displayLocation: `${addressToGeocode}${regionInfo}`
                  };
                }
              } else {
                console.log(`Skipping geocoding for report ${reportId} to avoid rate limits`);
              }
            }
            
            // 7. Try using region information for approximate location
            // If we have region information, we can use a pre-defined set of coordinates for common regions
            if (report.region) {
              const regionCoordinates = {
                'Hyderabad': { latitude: 17.3850, longitude: 78.4867 },
                'Mumbai': { latitude: 19.0760, longitude: 72.8777 },
                'Delhi': { latitude: 28.7041, longitude: 77.1025 },
                'Bangalore': { latitude: 12.9716, longitude: 77.5946 },
                'Chennai': { latitude: 13.0827, longitude: 80.2707 },
                'Kolkata': { latitude: 22.5726, longitude: 88.3639 }
                // Add more regions as needed
              };
              
              const region = Object.keys(regionCoordinates).find(
                r => report.region.toLowerCase().includes(r.toLowerCase())
              );
              
              if (region) {
                // Use the region coordinates with a small random offset (0.5-1km)
                const randomOffset = 0.01 * Math.random(); // ~1km at the equator
                const offset = () => (Math.random() - 0.5) * randomOffset;
                
                console.log(`Using region-based location for report ${reportId} (${region})`);
                return {
                  ...report,
                  latitude: regionCoordinates[region].latitude + offset(),
                  longitude: regionCoordinates[region].longitude + offset(),
                  originalLocation: report.region,
                  coordinateSource: 'region'
                };
              }
            }
            
            // 8. As a last resort, use default location with randomization
            // Make the randomization deterministic based on report ID to keep consistent between refreshes
            const reportIdString = reportId.toString();
            const hashCode = reportIdString.split('').reduce((a, b) => (a * 31 + b.charCodeAt(0)) & 0xfffffff, 0);
            const randomOffset = () => ((hashCode / 0xfffffff) - 0.5) * 0.1; // ~5km at the equator
            
            console.warn(`Using default location for report ${reportId}`);
            return {
              ...report,
              latitude: 17.3850 + randomOffset(),
              longitude: 78.4867 + randomOffset(),
              originalLocation: report.location || 'Unknown',
              coordinateSource: 'default'
            };
          } catch (err) {
            console.error(`Error processing report ${report?.reportId || report?._id || index}:`, err);
            return null;
          }
        });
        
        // Wait for all geocoding requests to complete
        const processedReports = (await Promise.all(processedReportsPromises)).filter(report => report !== null);
        
        console.log(`Processed ${processedReports.length} reports with coordinates`);
        console.log('Coordinate sources:', 
                   'Extracted:', processedReports.filter(r => r.coordinateSource === 'extracted').length,
                   'Geocoded:', processedReports.filter(r => r.coordinateSource === 'geocoded').length,
                   'Default:', processedReports.filter(r => r.coordinateSource === 'default').length);
        
        if (processedReports.length > 0) {
          console.log('Sample processed report:', processedReports[0]);
        }
        
        setDamageReports(processedReports);
        
        // Extract unique damage types and regions for filters
        const uniqueDamageTypes = [...new Set(processedReports.map(r => r.damageType))].filter(Boolean);
        const uniqueRegions = [...new Set(processedReports.map(r => r.region))].filter(Boolean);
        
        setDamageTypes(uniqueDamageTypes);
        setRegions(uniqueRegions);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching damage reports:', err);
        setError(err.message || 'Failed to load damage reports');
        setLoading(false);
      }
    };
    
    fetchDamageReports();
  }, [filters, extractCoordinates, geocodeAddress]);

  // Initialize map
  useEffect(() => {
    if (!mapInstanceRef.current && mapRef.current) {
      // Create map instance with maxZoom specified
      mapInstanceRef.current = L.map(mapRef.current, {
        maxZoom: 19,
        minZoom: 2
      }).setView([17.3850, 78.4866], 12);
      
      // Create marker cluster group
      clusterLayerRef.current = L.markerClusterGroup({
        maxClusterRadius: 50,
        disableClusteringAtZoom: 16,
        spiderfyOnMaxZoom: true
      });
      
      // Add cluster layer to map
      mapInstanceRef.current.addLayer(clusterLayerRef.current);
      
      // Set initial tile layer
      const initialStyle = mapStyles.find(style => style.value === mapStyle);
      L.tileLayer(initialStyle.url, {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }).addTo(mapInstanceRef.current);
    }
    
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  // Handle map style changes
  useEffect(() => {
    if (mapInstanceRef.current) {
      // Remove existing tile layers
      mapInstanceRef.current.eachLayer(layer => {
        if (layer instanceof L.TileLayer) {
          mapInstanceRef.current.removeLayer(layer);
        }
      });
      
      // Add new tile layer
      const selectedStyle = mapStyles.find(style => style.value === mapStyle);
      L.tileLayer(selectedStyle.url, {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }).addTo(mapInstanceRef.current);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapStyle]);
  
  // Update markers when damage reports change
  useEffect(() => {
    if (mapInstanceRef.current && clusterLayerRef.current) {
      // Clear existing markers
      clusterLayerRef.current.clearLayers();
      
      // Remove existing heatmap layer if it exists
      if (heatLayerRef.current && mapInstanceRef.current.hasLayer(heatLayerRef.current)) {
        mapInstanceRef.current.removeLayer(heatLayerRef.current);
        heatLayerRef.current = null;
      }
      
      // Create points for heatmap
      const heatPoints = [];
      
      // Log reports count for debugging
      console.log(`Creating map markers for ${damageReports.length} reports`);
      
      // Check if we have any reports to display
      if (damageReports.length === 0) {
        console.warn('No damage reports to display on map. Check API response and coordinate extraction.');
        setError(prevError => prevError || 'No reports to display. Try adjusting filters or refreshing.');
        return;
      }
      
      // Add markers for each damage report
      damageReports.forEach((report, index) => {
        // Ensure we have valid coordinates
        if (!report.latitude || !report.longitude) {
          console.warn(`Skipping marker for report ${report.reportId || report._id} - missing coordinates`);
          return;
        }
        
        const markerColor = getSeverityColor(report.severity);
        
        console.log(`Creating marker ${index + 1}/${damageReports.length}: ${report.reportId || report._id} at ${report.latitude}, ${report.longitude} (${report.coordinateSource})`);
        
        // Create custom icon - different styles based on coordinate source
        let iconHtml = '';
        if (report.coordinateSource === 'extracted') {
          // Use a standard pin for extracted coordinates
          iconHtml = `<div style="color: ${markerColor}; font-size: 32px; display: flex; justify-content: center; align-items: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                        </svg>
                      </div>`;
        } else if (report.coordinateSource === 'geocoded') {
          // Use a map pin with a dot for geocoded addresses
          iconHtml = `<div style="color: ${markerColor}; font-size: 32px; display: flex; justify-content: center; align-items: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                          <circle cx="12" cy="9" r="3" fill="white"/>
                        </svg>
                      </div>`;
        } else {
          // Use a dashed outline pin for default coordinates
          iconHtml = `<div style="color: ${markerColor}; font-size: 32px; display: flex; justify-content: center; align-items: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 24 24">
                          <path stroke="white" stroke-width="1" stroke-dasharray="2,2" d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                          <text x="12" y="10" text-anchor="middle" font-size="8" fill="white">?</text>
                        </svg>
                      </div>`;
        }
        
        const icon = L.divIcon({
          className: 'custom-div-icon',
          html: iconHtml,
          iconSize: [32, 32],
          iconAnchor: [16, 32]
        });
        
        // Format date nicely
        const reportDate = new Date(report.timestamp || report.createdAt).toLocaleString();
        
        // Get region info
        const regionInfo = report.region ? `<p style="margin: 10px 0 5px 0; color: #666;">Region</p>
          <p style="margin: 0;">${report.region}</p>` : '';
        
        // Add status info if available
        const statusInfo = report.status ? `<p style="margin: 10px 0 5px 0; color: #666;">Status</p>
          <p style="margin: 0;">${report.status}</p>` : '';
        
        // Add location source info
        let locationSourceInfo = '';
        if (report.coordinateSource === 'extracted') {
          locationSourceInfo = '<span style="color: green;">(✓ Exact coordinates)</span>';
        } else if (report.coordinateSource === 'geocoded') {
          locationSourceInfo = '<span style="color: orange;">(⌖ Geocoded address)</span>';
        } else {
          locationSourceInfo = '<span style="color: red;">(⚠ Approximate location)</span>';
        }
        
        // Create marker with popup
        const marker = L.marker([report.latitude, report.longitude], { icon });
        
        // Popup content
        const popupContent = document.createElement('div');
        popupContent.innerHTML = `
          <div style="min-width: 200px; padding: 10px;">
            <h4 style="margin: 0 0 10px 0;">Damage Report #${report.reportId || report._id}</h4>
            <hr style="margin: 5px 0;" />
            <p style="margin: 5px 0; color: #666;">Severity</p>
            <span style="display: inline-block; padding: 2px 8px; margin: 3px 0; background-color: ${markerColor}; color: white; border-radius: 12px; font-size: 12px;">${report.severity}</span>
            <p style="margin: 10px 0 5px 0; color: #666;">Type</p>
            <p style="margin: 0;">${report.damageType}</p>
            ${regionInfo}
            ${statusInfo}
            <p style="margin: 10px 0 5px 0; color: #666;">Reported</p>
            <p style="margin: 0;">${reportDate}</p>
            <p style="margin: 10px 0 5px 0; color: #666;">Location ${locationSourceInfo}</p>
            <p style="margin: 0; font-size: 12px;">${report.originalLocation}</p>
            <p style="margin: 5px 0 0 0; font-size: 12px;">(${report.latitude.toFixed(6)}, ${report.longitude.toFixed(6)})</p>
            <button style="margin-top: 10px; padding: 5px 10px; background-color: #1976d2; color: white; border: none; border-radius: 4px; cursor: pointer;">View Details</button>
          </div>
        `;
        
        // Add click event for the details button
        const detailsButton = popupContent.querySelector('button');
        detailsButton.addEventListener('click', () => {
          setSelectedReport(report);
          setDrawerOpen(true);
        });
        
        marker.bindPopup(popupContent);
        
        marker.on('click', () => {
          // We'll still set the selected report, but not open the drawer
          // until the user clicks the "View Details" button
          setSelectedReport(report);
        });
        
        // Add marker to cluster layer
        clusterLayerRef.current.addLayer(marker);
        
        // Add point to heatmap data
        // Intensity based on severity: High = 1.0, Medium = 0.7, Low = 0.3
        let intensity = 0.7;
        if (report.severity === 'High' || report.severity === 'HIGH') intensity = 1.0;
        else if (report.severity === 'Low' || report.severity === 'LOW') intensity = 0.3;
        
        // Reduce intensity slightly for approximate locations
        if (report.coordinateSource === 'default') {
          intensity *= 0.7;
        }
        
        heatPoints.push([report.latitude, report.longitude, intensity]);
      });
      
      // Create heatmap layer (but don't add to map yet)
      if (heatPoints.length > 0) {
        console.log(`Creating heatmap with ${heatPoints.length} points`);
        heatLayerRef.current = L.heatLayer(heatPoints, {
          radius: 25,
          blur: 15,
          maxZoom: 17,
          gradient: {
            0.3: '#3388ff',
            0.5: '#ffaa00',
            0.7: '#ff3300',
            0.9: '#bd0026'
          }
        });
        
        // Add heatmap layer if enabled
        if (showHeatmap) {
          heatLayerRef.current.addTo(mapInstanceRef.current);
        }
      } else {
        console.warn('No valid points found for heatmap');
      }
      
      // Auto-fit bounds if we have reports with valid coordinates
      if (damageReports.length > 0) {
        const reportsWithCoords = damageReports.filter(report => report.latitude && report.longitude);
        if (reportsWithCoords.length > 0) {
          const bounds = L.latLngBounds(reportsWithCoords.map(report => [report.latitude, report.longitude]));
          mapInstanceRef.current.fitBounds(bounds, { padding: [50, 50] });
        } else {
          console.warn('No reports with valid coordinates to fit bounds');
        }
      }
    }
  }, [damageReports, showHeatmap, getSeverityColor]);

  // Toggle heatmap layer when showHeatmap changes
  useEffect(() => {
    if (mapInstanceRef.current && heatLayerRef.current) {
      console.log(`Toggling heatmap visibility: ${showHeatmap}`);
      if (showHeatmap) {
        heatLayerRef.current.addTo(mapInstanceRef.current);
      } else if (mapInstanceRef.current.hasLayer(heatLayerRef.current)) {
        mapInstanceRef.current.removeLayer(heatLayerRef.current);
      }
    } else if (showHeatmap && !heatLayerRef.current) {
      // Create points for heatmap if the user activates it but we don't have a heatmap yet
      const heatPoints = damageReports
        .filter(report => report.latitude && report.longitude)
        .map(report => {
          // Intensity based on severity
          let intensity = 0.7; // default for Medium
          if (report.severity === 'High' || report.severity === 'HIGH') intensity = 1.0;
          else if (report.severity === 'Low' || report.severity === 'LOW') intensity = 0.3;
          
          return [report.latitude, report.longitude, intensity];
        });
      
      if (heatPoints.length > 0 && mapInstanceRef.current) {
        console.log(`Creating heatmap on toggle with ${heatPoints.length} points`);
        heatLayerRef.current = L.heatLayer(heatPoints, {
          radius: 25,
          blur: 15,
          maxZoom: 17,
          gradient: {
            0.3: '#3388ff',
            0.5: '#ffaa00',
            0.7: '#ff3300',
            0.9: '#bd0026'
          }
        });
        
        heatLayerRef.current.addTo(mapInstanceRef.current);
      } else {
        console.warn('Cannot create heatmap: no valid points or map instance');
      }
    }
  }, [showHeatmap, damageReports]);

  const handleStyleChange = (event) => {
    setMapStyle(event.target.value);
  };
  
  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  const clearFilters = () => {
    setFilters({
      severity: '',
      damageType: '',
      startDate: null,
      endDate: null,
      region: ''
    });
  };
  
  const handleDrawerClose = () => {
    setDrawerOpen(false);
    setSelectedReport(null);
  };
  
  const refreshData = () => {
    // Re-fetch data with current filters
    const currentFilters = {...filters};
    setFilters({...currentFilters});
  };

  return (
    <>
      <Paper sx={{ p: 1.5, mb: 2, borderRadius: 1, boxShadow: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
          <Stack direction="row" spacing={0.5} sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0.5 }}>
            <Typography variant="subtitle1" component="h2" sx={{ mr: 1.5, fontWeight: 500 }}>
              Damage Report Map
            </Typography>
            <Chip 
              icon={<WarningIcon sx={{ fontSize: '0.8rem' }} />} 
              label="High" 
              color="error" 
              size="small" 
              sx={{ height: 24, '& .MuiChip-label': { px: 1, fontSize: '0.75rem' } }} 
            />
            <Chip 
              icon={<WarningIcon sx={{ fontSize: '0.8rem' }} />} 
              label="Medium" 
              color="warning" 
              size="small" 
              sx={{ height: 24, '& .MuiChip-label': { px: 1, fontSize: '0.75rem' } }}
            />
            <Chip 
              icon={<WarningIcon sx={{ fontSize: '0.8rem' }} />} 
              label="Low" 
              color="info" 
              size="small" 
              sx={{ height: 24, '& .MuiChip-label': { px: 1, fontSize: '0.75rem' } }}
            />
            {loading && <CircularProgress size={18} sx={{ ml: 1.5 }} />}
          </Stack>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <FormControlLabel
              control={<Switch checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} size="small" />}
              label={<Box sx={{ display: 'flex', alignItems: 'center' }}>
                <HeatmapIcon sx={{ mr: 0.5, fontSize: '1rem' }} />
                <Typography variant="caption">Heatmap</Typography>
              </Box>}
              sx={{ mr: 1, '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
            />
            <FormControlLabel
              control={<Switch checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} size="small" />}
              label={<Typography variant="caption">Debug</Typography>}
              sx={{ mr: 1, '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
            />
            <Button 
              variant="outlined" 
              color="primary" 
              startIcon={<FilterListIcon sx={{ fontSize: '1rem' }} />}
              onClick={() => setFilterDrawerOpen(true)}
              size="small"
              sx={{ height: 32, fontSize: '0.8125rem', textTransform: 'none' }}
            >
              Filters
            </Button>
            <Button
              variant="outlined"
              color="primary"
              startIcon={<RefreshIcon sx={{ fontSize: '1rem' }} />}
              onClick={refreshData}
              size="small"
              sx={{ height: 32, fontSize: '0.8125rem', textTransform: 'none' }}
            >
              Refresh
            </Button>
            <FormControl sx={{ minWidth: 120 }} size="small">
              <InputLabel>Map Style</InputLabel>
              <Select
                value={mapStyle}
                label="Map Style"
                onChange={handleStyleChange}
                startAdornment={<LayersIcon sx={{ mr: 0.5, fontSize: '1rem', color: theme.palette.primary.main }} />}
                sx={{ height: 32, fontSize: '0.8125rem' }}
              >
                {mapStyles.map(style => (
                  <MenuItem key={style.value} value={style.value}>
                    {style.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Box>

        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 1.5, py: 0.5, '& .MuiAlert-message': { fontSize: '0.8125rem' } }}
            action={
              <Button color="inherit" size="small" sx={{ fontSize: '0.75rem', py: 0.25 }} onClick={refreshData}>
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        )}
        
        {showDebug && (
          <Paper sx={{ p: 1, mb: 1.5, bgcolor: 'rgba(0,0,0,0.02)', borderRadius: 1, overflow: 'auto', maxHeight: '150px', border: '1px solid rgba(0,0,0,0.05)' }}>
            <Typography variant="caption" sx={{ display: 'block', fontWeight: 500, mb: 0.5, color: 'text.secondary' }}>Debug Information</Typography>
            <Grid container spacing={1} sx={{ '& .MuiGrid-item': { p: 0.5 } }}>
              <Grid item xs={4}>
                <Box sx={{ bgcolor: 'background.paper', p: 0.75, borderRadius: 0.5, border: '1px solid rgba(0,0,0,0.06)' }}>
                  <Typography variant="caption" color="text.secondary">Reports</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>{damageReports.length}</Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box sx={{ bgcolor: 'background.paper', p: 0.75, borderRadius: 0.5, border: '1px solid rgba(0,0,0,0.06)' }}>
                  <Typography variant="caption" color="text.secondary">Markers</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>{damageReports.filter(r => r.latitude && r.longitude).length}</Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box sx={{ bgcolor: 'background.paper', p: 0.75, borderRadius: 0.5, border: '1px solid rgba(0,0,0,0.06)' }}>
                  <Typography variant="caption" color="text.secondary">Valid Coordinates</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>{damageReports.filter(r => r.latitude && r.longitude).length} / {damageReports.length}</Typography>
                </Box>
              </Grid>
            </Grid>
            
            {damageReports.length > 0 && (
              <Box sx={{ mt: 0.75 }}>
                <Typography variant="caption" sx={{ fontWeight: 500, color: 'text.secondary' }}>First Report Location:</Typography>
                <Box component="pre" sx={{ fontSize: '10px', mt: 0.25, p: 0.75, bgcolor: 'background.paper', borderRadius: 0.5, overflowX: 'auto', border: '1px solid rgba(0,0,0,0.06)' }}>
                  {JSON.stringify(damageReports[0]?.originalLocation || {}, null, 2)}
                </Box>
                <Box sx={{ display: 'flex', mt: 0.5, gap: 1 }}>
                  <Box sx={{ flex: 1, p: 0.5, bgcolor: 'background.paper', borderRadius: 0.5, fontSize: '10px', border: '1px solid rgba(0,0,0,0.06)' }}>
                    <Typography variant="caption" sx={{ fontWeight: 500, color: 'primary.main', fontSize: '9px' }}>Lat:</Typography> {damageReports[0]?.latitude?.toFixed(6)}
                  </Box>
                  <Box sx={{ flex: 1, p: 0.5, bgcolor: 'background.paper', borderRadius: 0.5, fontSize: '10px', border: '1px solid rgba(0,0,0,0.06)' }}>
                    <Typography variant="caption" sx={{ fontWeight: 500, color: 'primary.main', fontSize: '9px' }}>Lng:</Typography> {damageReports[0]?.longitude?.toFixed(6)}
                  </Box>
                </Box>
              </Box>
            )}
          </Paper>
        )}
        
        <Box sx={{ height: '72vh', width: '100%', borderRadius: 0.5, overflow: 'hidden', border: '1px solid rgba(0,0,0,0.08)' }}>
          <div ref={mapRef} style={{ width: '100%', height: '100%' }}></div>
        </Box>
        
        <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center', px: 0.5 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <InfoIcon sx={{ fontSize: '0.875rem', opacity: 0.7 }} />
            {damageReports.length} damage report{damageReports.length !== 1 ? 's' : ''} displayed
          </Typography>
          {selectedReport && (
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => setDrawerOpen(true)}
              startIcon={<DashboardIcon sx={{ fontSize: '0.875rem' }} />}
              size="small"
              sx={{ fontSize: '0.75rem', py: 0.5, textTransform: 'none' }}
            >
              View Selected Report
            </Button>
          )}
        </Box>
      </Paper>
      
      {/* Report Details Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={handleDrawerClose}
        sx={{
          width: 400,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 400,
            boxSizing: 'border-box',
            p: 2,
            bgcolor: '#FAFAFA'
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 500, fontSize: '1rem', letterSpacing: '0.0075em' }}>
            Report Details
          </Typography>
          <IconButton size="small" onClick={handleDrawerClose} sx={{ p: 0.5 }}>
            <CloseIcon sx={{ fontSize: '1.25rem' }} />
          </IconButton>
        </Box>
        <Divider sx={{ mb: 1.5 }} />
        {selectedReport && <ViewDamageReport report={selectedReport} />}
      </Drawer>
      
      {/* Filters Drawer */}
      <Drawer
        anchor="right"
        open={filterDrawerOpen}
        onClose={() => setFilterDrawerOpen(false)}
        sx={{
          width: 300,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 300,
            boxSizing: 'border-box',
            p: 2,
            bgcolor: '#FAFAFA'
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 500, fontSize: '1rem', letterSpacing: '0.0075em' }}>
            Filter Reports
          </Typography>
          <IconButton size="small" onClick={() => setFilterDrawerOpen(false)} sx={{ p: 0.5 }}>
            <CloseIcon sx={{ fontSize: '1.25rem' }} />
          </IconButton>
        </Box>
        <Divider sx={{ mb: 2 }} />
        
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                <InputLabel>Severity</InputLabel>
                <Select
                  value={filters.severity}
                  label="Severity"
                  onChange={(e) => handleFilterChange('severity', e.target.value)}
                  sx={{ fontSize: '0.875rem' }}
                >
                  <MenuItem value="">All Severities</MenuItem>
                  <MenuItem value="HIGH">High</MenuItem>
                  <MenuItem value="MEDIUM">Medium</MenuItem>
                  <MenuItem value="LOW">Low</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                <InputLabel>Damage Type</InputLabel>
                <Select
                  value={filters.damageType}
                  label="Damage Type"
                  onChange={(e) => handleFilterChange('damageType', e.target.value)}
                  sx={{ fontSize: '0.875rem' }}
                >
                  <MenuItem value="">All Types</MenuItem>
                  {damageTypes.map(type => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                <InputLabel>Region</InputLabel>
                <Select
                  value={filters.region}
                  label="Region"
                  onChange={(e) => handleFilterChange('region', e.target.value)}
                  sx={{ fontSize: '0.875rem' }}
                >
                  <MenuItem value="">All Regions</MenuItem>
                  {regions.map(region => (
                    <MenuItem key={region} value={region}>{region}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <DatePicker
                label="Start Date"
                value={filters.startDate}
                onChange={(date) => handleFilterChange('startDate', date)}
                renderInput={(params) => <TextField {...params} fullWidth size="small" sx={{ mb: 1.5, fontSize: '0.875rem' }} />}
                inputFormat="MM/dd/yyyy"
              />
            </Grid>
            
            <Grid item xs={12}>
              <DatePicker
                label="End Date"
                value={filters.endDate}
                onChange={(date) => handleFilterChange('endDate', date)}
                renderInput={(params) => <TextField {...params} fullWidth size="small" sx={{ fontSize: '0.875rem' }} />}
                inputFormat="MM/dd/yyyy"
              />
            </Grid>
            
            <Grid item xs={12} sx={{ mt: 1 }}>
              <Button 
                variant="contained" 
                color="primary" 
                fullWidth
                size="small"
                onClick={() => setFilterDrawerOpen(false)}
                sx={{ textTransform: 'none', py: 0.75, fontSize: '0.875rem' }}
              >
                Apply Filters
              </Button>
              <Button 
                variant="outlined" 
                fullWidth 
                size="small"
                onClick={clearFilters}
                sx={{ mt: 1.5, textTransform: 'none', py: 0.75, fontSize: '0.875rem' }}
              >
                Clear Filters
              </Button>
            </Grid>
          </Grid>
        </LocalizationProvider>
      </Drawer>
      
      {/* Debug Panel */}
      <Drawer
        anchor="bottom"
        open={showDebug}
        onClose={() => setShowDebug(false)}
        sx={{
          height: '40%',
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            height: '40%',
            boxSizing: 'border-box',
            p: 1.5,
            bgcolor: '#FAFAFA'
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 500, fontSize: '1rem', letterSpacing: '0.0075em', display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <InfoIcon sx={{ fontSize: '1rem', color: 'primary.main' }} />
            Debug Information
          </Typography>
          <IconButton size="small" onClick={() => setShowDebug(false)} sx={{ p: 0.5 }}>
            <CloseIcon sx={{ fontSize: '1.25rem' }} />
          </IconButton>
        </Box>
        
        <Divider sx={{ mb: 1.5 }} />
        
        <Grid container spacing={1} sx={{ mb: 1.5 }}>
          <Grid item xs={3}>
            <Paper sx={{ p: 1, bgcolor: 'background.paper', border: '1px solid rgba(25, 118, 210, 0.1)', borderLeft: '3px solid #1976d2', height: '100%' }}>
              <Typography variant="caption" color="text.secondary">Total Reports</Typography>
              <Typography variant="h6" sx={{ fontWeight: 500 }}>{damageReports.length}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={3}>
            <Paper sx={{ p: 1, bgcolor: 'background.paper', border: '1px solid rgba(76, 175, 80, 0.1)', borderLeft: '3px solid #4caf50', height: '100%' }}>
              <Typography variant="caption" color="text.secondary">Extracted</Typography>
              <Typography variant="h6" sx={{ fontWeight: 500, color: '#4caf50' }}>{damageReports.filter(r => r.coordinateSource === 'extracted').length}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={3}>
            <Paper sx={{ p: 1, bgcolor: 'background.paper', border: '1px solid rgba(255, 152, 0, 0.1)', borderLeft: '3px solid #ff9800', height: '100%' }}>
              <Typography variant="caption" color="text.secondary">Geocoded</Typography>
              <Typography variant="h6" sx={{ fontWeight: 500, color: '#ff9800' }}>{damageReports.filter(r => r.coordinateSource === 'geocoded').length}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={3}>
            <Paper sx={{ p: 1, bgcolor: 'background.paper', border: '1px solid rgba(244, 67, 54, 0.1)', borderLeft: '3px solid #f44336', height: '100%' }}>
              <Typography variant="caption" color="text.secondary">Default</Typography>
              <Typography variant="h6" sx={{ fontWeight: 500, color: '#f44336' }}>{damageReports.filter(r => r.coordinateSource === 'default').length}</Typography>
            </Paper>
          </Grid>
        </Grid>
        
        <Typography variant="caption" sx={{ display: 'block', fontWeight: 500, color: 'text.secondary', mb: 0.5 }}>
          Coordinate Extraction Analysis
        </Typography>
        
        <Box sx={{ height: 'calc(100% - 150px)', overflowY: 'auto', mb: 1, pr: 1 }}>
          {damageReports.length === 0 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', bgcolor: 'background.paper', borderRadius: 0.5, border: '1px dashed rgba(0,0,0,0.1)' }}>
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', p: 2, fontSize: '0.875rem' }}>
                No damage reports available.<br />Fetch reports to see debug information.
              </Typography>
            </Box>
          )}
          
          {damageReports.map((report, index) => (
            <Paper 
              key={report._id} 
              elevation={0}
              sx={{ 
                mb: 1, 
                p: 1, 
                borderRadius: 0.5, 
                border: '1px solid rgba(0,0,0,0.08)',
                borderLeft: report.coordinateSource === 'extracted' 
                  ? '3px solid #4caf50' 
                  : report.coordinateSource === 'geocoded' 
                    ? '3px solid #ff9800' 
                    : '3px solid #f44336',
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Report #{report.reportId || report._id?.substring(0, 8)} 
                </Typography>
                <Chip 
                  label={
                    report.coordinateSource === 'extracted' 
                      ? 'Exact' 
                      : report.coordinateSource === 'geocoded' 
                        ? 'Geocoded' 
                        : 'Approx'
                  }
                  size="small"
                  sx={{ 
                    height: 18, 
                    '& .MuiChip-label': { px: 0.5, py: 0, fontSize: '0.625rem', fontWeight: 500 },
                    bgcolor: report.coordinateSource === 'extracted' 
                      ? 'rgba(76, 175, 80, 0.1)' 
                      : report.coordinateSource === 'geocoded' 
                        ? 'rgba(255, 152, 0, 0.1)' 
                        : 'rgba(244, 67, 54, 0.1)',
                    color: report.coordinateSource === 'extracted' 
                      ? '#4caf50' 
                      : report.coordinateSource === 'geocoded' 
                        ? '#ff9800' 
                        : '#f44336',
                    borderRadius: 0.5,
                  }}
                />
              </Box>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                <Box sx={{ 
                  p: 0.5, 
                  bgcolor: 'background.paper', 
                  borderRadius: 0.5, 
                  border: '1px solid rgba(0,0,0,0.05)',
                  fontSize: '0.75rem',
                  flex: '1 1 45%',
                  minWidth: '150px'
                }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Lat:</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem', fontFamily: 'monospace' }}>{report.latitude?.toFixed(6) || 'N/A'}</Typography>
                </Box>
                
                <Box sx={{ 
                  p: 0.5, 
                  bgcolor: 'background.paper', 
                  borderRadius: 0.5, 
                  border: '1px solid rgba(0,0,0,0.05)',
                  fontSize: '0.75rem',
                  flex: '1 1 45%',
                  minWidth: '150px'
                }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Lng:</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem', fontFamily: 'monospace' }}>{report.longitude?.toFixed(6) || 'N/A'}</Typography>
                </Box>
                
                <Box sx={{ 
                  p: 0.5, 
                  bgcolor: 'background.paper', 
                  borderRadius: 0.5, 
                  border: '1px solid rgba(0,0,0,0.05)',
                  fontSize: '0.75rem',
                  width: '100%'
                }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>Original Location:</Typography>
                  <Typography variant="body2" sx={{ 
                    fontSize: '0.75rem', 
                    fontFamily: 'monospace', 
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis' 
                  }}>
                    {typeof report.originalLocation === 'object' 
                      ? JSON.stringify(report.originalLocation) 
                      : report.originalLocation || 'N/A'}
                  </Typography>
                </Box>
              </Box>
            </Paper>
          ))}
        </Box>
        
        <Button 
          variant="outlined" 
          color="primary" 
          size="small"
          onClick={() => setShowDebug(false)}
          sx={{ width: '100%', textTransform: 'none', fontSize: '0.875rem' }}
          startIcon={<CloseIcon sx={{ fontSize: '0.875rem' }} />}
        >
          Close Debug Panel
        </Button>
      </Drawer>
    </>
  );
}

export default MapView;