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
  
  // Helper function to extract coordinates from any location format
  const extractCoordinates = useCallback((location) => {
    // If location is not defined, return null
    if (!location) return null;
    
    console.log('Extracting coordinates from:', typeof location, location);
    
    // If it's already a valid pair of coordinates, return them
    if (Array.isArray(location) && location.length === 2 &&
        typeof location[0] === 'number' && typeof location[1] === 'number') {
      console.log('Found array coordinates:', location);
      return { latitude: location[1], longitude: location[0] };
    }
    
    // If it's a string, try to parse it
    if (typeof location === 'string') {
      try {
        // Try to parse as JSON
        const parsed = JSON.parse(location);
        console.log('Parsed location JSON:', parsed);
        
        // If parsed is an object with coordinates array property (GeoJSON format)
        if (parsed.coordinates && Array.isArray(parsed.coordinates) && parsed.coordinates.length === 2) {
          // GeoJSON format is [longitude, latitude]
          return {
            latitude: parsed.coordinates[1],
            longitude: parsed.coordinates[0]
          };
        }
        
        // If parsed has lat/lng or latitude/longitude properties
        if ((parsed.lat !== undefined && parsed.lng !== undefined) ||
            (parsed.latitude !== undefined && parsed.longitude !== undefined)) {
          return {
            latitude: parsed.lat || parsed.latitude,
            longitude: parsed.lng || parsed.longitude
          };
        }
        
        // Recursively try to extract from the parsed object
        return extractCoordinates(parsed);
      } catch (e) {
        // Not JSON, check for coordinate pattern in string
        // Match patterns like "17.3850, 78.4866" or "17.3850 78.4866"
        const coordMatch = location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
        if (coordMatch) {
          console.log('Found coordinate pattern in string:', coordMatch[1], coordMatch[2]);
          return { 
            latitude: parseFloat(coordMatch[1]), 
            longitude: parseFloat(coordMatch[2])
          };
        }
      }
    }
    
    // Handle GeoJSON format
    if (location.coordinates && Array.isArray(location.coordinates) && location.coordinates.length === 2) {
      console.log('Found GeoJSON coordinates:', location.coordinates);
      // GeoJSON format is [longitude, latitude]
      return {
        latitude: location.coordinates[1],
        longitude: location.coordinates[0]
      };
    }
    
    // Handle lat/lng properties
    if ((location.lat !== undefined && location.lng !== undefined) ||
        (location.latitude !== undefined && location.longitude !== undefined)) {
      console.log('Found lat/lng properties:', location.lat || location.latitude, location.lng || location.longitude);
      return {
        latitude: location.lat || location.latitude,
        longitude: location.lng || location.longitude
      };
    }
    
    // Handle nested location object common in MongoDB GeoJSON responses
    if (location.location && location.location.coordinates) {
      console.log('Found nested location coordinates:', location.location.coordinates);
      return {
        latitude: location.location.coordinates[1],
        longitude: location.location.coordinates[0]
      };
    }
    
    console.warn('Failed to extract coordinates from:', location);
    return null;
  }, []);
  
  // Helper function to geocode addresses using OpenStreetMap Nominatim API
  const geocodeAddress = useCallback(async (address) => {
    if (!address) return null;
    
    try {
      // Use Nominatim API to geocode the address
      const encodedAddress = encodeURIComponent(address);
      const response = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodedAddress}&format=json&limit=1&addressdetails=0`);
      
      if (!response.ok) {
        console.error('Geocoding API error:', response.statusText);
        return null;
      }
      
      const data = await response.json();
      
      if (data && data.length > 0) {
        const result = data[0];
        console.log('Geocoded address:', address, 'to coordinates:', result.lat, result.lon);
        return {
          latitude: parseFloat(result.lat),
          longitude: parseFloat(result.lon)
        };
      } else {
        console.warn('No geocoding results found for address:', address);
        return null;
      }
    } catch (error) {
      console.error('Geocoding error:', error);
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
        
        // Process and geocode reports if needed
        const processedReportsPromises = reports.map(async (report) => {
          try {
            // First try to extract coordinates using our existing function
            const coordinates = extractCoordinates(report.location);
            
            if (coordinates) {
              console.log(`Found coordinates for report ${report.reportId || report._id}:`, coordinates);
              return {
                ...report,
                latitude: coordinates.latitude,
                longitude: coordinates.longitude,
                originalLocation: report.location,
                coordinateSource: 'extracted'
              };
            }
            
            // If location is a string and looks like an address, try geocoding
            if (typeof report.location === 'string' && report.location.length > 0) {
              console.log(`Geocoding address for report ${report.reportId || report._id}:`, report.location);
              const geocoded = await geocodeAddress(report.location);
              
              if (geocoded) {
                return {
                  ...report,
                  latitude: geocoded.latitude,
                  longitude: geocoded.longitude,
                  originalLocation: report.location,
                  coordinateSource: 'geocoded'
                };
              }
            }
            
            // If all else fails, use default location for Hyderabad with a slight randomization
            // to show something on the map rather than nothing
            console.warn(`Using default location for report ${report.reportId || report._id}`);
            return {
              ...report,
              latitude: 17.3850 + (Math.random() - 0.5) * 0.05,
              longitude: 78.4867 + (Math.random() - 0.5) * 0.05,
              originalLocation: report.location,
              coordinateSource: 'default'
            };
          } catch (err) {
            console.error(`Error processing report ${report.reportId || report._id}:`, err);
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
      <Paper sx={{ p: 2, mb: 3, borderRadius: 2, boxShadow: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Stack direction="row" spacing={1} sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
            <Typography variant="h6" component="h2" sx={{ mr: 2 }}>
              Damage Report Map
            </Typography>
            <Chip icon={<WarningIcon />} label="High Severity" color="error" />
            <Chip icon={<WarningIcon />} label="Medium Severity" color="warning" />
            <Chip icon={<WarningIcon />} label="Low Severity" color="info" />
            {loading && <CircularProgress size={24} sx={{ ml: 2 }} />}
          </Stack>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControlLabel
              control={<Switch checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />}
              label={<Box sx={{ display: 'flex', alignItems: 'center' }}>
                <HeatmapIcon sx={{ mr: 0.5 }} />
                Heatmap
              </Box>}
              sx={{ mr: 2 }}
            />
            <FormControlLabel
              control={<Switch checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />}
              label="Debug"
              sx={{ mr: 2 }}
            />
            <Button 
              variant="outlined" 
              color="primary" 
              startIcon={<FilterListIcon />}
              onClick={() => setFilterDrawerOpen(true)}
              sx={{ height: 40 }}
            >
              Filters
            </Button>
            <Button
              variant="outlined"
              color="primary"
              startIcon={<RefreshIcon />}
              onClick={refreshData}
              sx={{ height: 40 }}
            >
              Refresh
            </Button>
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Map Style</InputLabel>
              <Select
                value={mapStyle}
                label="Map Style"
                onChange={handleStyleChange}
                startAdornment={<LayersIcon sx={{ mr: 1, color: theme.palette.primary.main }} />}
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
            sx={{ mb: 2 }}
            action={
              <Button color="inherit" size="small" onClick={refreshData}>
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        )}
        
        {showDebug && (
          <Paper sx={{ p: 2, mb: 2, bgcolor: '#f5f5f5', borderRadius: 1, overflow: 'auto', maxHeight: '200px' }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Debug Information</Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Typography variant="caption">Reports Received: {damageReports.length}</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption">Markers Displayed: {damageReports.filter(r => r.latitude && r.longitude).length}</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption">Valid Coordinates: {damageReports.filter(r => r.latitude && r.longitude).length} / {damageReports.length}</Typography>
              </Grid>
            </Grid>
            
            {damageReports.length > 0 && (
              <>
                <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>First Report Location Data:</Typography>
                <Box component="pre" sx={{ fontSize: '11px', mt: 1, p: 1, bgcolor: '#e0e0e0', borderRadius: 1, overflowX: 'auto' }}>
                  {JSON.stringify(damageReports[0]?.originalLocation || {}, null, 2)}
                </Box>
                <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>Extracted Coordinates:</Typography>
                <Box component="pre" sx={{ fontSize: '11px', mt: 1, p: 1, bgcolor: '#e0e0e0', borderRadius: 1 }}>
                  Latitude: {damageReports[0]?.latitude}, Longitude: {damageReports[0]?.longitude}
                </Box>
              </>
            )}
          </Paper>
        )}
        
        <Box sx={{ height: '70vh', width: '100%', borderRadius: 1, overflow: 'hidden' }}>
          <div ref={mapRef} style={{ width: '100%', height: '100%' }}></div>
        </Box>
        
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            {damageReports.length} damage reports shown on map
          </Typography>
          {selectedReport && (
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => setDrawerOpen(true)}
              startIcon={<DashboardIcon />}
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
          width: 450,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 450,
            boxSizing: 'border-box',
            p: 3
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Report Details</Typography>
          <IconButton onClick={handleDrawerClose}>
            <CloseIcon />
          </IconButton>
        </Box>
        <Divider sx={{ mb: 2 }} />
        {selectedReport && <ViewDamageReport report={selectedReport} />}
      </Drawer>
      
      {/* Filters Drawer */}
      <Drawer
        anchor="right"
        open={filterDrawerOpen}
        onClose={() => setFilterDrawerOpen(false)}
        sx={{
          width: 320,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 320,
            boxSizing: 'border-box',
            p: 3
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Filter Reports</Typography>
          <IconButton onClick={() => setFilterDrawerOpen(false)}>
            <CloseIcon />
          </IconButton>
        </Box>
        <Divider sx={{ mb: 3 }} />
        
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Severity</InputLabel>
                <Select
                  value={filters.severity}
                  label="Severity"
                  onChange={(e) => handleFilterChange('severity', e.target.value)}
                >
                  <MenuItem value="">All Severities</MenuItem>
                  <MenuItem value="HIGH">High</MenuItem>
                  <MenuItem value="MEDIUM">Medium</MenuItem>
                  <MenuItem value="LOW">Low</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Damage Type</InputLabel>
                <Select
                  value={filters.damageType}
                  label="Damage Type"
                  onChange={(e) => handleFilterChange('damageType', e.target.value)}
                >
                  <MenuItem value="">All Types</MenuItem>
                  {damageTypes.map(type => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Region</InputLabel>
                <Select
                  value={filters.region}
                  label="Region"
                  onChange={(e) => handleFilterChange('region', e.target.value)}
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
                renderInput={(params) => <TextField {...params} fullWidth />}
                inputFormat="MM/dd/yyyy"
              />
            </Grid>
            
            <Grid item xs={12}>
              <DatePicker
                label="End Date"
                value={filters.endDate}
                onChange={(date) => handleFilterChange('endDate', date)}
                renderInput={(params) => <TextField {...params} fullWidth />}
                inputFormat="MM/dd/yyyy"
              />
            </Grid>
            
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Button 
                variant="contained" 
                color="primary" 
                fullWidth
                onClick={() => setFilterDrawerOpen(false)}
              >
                Apply Filters
              </Button>
              <Button 
                variant="outlined" 
                fullWidth 
                onClick={clearFilters}
                sx={{ mt: 2 }}
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
          height: '50%',
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            height: '50%',
            boxSizing: 'border-box',
            p: 2
          },
        }}
      >
        <Typography variant="h6" sx={{ mb: 2 }}>
          Debug Panel
        </Typography>
        
        <Divider sx={{ mb: 2 }} />
        
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={4}>
            <Paper sx={{ p: 1, bgcolor: '#f0f7ff' }}>
              <Typography variant="subtitle2">Total Reports</Typography>
              <Typography variant="h4">{damageReports.length}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={4}>
            <Paper sx={{ p: 1, bgcolor: '#fff8e1' }}>
              <Typography variant="subtitle2">Extracted Coordinates</Typography>
              <Typography variant="h4">{damageReports.filter(r => r.coordinateSource === 'extracted').length}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={4}>
            <Paper sx={{ p: 1, bgcolor: '#fff1f0' }}>
              <Typography variant="subtitle2">Geocoded Addresses</Typography>
              <Typography variant="h4">{damageReports.filter(r => r.coordinateSource === 'geocoded').length}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={4}>
            <Paper sx={{ p: 1, bgcolor: '#f9fbe7' }}>
              <Typography variant="subtitle2">Default Locations</Typography>
              <Typography variant="h4">{damageReports.filter(r => r.coordinateSource === 'default').length}</Typography>
            </Paper>
          </Grid>
        </Grid>
        
        <Typography variant="subtitle1" sx={{ mb: 1 }}>
          Coordinate Extraction Debug Info
        </Typography>
        
        <Box sx={{ maxHeight: '300px', overflowY: 'auto', mb: 2 }}>
          {damageReports.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No damage reports available. Fetch reports to see debug information.
            </Typography>
          )}
          
          {damageReports.map((report, index) => (
            <Box 
              key={report._id} 
              sx={{ 
                mb: 2, 
                p: 2, 
                borderRadius: 1, 
                border: '1px solid #ddd', 
                backgroundColor: 
                  report.coordinateSource === 'extracted' ? '#f0f7ff' : 
                  report.coordinateSource === 'geocoded' ? '#fff8e1' : '#fff1f0'
              }}
            >
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Report #{report.reportId || report._id} 
                {report.coordinateSource === 'extracted' && ' (Extracted Coordinates)'}
                {report.coordinateSource === 'geocoded' && ' (Geocoded Address)'}
                {report.coordinateSource === 'default' && ' (Default Location)'}
              </Typography>
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                <strong>Original Location:</strong> {JSON.stringify(report.originalLocation)}
              </Typography>
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                <strong>Extracted Latitude:</strong> {report.latitude.toFixed(6)}
              </Typography>
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                <strong>Extracted Longitude:</strong> {report.longitude.toFixed(6)}
              </Typography>
              <Typography variant="body2">
                <strong>Region:</strong> {report.region || 'N/A'}
              </Typography>
            </Box>
          ))}
        </Box>
        
        <Button 
          variant="outlined" 
          color="primary" 
          onClick={() => setShowDebug(false)}
          sx={{ width: '100%' }}
        >
          Close Debug Panel
        </Button>
      </Drawer>
    </>
  );
}

export default MapView;