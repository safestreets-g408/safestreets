
export const formatLocation = (location) => {
  if (!location) return 'Location not available';
  
  // If location is a string that might contain an address
  if (typeof location === 'string') {
    // Check if it's likely to be a coordinate string
    const coordMatch = location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
    if (coordMatch) {
      // It's a coordinate string like "12.345, 67.890"
      return `Lat: ${parseFloat(coordMatch[1]).toFixed(6)}, Long: ${parseFloat(coordMatch[2]).toFixed(6)}`;
    }
    // Return the string as an address
    return location;
  }
  
  // If it's an object with address property
  if (location.address) return location.address;
  
  // If it's an object with coordinates array (GeoJSON format)
  if (location.coordinates && location.coordinates.length === 2) {
    // Check if coordinates are valid numbers before using toFixed
    const lat = typeof location.coordinates[1] === 'number' ? location.coordinates[1].toFixed(6) : 'N/A';
    const lng = typeof location.coordinates[0] === 'number' ? location.coordinates[0].toFixed(6) : 'N/A';
    return `Lat: ${lat}, Long: ${lng}`;
  }
  
  // If it has lat/lng properties directly
  if ((location.lat !== undefined && location.lng !== undefined) || 
      (location.latitude !== undefined && location.longitude !== undefined)) {
    const lat = location.lat || location.latitude;
    const lng = location.lng || location.longitude;
    return `Lat: ${parseFloat(lat).toFixed(6)}, Long: ${parseFloat(lng).toFixed(6)}`;
  }
  
  return 'Location not available';
};


export const getCoordinatesString = (location) => {
  if (!location) return 'Coordinates not available';
  
  // If location is a string with coordinates
  if (typeof location === 'string') {
    const coordMatch = location.match(/([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)/);
    if (coordMatch) {
      return `${parseFloat(coordMatch[1]).toFixed(6)}, ${parseFloat(coordMatch[2]).toFixed(6)}`;
    }
    return 'Coordinates not available';
  }
  
  // If it's an object with coordinates array (GeoJSON format)
  if (location.coordinates && location.coordinates.length === 2) {
    const [longitude, latitude] = location.coordinates;
    // Check if coordinates are valid numbers before using toFixed
    const lat = typeof latitude === 'number' ? latitude.toFixed(6) : 'N/A';
    const lng = typeof longitude === 'number' ? longitude.toFixed(6) : 'N/A';
    return `${lat}, ${lng}`;
  }
  
  // If it has lat/lng properties directly
  if ((location.lat !== undefined && location.lng !== undefined) || 
      (location.latitude !== undefined && location.longitude !== undefined)) {
    const lat = location.lat || location.latitude;
    const lng = location.lng || location.longitude;
    return `${parseFloat(lat).toFixed(6)}, ${parseFloat(lng).toFixed(6)}`;
  }
  
  return 'Coordinates not available';
};


export const formatDate = (date, includeTime = true) => {
  if (!date) return '';
  
  const dateObj = new Date(date);
  const options = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...(includeTime && {
      hour: '2-digit',
      minute: '2-digit'
    })
  };
  
  return dateObj.toLocaleDateString('en-US', options);
};
