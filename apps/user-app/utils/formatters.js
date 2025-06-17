import { getAuthToken, getFieldWorkerData } from './auth';

// Format timestamp to readable date
export const formatDate = (timestamp) => {
  if (!timestamp) return 'N/A';
  
  const date = new Date(timestamp);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

// Format timestamp to include time
export const formatDateTime = (timestamp) => {
  if (!timestamp) return 'N/A';
  
  const date = new Date(timestamp);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

// Format API error messages
export const formatErrorMessage = (error) => {
  if (!error) return 'An unknown error occurred';
  
  if (typeof error === 'string') return error;
  
  if (error.message) return error.message;
  
  return 'An error occurred. Please try again.';
};

// Get color code based on damage severity
export const getSeverityColor = (severity) => {
  switch (severity?.toLowerCase()) {
    case 'critical':
      return '#FF3B30'; // Red
    case 'high':
      return '#FF9500'; // Orange
    case 'medium':
      return '#FFCC00'; // Yellow
    case 'low':
      return '#34C759'; // Green
    default:
      return '#8E8E93'; // Gray
  }
};

// Get color code based on report status
export const getStatusColor = (status) => {
  switch (status?.toLowerCase()) {
    case 'pending':
      return '#FF9500'; // Orange
    case 'assigned':
      return '#5AC8FA'; // Blue
    case 'in_progress':
      return '#5856D6'; // Purple
    case 'completed':
      return '#34C759'; // Green
    case 'rejected':
      return '#FF3B30'; // Red
    default:
      return '#8E8E93'; // Gray
  }
};

// Format status text for display
export const formatStatus = (status) => {
  if (!status) return 'Unknown';
  
  return status
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};
