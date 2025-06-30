import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Link
} from '@mui/material';
import AssignmentIcon from '@mui/icons-material/Assignment';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import PropTypes from 'prop-types';

const ReportMessage = ({ report }) => {
  // Get status color
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return '#10b981';
      case 'pending':
        return '#f59e0b';
      case 'in progress':
      case 'in_progress':
        return '#3b82f6';
      case 'cancelled':
        return '#ef4444';
      case 'on_hold':
      case 'on hold':
        return '#6b7280';
      default:
        return '#6b7280';
    }
  };
  
  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return <ErrorIcon fontSize="small" sx={{ color: '#ef4444' }} />;
      case 'medium':
        return <WarningIcon fontSize="small" sx={{ color: '#f59e0b' }} />;
      case 'low':
        return <InfoIcon fontSize="small" sx={{ color: '#3b82f6' }} />;
      default:
        return <InfoIcon fontSize="small" sx={{ color: '#6b7280' }} />;
    }
  };
  
  if (!report) return null;
  
  // Handle case when report data might be corrupted or missing fields
  if (!report || !report.reportId) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: 2,
          borderRadius: 2,
          border: '1px solid rgba(239, 68, 68, 0.2)',
          bgcolor: 'rgba(239, 68, 68, 0.05)',
          my: 1
        }}
      >
        <Typography variant="body2" color="error">
          This damage report could not be displayed correctly. The data may be corrupted.
        </Typography>
      </Paper>
    );
  }
  
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        borderRadius: 2,
        border: '1px solid rgba(59, 130, 246, 0.2)',
        bgcolor: 'rgba(59, 130, 246, 0.05)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
        my: 1,
        maxWidth: '100%'
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <AssignmentIcon sx={{ color: '#3b82f6' }} />
        <Typography variant="subtitle1" fontWeight={600} sx={{ color: '#3b82f6' }}>
          Damage Report: {report.reportId}
        </Typography>
        {report.status && (
          <Chip 
            label={report.status}
            size="small"
            sx={{ 
              color: 'white',
              bgcolor: getStatusColor(report.status),
              fontWeight: 500,
              ml: 'auto'
            }}
          />
        )}
      </Box>
      
      {report.damageType && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {report.severity && getSeverityIcon(report.severity)}
          <Typography variant="body2">
            {report.damageType} {report.severity && `(${report.severity} severity)`}
          </Typography>
        </Box>
      )}
      
      {report.location && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="body2">
            <strong>Location:</strong> {report.location} {report.region && `(${report.region})`}
          </Typography>
        </Box>
      )}
      
      {report.description && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            {report.description}
          </Typography>
        </Box>
      )}
      
      <Box sx={{ 
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        mt: 2,
        pt: 1,
        borderTop: '1px solid rgba(0,0,0,0.06)'
      }}>
        <Box>
          <Typography variant="caption" color="text.secondary">
            Action: {report.action || 'Not specified'}
          </Typography>
        </Box>
        <Link 
          href={`/reports/view/${report.reportId}`}
          color="primary"
          underline="hover"
          sx={{
            fontWeight: 500,
            fontSize: '0.875rem',
            '&:hover': {
              color: '#3b82f6'
            }
          }}
        >
          View Full Report
        </Link>
      </Box>
    </Paper>
  );
};

ReportMessage.propTypes = {
  report: PropTypes.shape({
    reportId: PropTypes.string.isRequired,
    status: PropTypes.string,
    severity: PropTypes.string,
    damageType: PropTypes.string,
    location: PropTypes.string,
    region: PropTypes.string,
    description: PropTypes.string,
    action: PropTypes.string
  }).isRequired
};

export default ReportMessage;
