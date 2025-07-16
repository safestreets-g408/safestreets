import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Divider
} from '@mui/material';
import {
  LocationOn as LocationIcon,
  CalendarToday as DateIcon,
  Assignment as ReportIcon
} from '@mui/icons-material';
import PropTypes from 'prop-types';
import { format } from 'date-fns';

const ReportMessage = ({ report }) => {
  // Format date with better handling of invalid dates
  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM dd, yyyy');
    } catch (error) {
      return 'Invalid date';
    }
  };

  if (!report) {
    return (
      <Box sx={{ p: 1 }}>
        <Typography variant="body2" color="error">
          Error: Report data is missing
        </Typography>
      </Box>
    );
  }

  return (
    <Paper
      elevation={0}
      sx={{
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
      }}
    >
      <Box sx={{ p: 1.5, bgcolor: 'action.hover' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <ReportIcon fontSize="small" color="primary" />
          <Typography variant="subtitle2" fontWeight={600}>
            Damage Report #{report.reportId || report._id?.substring(0, 6)}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <DateIcon fontSize="small" color="action" />
            <Typography variant="caption" color="text.secondary">
              {formatDate(report.createdAt || report.date)}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <LocationIcon fontSize="small" color="action" />
            <Typography variant="caption" color="text.secondary" noWrap>
              {report.location?.address || report.address || 'No location data'}
            </Typography>
          </Box>
        </Box>
        
        <Chip 
          label={report.status || 'Submitted'} 
          size="small"
          color={
            report.status === 'completed' ? 'success' :
            report.status === 'in_progress' ? 'info' :
            report.status === 'pending' ? 'warning' : 'default'
          }
          variant="outlined"
        />
      </Box>
      
      <Divider />
      
      <Box sx={{ p: 1.5 }}>
        <Typography variant="body2" gutterBottom>
          {report.description || report.details || 'No description provided'}
        </Typography>
        
        {report.damageType && (
          <Typography variant="caption" color="text.secondary" display="block" mt={1}>
            Damage Type: {report.damageType}
          </Typography>
        )}
        
        {report.severity && (
          <Typography variant="caption" color="text.secondary" display="block">
            Severity: {report.severity}
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

ReportMessage.propTypes = {
  report: PropTypes.object
};

export default ReportMessage;
