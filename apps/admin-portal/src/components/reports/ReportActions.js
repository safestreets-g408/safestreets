import React from 'react';
import { Box, Tooltip, IconButton } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import FileDownloadIcon from '@mui/icons-material/FileDownload';

const ReportActions = ({ report, onView, onEdit, onDelete, onDownload, colors }) => {
  // Prevent event bubbling and ensure only the clicked action is triggered
  const handleAction = (action, event) => {
    event.stopPropagation();
    action(report);
  };
  
  // Common icon button styles for a minimal look
  const iconButtonStyle = {
    padding: '4px',
    '&:hover': { 
      backgroundColor: 'rgba(0, 0, 0, 0.04)',
    }
  };
  
  return (
    <Box sx={{ display: 'flex', gap: 0.5 }}>
      <Tooltip title="View Details" arrow placement="top">
        <IconButton 
          size="small"
          onClick={(e) => handleAction(onView, e)}
          sx={{ 
            ...iconButtonStyle,
            color: 'primary.main',
          }}
        >
          <VisibilityIcon sx={{ fontSize: '1rem' }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Edit Report" arrow placement="top">
        <IconButton 
          size="small"
          onClick={(e) => handleAction(onEdit, e)}
          sx={{ 
            ...iconButtonStyle,
            color: 'text.secondary',
          }}
        >
          <EditIcon sx={{ fontSize: '1rem' }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Delete Report" arrow placement="top">
        <IconButton 
          size="small"
          onClick={(e) => handleAction(onDelete, e)}
          sx={{ 
            ...iconButtonStyle,
            color: 'error.main',
          }}
        >
          <DeleteIcon sx={{ fontSize: '1rem' }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Download Report" arrow placement="top">
        <IconButton 
          size="small"
          onClick={(e) => handleAction(onDownload, e)}
          sx={{ 
            ...iconButtonStyle,
            color: 'text.secondary',
          }}
        >
          <FileDownloadIcon sx={{ fontSize: '1rem' }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
};

export default ReportActions;
