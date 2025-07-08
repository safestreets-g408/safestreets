import React, { useState, useEffect } from 'react';
import { useTheme, alpha } from '@mui/material/styles';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Typography,
  Box,
  Button,
  CircularProgress,
  TextField,
  List,
  Paper,
  Chip,
  InputAdornment,
  Alert,
  FormControl,
  InputLabel,
  MenuItem,
  Select
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import AssignmentIcon from '@mui/icons-material/Assignment';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import FilterListIcon from '@mui/icons-material/FilterList';
import api from '../../services/apiService';

const ReportSelectorDialog = ({ open, onClose, onSelectReport, tenantId }) => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reports, setReports] = useState([]);
  const [filteredReports, setFilteredReports] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedReport, setSelectedReport] = useState(null);
  const [statusFilter, setStatusFilter] = useState('');
  const [severityFilter, setSeverityFilter] = useState('');
  
  // Fetch damage reports
  useEffect(() => {
    const fetchReports = async () => {
      if (!open) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await api.get(`/damage/reports`, {
          params: {
            tenant: tenantId
          }
        });
        
        setReports(response.data);
        setFilteredReports(response.data);
      } catch (err) {
        console.error('Error fetching damage reports:', err);
        setError('Failed to load damage reports. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchReports();
  }, [open, tenantId]);
  
  // Filter reports based on search and filters
  useEffect(() => {
    if (!reports.length) return;
    
    let filtered = [...reports];
    
    // Apply search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(report => 
        report.reportId?.toLowerCase().includes(term) ||
        report.region?.toLowerCase().includes(term) ||
        report.location?.toLowerCase().includes(term) ||
        report.damageType?.toLowerCase().includes(term) ||
        report.description?.toLowerCase().includes(term)
      );
    }
    
    // Apply status filter
    if (statusFilter) {
      filtered = filtered.filter(report => report.status === statusFilter);
    }
    
    // Apply severity filter
    if (severityFilter) {
      filtered = filtered.filter(report => report.severity === severityFilter);
    }
    
    setFilteredReports(filtered);
  }, [reports, searchTerm, statusFilter, severityFilter]);
  
  // Handle selecting a report
  const handleSelectReport = (report) => {
    setSelectedReport(report);
  };
  
  // Handle sending the selected report
  const handleSendReport = () => {
    if (selectedReport) {
      onSelectReport(selectedReport);
    }
  };
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return theme.palette.success.main;
      case 'pending':
        return theme.palette.warning.main;
      case 'in progress':
      case 'in_progress':
        return theme.palette.info.main;
      case 'cancelled':
        return theme.palette.error.main;
      case 'on_hold':
      case 'on hold':
        return theme.palette.grey[500];
      default:
        return theme.palette.grey[500];
    }
  };
  
  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return <ErrorIcon fontSize="small" sx={{ color: theme.palette.error.main }} />;
      case 'medium':
        return <WarningIcon fontSize="small" sx={{ color: theme.palette.warning.main }} />;
      case 'low':
        return <InfoIcon fontSize="small" sx={{ color: theme.palette.info.main }} />;
      default:
        return <InfoIcon fontSize="small" sx={{ color: theme.palette.grey[500] }} />;
    }
  };
  
  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 3,
          overflow: 'hidden',
          maxHeight: '80vh'
        }
      }}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        background: theme.palette.mode === 'dark'
          ? `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(theme.palette.secondary.main || theme.palette.primary.light, 0.05)} 100%)`
          : 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
        borderBottom: `1px solid ${theme.palette.mode === 'dark'
          ? alpha(theme.palette.primary.main, 0.2)
          : 'rgba(139, 92, 246, 0.15)'}`,
        py: 2
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AssignmentIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Select Damage Report to Share
          </Typography>
        </Box>
        <IconButton onClick={onClose} edge="end" size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent sx={{ p: 3 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        ) : (
          <>
            {/* Search and filters */}
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                placeholder="Search reports by ID, location, damage type, etc."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon color="action" />
                    </InputAdornment>
                  ),
                  sx: {
                    borderRadius: 3,
                    bgcolor: 'background.paper',
                    boxShadow: theme.palette.mode === 'dark'
                      ? `0 2px 5px ${alpha(theme.palette.common.black, 0.15)}`
                      : '0 2px 5px rgba(0,0,0,0.05)',
                    '& fieldset': { 
                      borderColor: theme.palette.mode === 'dark'
                        ? alpha(theme.palette.divider, 0.7)
                        : 'rgba(0,0,0,0.1)' 
                    },
                  }
                }}
                size="small"
              />
              
              <Box sx={{ display: 'flex', mt: 2, gap: 2 }}>
                <FormControl size="small" sx={{ minWidth: 150 }}>
                  <InputLabel id="status-filter-label">Status</InputLabel>
                  <Select
                    labelId="status-filter-label"
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    label="Status"
                    startAdornment={
                      <InputAdornment position="start">
                        <FilterListIcon fontSize="small" />
                      </InputAdornment>
                    }
                  >
                    <MenuItem value="">All Statuses</MenuItem>
                    <MenuItem value="Pending">Pending</MenuItem>
                    <MenuItem value="In Progress">In Progress</MenuItem>
                    <MenuItem value="Completed">Completed</MenuItem>
                    <MenuItem value="On Hold">On Hold</MenuItem>
                    <MenuItem value="Cancelled">Cancelled</MenuItem>
                  </Select>
                </FormControl>
                
                <FormControl size="small" sx={{ minWidth: 150 }}>
                  <InputLabel id="severity-filter-label">Severity</InputLabel>
                  <Select
                    labelId="severity-filter-label"
                    value={severityFilter}
                    onChange={(e) => setSeverityFilter(e.target.value)}
                    label="Severity"
                    startAdornment={
                      <InputAdornment position="start">
                        <WarningIcon fontSize="small" />
                      </InputAdornment>
                    }
                  >
                    <MenuItem value="">All Severities</MenuItem>
                    <MenuItem value="High">High</MenuItem>
                    <MenuItem value="Medium">Medium</MenuItem>
                    <MenuItem value="Low">Low</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Box>
            
            {/* Reports list */}
            {filteredReports.length === 0 ? (
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center', 
                minHeight: 200, 
                p: 3,
                bgcolor: 'rgba(0,0,0,0.02)',
                borderRadius: 2
              }}>
                <Typography color="textSecondary" variant="body1">
                  No damage reports found matching your criteria
                </Typography>
              </Box>
            ) : (
              <List sx={{ 
                maxHeight: 400, 
                overflow: 'auto',
                mb: 2,
                '&::-webkit-scrollbar': {
                  width: '8px',
                },
                '&::-webkit-scrollbar-track': {
                  background: theme.palette.mode === 'dark'
                    ? alpha(theme.palette.action.hover, 0.1)
                    : 'rgba(0, 0, 0, 0.03)',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: theme.palette.mode === 'dark'
                    ? alpha(theme.palette.primary.main, 0.3)
                    : 'rgba(139, 92, 246, 0.3)',
                  borderRadius: '4px',
                  '&:hover': {
                    background: theme.palette.mode === 'dark'
                      ? alpha(theme.palette.primary.main, 0.5)
                      : 'rgba(139, 92, 246, 0.5)',
                  },
                },
              }}>
                {filteredReports.map((report) => (
                  <Paper 
                    key={report._id} 
                    elevation={0}
                    onClick={() => handleSelectReport(report)}
                    sx={{
                      mb: 2,
                      p: 2,
                      borderRadius: 2,
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      border: selectedReport?._id === report._id 
                        ? `2px solid ${theme.palette.primary.main}` 
                        : `1px solid ${theme.palette.mode === 'dark' ? alpha(theme.palette.divider, 0.6) : 'rgba(0,0,0,0.08)'}`,
                      boxShadow: selectedReport?._id === report._id
                        ? `0 4px 12px ${alpha(theme.palette.primary.main, 0.15)}`
                        : theme.palette.mode === 'dark' 
                          ? `0 2px 5px ${alpha(theme.palette.common.black, 0.15)}`
                          : '0 2px 5px rgba(0,0,0,0.03)',
                      '&:hover': {
                        boxShadow: theme.palette.mode === 'dark'
                          ? `0 4px 12px ${alpha(theme.palette.common.black, 0.3)}`
                          : '0 4px 12px rgba(0,0,0,0.1)',
                        transform: 'translateY(-2px)',
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        {report.reportId}
                      </Typography>
                      <Box>
                        <Chip 
                          label={report.status}
                          size="small"
                          sx={{ 
                            color: 'white',
                            bgcolor: getStatusColor(report.status),
                            fontWeight: 500
                          }}
                        />
                      </Box>
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      {getSeverityIcon(report.severity)}
                      <Typography variant="body2" color="text.secondary">
                        {report.damageType} ({report.severity} severity)
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Location:</strong> {report.location}
                      </Typography>
                    </Box>
                    
                    {report.description && (
                      <Typography 
                        variant="body2" 
                        color="text.secondary" 
                        sx={{ 
                          mt: 1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                        }}
                      >
                        {report.description}
                      </Typography>
                    )}
                    
                    {selectedReport?._id === report._id && (
                      <Box sx={{ 
                        display: 'flex', 
                        justifyContent: 'center', 
                        mt: 2, 
                        color: 'primary.main' 
                      }}>
                        <CheckCircleIcon />
                      </Box>
                    )}
                  </Paper>
                ))}
              </List>
            )}
          </>
        )}
      </DialogContent>
      
      <DialogActions sx={{ 
        p: 2, 
        borderTop: `1px solid ${theme.palette.divider}`,
        justifyContent: 'space-between',
      }}>
        <Button 
          onClick={onClose}
          variant="outlined"
          color="inherit"
          sx={{ 
            borderRadius: 2,
            px: 3,
            borderColor: theme.palette.mode === 'dark' ? theme.palette.divider : 'rgba(0,0,0,0.2)',
            color: 'text.secondary'
          }}
        >
          Cancel
        </Button>
        <Button 
          onClick={handleSendReport}
          variant="contained"
          disabled={!selectedReport}
          sx={{ 
            borderRadius: 2,
            px: 3,
            background: theme.palette.mode === 'dark'
              ? `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`
              : 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)',
            '&:hover': {
              boxShadow: '0 6px 15px rgba(139, 92, 246, 0.4)',
            },
          }}
        >
          Share in Chat
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ReportSelectorDialog;
