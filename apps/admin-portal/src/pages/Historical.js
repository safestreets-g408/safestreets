import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Paper, 
  Button, 
  Box, 
  Card, 
  CardContent, 
  CardMedia, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  CircularProgress,
  Tabs,
  Tab,
  Chip,
  Alert,
  Snackbar,
  useTheme
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { format } from 'date-fns';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import FilterAltIcon from '@mui/icons-material/FilterAlt';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ViewListIcon from '@mui/icons-material/ViewList';
import ViewModuleIcon from '@mui/icons-material/ViewModule';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import { api } from '../utils/api';
import { API_BASE_URL, API_ENDPOINTS, TOKEN_KEY } from '../config/constants';

const Historical = () => {
  const theme = useTheme();
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [region, setRegion] = useState('all');
  const [damageType, setDamageType] = useState('all');
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [view, setView] = useState('reports'); // 'reports', 'trends'
  const [viewMode, setViewMode] = useState('card'); // 'card', 'list'
  const [trendData, setTrendData] = useState([]);
  const [regionData, setRegionData] = useState([]);
  const [severityData, setSeverityData] = useState([]);
  const [stats, setStats] = useState({
    totalReports: 0,
    repaired: 0,
    inProgress: 0,
    highSeverity: 0
  });
  const [error, setError] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  // Fetch data from the API
  useEffect(() => {
    const fetchReportsData = async () => {
      try {
        setLoading(true);
        const data = await api.get(`${API_ENDPOINTS.DAMAGE_REPORTS}/reports`);
        setReports(data);
        
        // Process data for trends and analysis
        processDataForAnalytics(data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching reports data:', err);
        setError(err.message || 'Failed to fetch reports data');      
        setSnackbar({
          open: true,
          message: 'Failed to load reports',
          severity: 'error'
        });
        setLoading(false);
      }
    };
    
    fetchReportsData();
  }, []);

  // Process API data for analytics
  const processDataForAnalytics = (data) => {
    if (!Array.isArray(data) || data.length === 0) {
      console.warn('No data available for analytics processing');
      return;
    }

    // Group reports by month
    const reportsByMonth = {};
    const reportsByRegion = {};
    const reportsBySeverity = {};

    // Count reports by status and severity
    let completed = 0;
    let inProgress = 0;
    let highSeverity = 0;

    data.forEach(report => {
      // Extract month from createdAt
      const date = new Date(report.createdAt);
      const monthKey = format(date, 'MMM');
      
      // Initialize month data if it doesn't exist
      if (!reportsByMonth[monthKey]) {
        reportsByMonth[monthKey] = {
          month: monthKey,
          waterDamage: 0,
          structural: 0,
          electrical: 0,
          other: 0
        };
      }

      // Count by damage type
      const normalizedDamageType = report.damageType.toLowerCase();
      if (normalizedDamageType.includes('water')) {
        reportsByMonth[monthKey].waterDamage++;
      } else if (normalizedDamageType.includes('structural')) {
        reportsByMonth[monthKey].structural++;
      } else if (normalizedDamageType.includes('electrical')) {
        reportsByMonth[monthKey].electrical++;
      } else {
        reportsByMonth[monthKey].other++;
      }

      // Count by region
      if (!reportsByRegion[report.region]) {
        reportsByRegion[report.region] = 0;
      }
      reportsByRegion[report.region]++;

      // Initialize severity data if it doesn't exist
      if (!reportsBySeverity[monthKey]) {
        reportsBySeverity[monthKey] = {
          month: monthKey,
          high: 0,
          medium: 0,
          low: 0
        };
      }

      // Count by severity
      const normalizedSeverity = report.severity.toLowerCase();
      if (normalizedSeverity.includes('high')) {
        reportsBySeverity[monthKey].high++;
        highSeverity++;
      } else if (normalizedSeverity.includes('medium')) {
        reportsBySeverity[monthKey].medium++;
      } else {
        reportsBySeverity[monthKey].low++;
      }

      // Status counts
      if (report.status === 'Completed' || report.status === 'Repaired') {
        completed++;
      } else if (report.status === 'In Progress' || report.status === 'Assigned') {
        inProgress++;
      }
    });

    // Convert to array format for charts
    const trendDataArray = Object.values(reportsByMonth);
    setTrendData(trendDataArray);
    
    // Prepare region data
    const regionDataArray = Object.keys(reportsByRegion).map(region => ({
      name: region,
      value: reportsByRegion[region]
    }));
    setRegionData(regionDataArray);
    
    // Prepare severity data
    const severityDataArray = Object.values(reportsBySeverity);
    setSeverityData(severityDataArray);
    
    // Set statistics
    setStats({
      totalReports: data.length,
      repaired: completed,
      inProgress,
      highSeverity
    });
  };

  const handleFilter = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Build query parameters
      let queryParams = [];
      
      if (startDate) {
        queryParams.push(`startDate=${format(startDate, 'yyyy-MM-dd')}`);
      }
      
      if (endDate) {
        queryParams.push(`endDate=${format(endDate, 'yyyy-MM-dd')}`);
      }
      
      if (region && region !== 'all') {
        queryParams.push(`region=${region}`);
      }
      
      if (damageType && damageType !== 'all') {
        queryParams.push(`damageType=${damageType}`);
      }
      
      const queryString = queryParams.length > 0 ? `?${queryParams.join('&')}` : '';
      
      const filteredData = await api.get(`${API_ENDPOINTS.DAMAGE_REPORTS}/reports${queryString}`);
      setReports(filteredData);
      
      // Process filtered data for analytics if in trends view
      if (view === 'trends') {
        processDataForAnalytics(filteredData);
      }
      
      setSnackbar({
        open: true,
        message: `${filteredData.length} matching reports found`,
        severity: 'success'
      });
      
      setLoading(false);
    } catch (err) {
      console.error('Error applying filters:', err);
      setError(err.message || 'Failed to apply filters');
      setSnackbar({
        open: true,
        message: 'Filter application failed',
        severity: 'error'
      });
      setLoading(false);
    }
  };

  const handleExportData = () => {
    // Create CSV content
    const headers = ['ID', 'Date', 'Location', 'Region', 'Damage Type', 'Status', 'Severity', 'Description'];
    const csvContent = [
      headers.join(','),
      ...reports.map(report => [
        report.reportId,
        format(new Date(report.createdAt), 'yyyy-MM-dd'),
        report.location,
        report.region,
        report.damageType,
        report.status,
        report.severity,
        `"${report.description ? report.description.replace(/"/g, '""') : ''}"`
      ].join(','))
    ].join('\n');

    // Create and download the file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `damage_reports_${format(new Date(), 'yyyy-MM-dd')}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    setSnackbar({
      open: true,
      message: 'Data exported',
      severity: 'success'
    });
  };

  // Get image URL with authentication token
  const getAuthenticatedImageUrl = (reportId, type) => {
    const token = localStorage.getItem(TOKEN_KEY);
    return `${API_BASE_URL}${API_ENDPOINTS.DAMAGE_REPORTS}/report/${reportId}/image/${type}?token=${token}`;
  };
  
  // View a specific report
  const viewReport = async (reportId) => {
    try {
      setLoading(true);
      setError(null);
      
      const reportData = await api.get(`${API_ENDPOINTS.DAMAGE_REPORTS}/report/${reportId}`);
      console.log('Retrieved report data:', reportData);
      
      // Check if reportId is in the correct format
      const reportIdToUse = reportData.reportId || reportId;
      
      const beforeImageUrl = getAuthenticatedImageUrl(reportIdToUse, 'before');
      const afterImageUrl = reportData.afterImage ? 
        getAuthenticatedImageUrl(reportIdToUse, 'after') : 
        'https://via.placeholder.com/300x200?text=No+After+Image';
      
      console.log('Using image URLs:', { 
        beforeImageUrl, 
        afterImageUrl, 
        reportIdUsed: reportIdToUse
      });
      
      setSelectedReport({
        ...reportData,
        beforeImage: beforeImageUrl,
        afterImage: afterImageUrl
      });
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching report details:', err);
      setError(err.message || 'Failed to fetch report details');
      setSnackbar({
        open: true,
        message: 'Failed to load report details',
        severity: 'error'
      });
      setLoading(false);
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Using theme colors for charts
  const COLORS = [
    theme.palette.primary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main
  ];
  
  const STATUS_COLORS = {
    'Repaired': theme.palette.success.main,
    'Completed': theme.palette.success.main,
    'Pending': theme.palette.warning.main,
    'Assigned': theme.palette.info.main,
    'In Progress': theme.palette.info.main,
    'In-Progress': theme.palette.info.main
  };
  
  const SEVERITY_COLORS = {
    'High': theme.palette.error.main,
    'Medium': theme.palette.warning.main,
    'Low': theme.palette.success.main
  };

  const getStatusChip = (status) => {
    return (
      <Chip 
        label={status} 
        size="small" 
        sx={{ 
          backgroundColor: theme.palette.mode === 'dark' 
            ? `${STATUS_COLORS[status]}22` // 22 = 13% opacity in hex
            : `${STATUS_COLORS[status]}15`, // 15 = 8% opacity in hex
          color: STATUS_COLORS[status] || '#757575',
          fontWeight: 600,
          height: 22,
          fontSize: '0.7rem',
          borderRadius: 1
        }} 
      />
    );
  };

  const getSeverityChip = (severity) => {
    return (
      <Chip 
        label={severity} 
        size="small" 
        sx={{ 
          backgroundColor: theme.palette.mode === 'dark' 
            ? `${SEVERITY_COLORS[severity]}22` // 22 = 13% opacity in hex
            : `${SEVERITY_COLORS[severity]}15`, // 15 = 8% opacity in hex
          color: SEVERITY_COLORS[severity] || '#757575',
          fontWeight: 600,
          height: 22,
          fontSize: '0.7rem',
          borderRadius: 1
        }} 
      />
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 2, mb: 4 }}>
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={4000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%', fontSize: '0.875rem', borderRadius: 1 }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2.5, flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tabs 
            value={view} 
            onChange={(e, newValue) => setView(newValue)}
            indicatorColor="primary"
            textColor="primary"
            variant="standard"
            sx={{ 
              minHeight: '32px', 
              '& .MuiTab-root': { 
                py: 0, 
                minHeight: '32px', 
                textTransform: 'none', 
                fontSize: '0.85rem',
                fontWeight: 500,
                letterSpacing: '0.01em'
              },
              '& .Mui-selected': {
                fontWeight: 600
              },
              '& .MuiTabs-indicator': {
                height: 3,
                borderTopLeftRadius: 3,
                borderTopRightRadius: 3
              }
            }}
          >
            <Tab label="Reports" value="reports" />
            <Tab label="Analysis" value="trends" />
          </Tabs>
        </Box>
          
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
          {view === 'reports' && !selectedReport && (              
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.01)',
              borderRadius: 2,
              px: 1.5,
              py: 0.5,
              mr: 1.5,
              boxShadow: theme.palette.mode === 'dark' ? '0 1px 3px rgba(0,0,0,0.2)' : '0 1px 2px rgba(0,0,0,0.05)'
            }}>
              <Typography variant="caption" sx={{ mr: 1.5, color: 'text.secondary', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                View
              </Typography>
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={(e, newMode) => newMode && setViewMode(newMode)}
                size="small"
                sx={{ 
                  height: '28px',
                  '& .MuiToggleButton-root': { 
                    py: 0, 
                    px: 1,
                    border: 'none',
                    borderRadius: 1.5,
                    display: 'flex',
                    alignItems: 'center',
                    minWidth: '60px',
                    '&.Mui-selected': {
                      bgcolor: theme.palette.primary.main,
                      color: '#fff',
                      '&:hover': {
                        bgcolor: theme.palette.primary.dark
                      }
                    },
                    '&:hover': {
                      bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.04)'
                    }
                  }
                }}
              >
                <ToggleButton value="card" aria-label="card view">
                  <ViewModuleIcon sx={{ fontSize: '1rem', mr: 0.7 }} />
                  <Typography variant="caption" sx={{ fontSize: '0.75rem', fontWeight: 500 }}>Cards</Typography>
                </ToggleButton>
                <ToggleButton value="list" aria-label="list view">
                  <ViewListIcon sx={{ fontSize: '1rem', mr: 0.7 }} />
                  <Typography variant="caption" sx={{ fontSize: '0.75rem', fontWeight: 500 }}>List</Typography>
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
          )}
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.01)',
            borderRadius: 2,
            px: 1.5,
            py: 0.75,
            boxShadow: theme.palette.mode === 'dark' ? '0 1px 3px rgba(0,0,0,0.2)' : '0 1px 2px rgba(0,0,0,0.05)'
          }}>
            <FilterAltIcon sx={{ fontSize: "1rem", color: theme.palette.primary.main, mr: 1 }} />
            <Box component="form" sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <DatePicker
                  label="Start"
                  value={startDate}
                  onChange={(newValue) => setStartDate(newValue)}
                  slotProps={{ 
                    textField: { 
                      size: 'small', 
                      sx: { 
                        width: '110px',
                        '& .MuiOutlinedInput-root': {
                          borderRadius: 1.5,
                          backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#fff'
                        }
                      },
                      InputLabelProps: { 
                        shrink: true,
                        sx: { fontSize: '0.85rem' } 
                      } 
                    } 
                  }}
                />
              </LocalizationProvider>
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <DatePicker
                  label="End"
                  value={endDate}
                  onChange={(newValue) => setEndDate(newValue)}
                  slotProps={{ 
                    textField: { 
                      size: 'small', 
                      sx: { 
                        width: '110px',
                        '& .MuiOutlinedInput-root': {
                          borderRadius: 1.5,
                          backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#fff'
                        }
                      },
                      InputLabelProps: { 
                        shrink: true,
                        sx: { fontSize: '0.85rem' } 
                      } 
                    } 
                  }}
                />
              </LocalizationProvider>
              <FormControl size="small" sx={{ 
                minWidth: '100px',
                '& .MuiOutlinedInput-root': {
                  borderRadius: 1.5,
                  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#fff'
                }
              }}>
                <InputLabel sx={{ fontSize: '0.85rem' }}>Region</InputLabel>
                <Select
                  value={region}
                  label="Region"
                  onChange={(e) => setRegion(e.target.value)}
                  MenuProps={{
                    PaperProps: {
                      sx: { borderRadius: 1.5, mt: 0.5 }
                    }
                  }}
                >
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="North">North</MenuItem>
                  <MenuItem value="South">South</MenuItem>
                  <MenuItem value="East">East</MenuItem>
                  <MenuItem value="West">West</MenuItem>
                </Select>
              </FormControl>
              <FormControl size="small" sx={{ 
                minWidth: '100px',
                '& .MuiOutlinedInput-root': {
                  borderRadius: 1.5,
                  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#fff'
                }
              }}>
                <InputLabel sx={{ fontSize: '0.85rem' }}>Type</InputLabel>
                <Select
                  value={damageType}
                  label="Type"
                  onChange={(e) => setDamageType(e.target.value)}
                  MenuProps={{
                    PaperProps: {
                      sx: { borderRadius: 1.5, mt: 0.5 }
                    }
                  }}
                >
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="Water Damage">Water</MenuItem>
                  <MenuItem value="Structural">Structure</MenuItem>
                  <MenuItem value="Electrical">Electric</MenuItem>
                  <MenuItem value="Other">Other</MenuItem>
                </Select>
              </FormControl>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleFilter}
                disabled={loading}
                size="small"
                sx={{ 
                  height: '32px', 
                  minWidth: '80px', 
                  px: 2, 
                  borderRadius: 1.5,
                  fontWeight: 600,
                  boxShadow: 'none',
                  '&:hover': {
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                  }
                }}
              >
                {loading ? <CircularProgress size={14} color="inherit" /> : 'Apply'}
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                onClick={handleExportData}
                disabled={loading || reports.length === 0}
                size="small"
                sx={{ 
                  minWidth: '40px', 
                  height: '32px', 
                  p: 1,
                  borderRadius: 1.5,
                  borderWidth: 1.5,
                  '&:hover': {
                    borderWidth: 1.5
                  }
                }}
              >
                <FileDownloadIcon fontSize="small" />
              </Button>
            </Box>
          </Box>
        </Box>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress size={20} thickness={4} />
        </Box>
      )}

      {error && !loading && (
        <Alert severity="error" variant="outlined" sx={{ mb: 2, py: 0.5, fontSize: '0.875rem' }}>
          {error}
        </Alert>
      )}

      {!loading && !error && reports.length === 0 && (
        <Alert severity="info" variant="outlined" sx={{ mb: 2, py: 0.5, fontSize: '0.875rem' }}>
          No matching reports found.
        </Alert>
      )}

      {!loading && view === 'reports' && (
        <>
          {selectedReport ? (
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Button 
                  variant="text" 
                  size="small"
                  sx={{ mr: 1, p: 0, minWidth: 'auto' }}
                  onClick={() => setSelectedReport(null)}
                >
                  <ArrowBackIcon fontSize="small" />
                </Button>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  #{selectedReport.reportId} - {selectedReport.location}
                </Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ mb: 2, p: 0, bgcolor: 'transparent' }}>
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Date:</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                          {format(new Date(selectedReport.createdAt), 'yyyy-MM-dd')}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Region:</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 'medium' }}>{selectedReport.region}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Damage Type:</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 'medium' }}>{selectedReport.damageType}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Status:</Typography>
                        {getStatusChip(selectedReport.status)}
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Severity:</Typography>
                        {getSeverityChip(selectedReport.severity)}
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Assigned To:</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 'medium' }}>{selectedReport.assignedTo ? selectedReport.assignedTo.name : 'Unassigned'}</Typography>
                      </Grid>
                      {selectedReport.resolvedAt && (
                        <Grid item xs={12}>
                          <Typography variant="subtitle2" color="textSecondary">Resolved At:</Typography>
                          <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                            {format(new Date(selectedReport.resolvedAt), 'yyyy-MM-dd')}
                          </Typography>
                        </Grid>
                      )}
                    </Grid>
                  </Box>
                  <Typography variant="body2" sx={{ mt: 1, mb: 0.5, fontWeight: 500 }}>Description</Typography>
                  <Paper variant="outlined" sx={{ p: 1, borderRadius: 0.5 }}>
                    <Typography variant="caption">{selectedReport.description || 'No description provided.'}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ mb: 0.5, fontWeight: 500 }}>Before/After</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card sx={{ boxShadow: 0, borderRadius: 0.5, overflow: 'hidden', border: '1px solid', borderColor: theme.palette.divider }}>
                        <Box sx={{ position: 'relative', height: 180, bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#f5f5f5' }}>
                          <CardMedia
                            component="img"
                            height="180"
                            image={selectedReport.beforeImage}
                            alt="Before repair"
                            sx={{ objectFit: 'cover' }}
                            onError={(e) => {
                              console.log('Before image load error');
                              e.target.onerror = null;
                              e.target.src = `https://via.placeholder.com/400x180/f0f0f0/999999?text=No+Before+Image`;
                            }}
                          />
                        </Box>
                        <CardContent sx={{ py: 0.5, px: 1, bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#fafafa' }}>
                          <Typography variant="caption" sx={{ fontSize: '0.7rem', textAlign: 'center', display: 'block' }}>Before</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card sx={{ boxShadow: 0, borderRadius: 0.5, overflow: 'hidden', border: '1px solid', borderColor: theme.palette.divider }}>
                        <Box sx={{ position: 'relative', height: 180, bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#f5f5f5' }}>
                          <CardMedia
                            component="img"
                            height="180"
                            image={selectedReport.afterImage}
                            alt="After repair"
                            sx={{ objectFit: 'cover' }}
                            onError={(e) => {
                              console.log('After image load error');
                              e.target.onerror = null;
                              e.target.src = `https://via.placeholder.com/400x180/f0f0f0/999999?text=No+After+Image`;
                            }}
                          />
                        </Box>
                        <CardContent sx={{ py: 0.5, px: 1, bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#fafafa' }}>
                          <Typography variant="caption" sx={{ fontSize: '0.7rem', textAlign: 'center', display: 'block' }}>After</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </Box>
          ) : (
            <Box>
              {viewMode === 'card' ? (
                <Grid container spacing={2}>
                  {reports.map((report) => (
                    <Grid item xs={12} sm={6} md={4} key={report._id}>
                      <Card 
                        sx={{ 
                          cursor: 'pointer', 
                          transition: 'all 0.2s',
                          borderRadius: 2,
                          overflow: 'hidden',
                          boxShadow: theme.palette.mode === 'dark' ? '0 3px 10px rgba(0,0,0,0.4)' : '0 2px 8px rgba(0,0,0,0.08)',
                          border: 'none',
                          height: '100%',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: theme.palette.mode === 'dark' ? '0 5px 15px rgba(0,0,0,0.5)' : '0 4px 12px rgba(0,0,0,0.15)'
                          }
                        }} 
                        onClick={() => viewReport(report.reportId)}
                      >
                        <Box sx={{ height: 160, position: 'relative', overflow: 'hidden' }}>
                          <img 
                            src={getAuthenticatedImageUrl(report.reportId, 'before')} 
                            alt={report.location} 
                            style={{ 
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover',
                              transition: 'transform 0.3s'
                            }}
                            onError={(e) => {
                              console.log('Image load error for report:', report.reportId);
                              e.target.onerror = null;
                              e.target.src = `https://via.placeholder.com/300x160/f0f0f0/999999?text=No+Image`;
                            }}
                          />
                          <Box sx={{ 
                            position: 'absolute', 
                            top: 10, 
                            right: 10, 
                            bgcolor: 'rgba(0,0,0,0.6)', 
                            color: '#fff',
                            borderRadius: 5,
                            px: 1,
                            py: 0.2,
                            fontSize: '0.65rem'
                          }}>
                            #{report.reportId}
                          </Box>
                        </Box>
                        <CardContent sx={{ py: 1.5, px: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body1" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                              {report.location}
                            </Typography>
                            <Typography color="textSecondary" variant="caption" sx={{ fontSize: '0.7rem', fontWeight: 500 }}>
                              {format(new Date(report.createdAt), 'MMM dd, yyyy')}
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', gap: 0.7, flexWrap: 'wrap', mt: 1.5 }}>
                            <Chip 
                              label={report.damageType} 
                              size="small" 
                              sx={{ 
                                height: 22,
                                fontSize: '0.7rem',
                                fontWeight: 500,
                                bgcolor: theme.palette.primary.main,
                                color: '#fff',
                                borderRadius: 1
                              }} 
                            />
                            {getStatusChip(report.status)}
                            {getSeverityChip(report.severity)}
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Box sx={{ 
                  borderRadius: 2,
                  overflow: 'hidden',
                  boxShadow: theme.palette.mode === 'dark' ? '0 3px 10px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.05)'
                }}>
                  {reports.map((report, index) => (
                    <Box 
                      key={report._id}
                      sx={{ 
                        display: 'flex', 
                        p: 1.5,
                        cursor: 'pointer',
                        borderBottom: index < reports.length - 1 ? '1px solid' : 'none',
                        borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)',
                        bgcolor: theme.palette.mode === 'dark' ? 'rgba(0, 0, 0, 0.2)' : '#fff',
                        transition: 'all 0.2s',
                        '&:hover': {
                          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                          transform: 'translateX(2px)'
                        }
                      }}
                      onClick={() => viewReport(report.reportId)}
                    >
                      <Box sx={{ 
                        width: 80, 
                        height: 80, 
                        mr: 2, 
                        overflow: 'hidden', 
                        flexShrink: 0, 
                        borderRadius: 1.5, 
                        boxShadow: '0 2px 5px rgba(0,0,0,0.15)'
                      }}>
                        <img 
                          src={getAuthenticatedImageUrl(report.reportId, 'before')} 
                          alt={report.location}
                          style={{ 
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover' 
                          }}
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = `https://via.placeholder.com/80x80/f0f0f0/999999?text=No+Image`;
                          }}
                        />
                      </Box>
                      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body1" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                            {report.location}
                          </Typography>
                          <Box sx={{
                            display: 'flex',
                            alignItems: 'center',
                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
                            borderRadius: 4,
                            px: 1,
                            py: 0.3
                          }}>
                            <Typography color="textSecondary" variant="caption" sx={{ fontSize: '0.7rem', fontWeight: 500 }}>
                              {format(new Date(report.createdAt), 'MMM dd, yyyy')}
                            </Typography>
                          </Box>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                          <Typography variant="caption" sx={{ 
                            color: theme.palette.primary.main, 
                            mr: 1.5, 
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(25, 118, 210, 0.15)' : 'rgba(25, 118, 210, 0.1)',
                            px: 0.8,
                            py: 0.2,
                            borderRadius: 1
                          }}>
                            #{report.reportId}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 0.8 }}>
                            <Chip 
                              label={report.damageType} 
                              size="small" 
                              sx={{ 
                                height: 22,
                                fontSize: '0.7rem',
                                fontWeight: 500,
                                bgcolor: theme.palette.primary.main,
                                color: '#fff',
                                borderRadius: 1
                              }} 
                            />
                            {getStatusChip(report.status)}
                            {getSeverityChip(report.severity)}
                          </Box>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
                        <Box sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 30,
                          height: 30,
                          borderRadius: '50%',
                          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
                          transition: 'all 0.2s',
                          '&:hover': {
                            bgcolor: theme.palette.primary.main,
                            color: '#fff'
                          }
                        }}>
                          <VisibilityIcon sx={{ fontSize: '1.1rem' }} />
                        </Box>
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
            </Box>
          )}
        </>
      )}

      {!loading && view === 'trends' && (
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 500, mb: 1, color: theme.palette.text.secondary, fontSize: '0.8rem' }}>
            Damage Analysis
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <Box sx={{ 
                mb: 2
              }}>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: 'block', color: theme.palette.text.secondary }}>
                  Monthly Damage Types
                </Typography>
                <Box sx={{ height: 300, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={trendData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : '#f0f0f0'} />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip contentStyle={{ borderRadius: 8 }} />
                      <Legend />
                      <Bar dataKey="waterDamage" name="Water Damage" stackId="a" fill="#8884d8" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="structural" name="Structural" stackId="a" fill="#82ca9d" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="electrical" name="Electrical" stackId="a" fill="#ffc658" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="other" name="Other" stackId="a" fill="#ff8042" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: 'block', color: theme.palette.text.secondary }}>
                  Severity Trends
                </Typography>
                <Box sx={{ height: 300, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={severityData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : '#f0f0f0'} />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip contentStyle={{ borderRadius: 8 }} />
                      <Legend />
                      <Line type="monotone" dataKey="high" name="High Severity" stroke="#f44336" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="medium" name="Medium Severity" stroke="#ff9800" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="low" name="Low Severity" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: 'block', color: theme.palette.text.secondary }}>
                  Regional Distribution
                </Typography>
                <Box sx={{ height: 300, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={regionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {regionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ borderRadius: 8 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
              
              <Box>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: 'block', color: theme.palette.text.secondary }}>
                  Summary
                </Typography>
                <Paper variant="outlined" sx={{ 
                  p: 1.5, 
                  borderRadius: 0.5,
                  borderColor: theme.palette.divider,
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.02)' : 'rgba(0, 0, 0, 0.01)'
                }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Box sx={{ 
                        p: 1.5, 
                        bgcolor: theme.palette.mode === 'dark' ? 'rgba(33, 150, 243, 0.08)' : '#f5f9ff',
                        borderRadius: 0.5, 
                        textAlign: 'center',
                        border: '1px solid',
                        borderColor: theme.palette.mode === 'dark' ? 'rgba(33, 150, 243, 0.2)' : 'rgba(33, 150, 243, 0.1)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Typography variant="body2" sx={{ fontSize: '0.7rem', color: theme.palette.text.secondary, mb: 0.5 }}>Total Reports</Typography>
                        <Typography variant="h5" sx={{ fontWeight: 500, color: theme.palette.primary.main, lineHeight: 1 }}>{stats.totalReports}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ 
                        p: 1.5, 
                        bgcolor: theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.08)' : '#f5fff7',
                        borderRadius: 0.5, 
                        textAlign: 'center',
                        border: '1px solid',
                        borderColor: theme.palette.mode === 'dark' ? 'rgba(76, 175, 80, 0.2)' : 'rgba(76, 175, 80, 0.1)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Typography variant="body2" sx={{ fontSize: '0.7rem', color: theme.palette.text.secondary, mb: 0.5 }}>Repaired</Typography>
                        <Typography variant="h5" sx={{ fontWeight: 500, color: theme.palette.success.main, lineHeight: 1 }}>{stats.repaired}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ 
                        p: 1.5, 
                        bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 152, 0, 0.08)' : '#fffbf2',
                        borderRadius: 0.5, 
                        textAlign: 'center',
                        border: '1px solid',
                        borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 152, 0, 0.2)' : 'rgba(255, 152, 0, 0.1)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Typography variant="body2" sx={{ fontSize: '0.7rem', color: theme.palette.text.secondary, mb: 0.5 }}>In Progress</Typography>
                        <Typography variant="h5" sx={{ fontWeight: 500, color: theme.palette.warning.main, lineHeight: 1 }}>{stats.inProgress}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ 
                        p: 1.5, 
                        bgcolor: theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.08)' : '#fff8f7',
                        borderRadius: 0.5, 
                        textAlign: 'center',
                        border: '1px solid',
                        borderColor: theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.2)' : 'rgba(244, 67, 54, 0.1)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Typography variant="body2" sx={{ fontSize: '0.7rem', color: theme.palette.text.secondary, mb: 0.5 }}>Critical</Typography>
                        <Typography variant="h5" sx={{ fontWeight: 500, color: theme.palette.error.main, lineHeight: 1 }}>{stats.highSeverity}</Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Box>
            </Grid>
          </Grid>
        </Box>
      )}
    </Container>
  );
};

export default Historical;
