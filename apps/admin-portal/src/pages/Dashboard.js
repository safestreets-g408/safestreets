import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  LinearProgress, IconButton, Avatar, List, ListItem,
  ListItemText, ListItemAvatar, Divider, Chip, Button,
  useTheme, alpha, CircularProgress, Dialog, DialogTitle,
  DialogContent, DialogActions, TextField, MenuItem, Select,
  FormControl, InputLabel, FormHelperText, Snackbar, Alert
} from '@mui/material';
import ReportIcon from '@mui/icons-material/Report';
import BuildIcon from '@mui/icons-material/Build';
import CheckCircleIcon from '@mui/icons-material/CheckCircle'; 
import WarningIcon from '@mui/icons-material/Warning';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import RefreshIcon from '@mui/icons-material/Refresh';
import AssignmentIcon from '@mui/icons-material/Assignment';
import PersonIcon from '@mui/icons-material/Person';
import { styled } from '@mui/material/styles';
import { api } from '../utils/api';

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.3s, box-shadow 0.3s',
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: '0 10px 20px rgba(0,0,0,0.05)',
  overflow: 'hidden',
  border: '1px solid',
  borderColor: alpha(theme.palette.divider, 0.1),
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 15px 30px rgba(0,0,0,0.1)'
  }
}));

const GlassCard = styled(Paper)(({ theme }) => ({
  backdropFilter: 'blur(10px)',
  backgroundColor: alpha(theme.palette.background.paper, 0.8),
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: '0 10px 30px rgba(0,0,0,0.08)',
  border: '1px solid',
  borderColor: alpha(theme.palette.divider, 0.1),
}));

const ProgressBar = styled(LinearProgress)(({ theme }) => ({
  height: 8,
  borderRadius: 4,
  backgroundColor: alpha(theme.palette.divider, 0.1),
  '& .MuiLinearProgress-bar': {
    borderRadius: 4,
  }
}));

const Dashboard = () => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState({
    totalReports: 0,
    pendingRepairs: 0,
    completedRepairs: 0,
    criticalDamages: 0,
    recentReports: []
  });
  const [aiReports, setAiReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [repairDialog, setRepairDialog] = useState(false);
  const [assignDialog, setAssignDialog] = useState(false);
  const [generatedReportIds, setGeneratedReportIds] = useState([]);
  const [reportFormData, setReportFormData] = useState({
    region: '',
    location: '',
    description: '',
    action: '',
    assignTo: '' // Add field for worker assignment
  });
  const [fieldWorkers, setFieldWorkers] = useState([]);
  const [selectedWorker, setSelectedWorker] = useState('');
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  const [aiReportsLoading, setAiReportsLoading] = useState(true);

  const getSeverityColor = (severity) => {
    switch(severity.toLowerCase()) {
      case 'critical':
        return theme.palette.error.main;
      case 'high':
        return theme.palette.warning.main;
      case 'medium':
        return theme.palette.info.main;
      case 'low':
        return theme.palette.success.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get all reports from the last 30 days
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

      const response = await api.get('/damage/reports', {
        params: {
          startDate: thirtyDaysAgo.toISOString(),
          endDate: new Date().toISOString()
        }
      });

      // Process the response data
      const totalReports = response.length;
      const pendingRepairs = response.filter(report => report.status === 'Pending').length;
      const completedRepairs = response.filter(report => report.status === 'Completed').length;
      const criticalDamages = response.filter(report => report.severity === 'Critical').length;

      // Get the 3 most recent reports
      const recentReports = response
        .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
        .slice(0, 3)
        .map(report => ({
          _id: report._id,
          id: report._id,
          reportId: report.reportId,
          title: report.description || 'No description',
          severity: report.severity || 'Low',
          location: report.region || 'Unknown Location',
          region: report.region,
          timestamp: formatTimestamp(report.createdAt),
          status: report.status || 'Pending',
          assignedTo: report.assignedTo
        }));

      setDashboardData({
        totalReports,
        pendingRepairs,
        completedRepairs,
        criticalDamages,
        recentReports
      });
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchAiReports = async () => {
    try {
      setAiReportsLoading(true);
      const response = await api.get('/images/reports');
      if (response && response.reports) {
        setAiReports(response.reports);
        
        // Get the list of AI reports that already have damage reports generated
        try {
          const generatedReports = await api.get('/damage/reports/generated-from-ai');
          if (generatedReports && Array.isArray(generatedReports)) {
            // Map the aiReportId values to an array of report IDs
            const generatedIds = generatedReports.map(report => report.aiReportId);
            setGeneratedReportIds(generatedIds);
          }
        } catch (innerErr) {
          console.error('Error fetching generated report IDs:', innerErr);
        }
      }
    } catch (err) {
      console.error('Error fetching AI reports:', err);
    } finally {
      setAiReportsLoading(false);
    }
  };
  
  const fetchFieldWorkers = async () => {
    try {
      const workers = await api.get('/field/workers');
      console.log('Fetched field workers:', workers);
      if (Array.isArray(workers)) {
        // Transform workers to ensure consistent format
        const transformedWorkers = workers.map(worker => ({
          id: worker._id,
          workerId: worker.workerId,
          name: worker.name,
          specialization: worker.specialization || '',
          region: worker.region || '',
          status: worker.status || 'Available'
        }));
        setFieldWorkers(transformedWorkers);
      }
    } catch (err) {
      console.error('Error fetching field workers:', err);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    fetchAiReports();
    fetchFieldWorkers();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const formatTimestamp = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now - date) / (1000 * 60 * 60));
    
    if (diffInHours < 24) {
      return `${diffInHours} hours ago`;
    } else {
      const diffInDays = Math.floor(diffInHours / 24);
      return `${diffInDays} days ago`;
    }
  };

  // Add a refresh handler
  const handleRefresh = () => {
    fetchDashboardData();
    fetchAiReports();
  };

  const handleOpenReportDialog = (report) => {
    setSelectedReport(report);
    setReportFormData({
      region: '',
      location: '',
      description: `Damage detected: ${report.damageType} - Severity: ${report.severity}`,
      action: 'Inspection Required',
      assignTo: '' // Reset worker assignment
    });
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setRepairDialog(false);
    setAssignDialog(false);
    setSelectedReport(null);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setReportFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const createDamageReport = async () => {
    try {
      if (!selectedReport) return;
      
      const payload = {
        region: reportFormData.region,
        location: reportFormData.location,
        description: reportFormData.description,
        action: reportFormData.action,
        damageType: selectedReport.damageType,
        severity: selectedReport.severity,
        priority: selectedReport.priority,
        reporter: 'admin',
        aiReportId: selectedReport.id
      };
      
      let response;
      
      // If a worker is selected, use the combined endpoint
      if (reportFormData.assignTo) {
        payload.workerId = reportFormData.assignTo;
        response = await api.post('/damage/reports/create-and-assign', payload);
      } else {
        response = await api.post('/damage/reports/create-from-ai', payload);
      }
      
      if (response && response.success) {
        setNotification({
          open: true,
          message: response.message || 'Damage report created successfully!',
          severity: 'success'
        });
        
        // Add the report ID to the list of generated reports
        setGeneratedReportIds(prevIds => [...prevIds, selectedReport.id]);
        
        setOpenDialog(false);
        fetchDashboardData(); // Refresh dashboard data
      }
    } catch (err) {
      console.error('Error creating damage report:', err);
      setNotification({
        open: true,
        message: 'Failed to create damage report: ' + err.message,
        severity: 'error'
      });
    }
  };

  const handleOpenRepairDialog = (report) => {
    setSelectedReport(report);
    setRepairDialog(true);
  };

  const handleOpenAssignDialog = (report) => {
    console.log('Opening assign dialog with report:', report);
    setSelectedReport(report);
    setAssignDialog(true);
    setSelectedWorker('');
  };

  const assignRepair = async () => {
    if (!selectedReport || !selectedWorker) {
      console.error('Cannot assign: Missing report or worker', { 
        report: selectedReport, 
        worker: selectedWorker 
      });
      
      setNotification({
        open: true,
        message: 'Cannot assign: Missing report or worker information',
        severity: 'error'
      });
      return;
    }
    
    // Ensure we have a valid report ID
    const reportId = selectedReport._id || selectedReport.id;
    if (!reportId) {
      console.error('Missing report ID in selected report:', selectedReport);
      setNotification({
        open: true,
        message: 'Error: Report ID is missing',
        severity: 'error'
      });
      return;
    }
    
    console.log('Assigning repair', { reportId, workerId: selectedWorker });
    
    try {
      const response = await api.patch(`/damage/reports/${reportId}/assign`, {
        workerId: selectedWorker
      });
      
      if (response && response.success) {
        setNotification({
          open: true,
          message: 'Repair task assigned successfully!',
          severity: 'success'
        });
        setAssignDialog(false);
        fetchDashboardData();
      }
    } catch (err) {
      console.error('Error assigning repair task:', err);
      setNotification({
        open: true,
        message: 'Failed to assign repair task: ' + (err.message || 'Unknown error'),
        severity: 'error'
      });
    }
  };

  const unassignRepair = async () => {
    if (!selectedReport) {
      console.error('Cannot unassign: Missing report', { report: selectedReport });
      return;
    }
    
    const reportId = selectedReport._id || selectedReport.id;
    console.log('Unassigning repair', { reportId });
    
    try {
      const response = await api.patch(`/damage/reports/${reportId}/unassign`);
      
      if (response && response.success) {
        setNotification({
          open: true,
          message: 'Repair task unassigned successfully!',
          severity: 'success'
        });
        setRepairDialog(false);
        fetchDashboardData();
      }
    } catch (err) {
      console.error('Error unassigning repair task:', err);
      setNotification({
        open: true,
        message: 'Failed to unassign repair task: ' + (err.message || 'Unknown error'),
        severity: 'error'
      });
    }
  };

  const handleCloseNotification = () => {
    setNotification(prev => ({
      ...prev,
      open: false
    }));
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="error" variant="h6">
          Error loading dashboard: {error}
        </Typography>
        <Button onClick={handleRefresh} sx={{ mt: 2 }}>
          Try Again
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, md: 4 }, bgcolor: 'background.default', borderRadius: 3 }}>
      <Box 
        display="flex" 
        flexDirection={{ xs: 'column', sm: 'row' }} 
        justifyContent="space-between" 
        alignItems={{ xs: 'flex-start', sm: 'center' }} 
        mb={4}
        gap={2}
      >
        <Box>
          <Typography variant="h4" fontWeight="800" color="text.primary" sx={{ letterSpacing: '-0.5px' }}>
            Dashboard Overview
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5 }}>
            Welcome back! Here's what's happening today.
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button 
            variant="outlined" 
            startIcon={<RefreshIcon />}
            sx={{ 
              borderRadius: 2,
              textTransform: 'none',
              fontWeight: 600
            }}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
          <Button 
            variant="contained" 
            disableElevation
            sx={{ 
              borderRadius: 2,
              textTransform: 'none',
              fontWeight: 600,
              boxShadow: `0 4px 14px ${alpha(theme.palette.primary.main, 0.4)}`
            }}
          >
            New Report
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {[
          { title: 'Total Reports', value: dashboardData.totalReports, icon: <ReportIcon />, color: theme.palette.primary.main, progress: 70, trend: 'up', change: '+12%' },
          { title: 'Pending Repairs', value: dashboardData.pendingRepairs, icon: <BuildIcon />, color: theme.palette.warning.main, progress: 45, trend: 'down', change: '-8%' },
          { title: 'Completed Repairs', value: dashboardData.completedRepairs, icon: <CheckCircleIcon />, color: theme.palette.success.main, progress: 85, trend: 'up', change: '+23%' },
          { title: 'Critical Damages', value: dashboardData.criticalDamages, icon: <WarningIcon />, color: theme.palette.error.main, progress: 25, trend: 'down', change: '-15%' }
        ].map((item, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <StyledCard>
              <CardContent sx={{ p: 3 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Avatar 
                    sx={{ 
                      bgcolor: alpha(item.color, 0.15), 
                      color: item.color,
                      width: 48,
                      height: 48
                    }}
                  >
                    {item.icon}
                  </Avatar>
                  <Box display="flex" alignItems="center">
                    <Chip 
                      icon={item.trend === 'up' ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
                      label={item.change}
                      size="small"
                      color={item.trend === 'up' ? 'success' : 'error'}
                      sx={{ 
                        borderRadius: 1, 
                        fontWeight: 'bold',
                        height: 24
                      }}
                    />
                    <IconButton size="small" sx={{ ml: 1 }}>
                      <MoreHorizIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </Box>
                <Typography variant="h3" sx={{ mt: 3, mb: 0.5 }} color="text.primary" fontWeight="700">
                  {item.value}
                </Typography>
                <Typography color="text.secondary" fontWeight="500" variant="body2">
                  {item.title}
                </Typography>
                <ProgressBar 
                  variant="determinate" 
                  value={item.progress} 
                  sx={{ mt: 2 }} 
                  color={
                    item.title === 'Total Reports' ? 'primary' :
                    item.title === 'Pending Repairs' ? 'warning' :
                    item.title === 'Completed Repairs' ? 'success' : 'error'
                  }
                />
              </CardContent>
            </StyledCard>
          </Grid>
        ))}

        {/* AI Analysis Reports Section */}
        <Grid item xs={12}>
          <GlassCard sx={{ p: { xs: 2, md: 3 } }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
              <Typography variant="h6" fontWeight="700" color="text.primary">
                Recent AI Analysis Reports
              </Typography>
              <Button 
                endIcon={<RefreshIcon />}
                sx={{ 
                  textTransform: 'none',
                  fontWeight: 600,
                  borderRadius: 2
                }}
                onClick={fetchAiReports}
              >
                Refresh
              </Button>
            </Box>
            
            {aiReportsLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress size={30} />
              </Box>
            ) : aiReports.length === 0 ? (
              <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography color="text.secondary">No AI analysis reports found</Typography>
              </Box>
            ) : (
              <List sx={{ px: { xs: 0, md: 1 } }}>
                {aiReports.slice(0, 5).map((report, index) => (
                  <React.Fragment key={report.id}>
                    <ListItem 
                      sx={{ 
                        py: 2,
                        px: { xs: 1, md: 2 },
                        borderRadius: 2,
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          bgcolor: alpha(theme.palette.action.hover, 0.5),
                          transform: 'translateX(4px)'
                        }
                      }}
                    >
                      <ListItemAvatar>
                        <Avatar 
                          sx={{ 
                            bgcolor: alpha(getSeverityColor(report.severity), 0.15),
                            color: getSeverityColor(report.severity),
                            fontWeight: 'bold'
                          }}
                        >
                          {report.damageType[0]}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={
                          <Box display="flex" justifyContent="space-between" alignItems="center">
                            <Box display="flex" alignItems="center" gap={1}>
                              <Typography variant="subtitle1" fontWeight="600" color="text.primary">
                                {report.damageType}
                              </Typography>
                              <Chip 
                                label={report.severity} 
                                size="small" 
                                color={report.severity === 'HIGH' ? 'error' : report.severity === 'MEDIUM' ? 'warning' : 'success'}
                                sx={{ 
                                  height: 20, 
                                  fontSize: '0.65rem',
                                  fontWeight: 600,
                                  borderRadius: 1
                                }} 
                              />
                            </Box>
                            <Box display="flex" gap={1}>
                              <Button 
                                size="small" 
                                variant={generatedReportIds.includes(report.id) ? "contained" : "outlined"}
                                color={generatedReportIds.includes(report.id) ? "success" : "primary"} 
                                startIcon={generatedReportIds.includes(report.id) ? <CheckCircleIcon /> : <AssignmentIcon />}
                                onClick={() => handleOpenReportDialog(report)
                                }
                                sx={{ borderRadius: 1 }}
                              >
                                {generatedReportIds.includes(report.id) ? "Report Generated" : "Generate Report"}
                              </Button>
                            </Box>
                          </Box>
                        }
                        secondary={
                          <Box display="flex" justifyContent="space-between" mt={1}>
                            <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box component="span" sx={{ 
                                width: 8, 
                                height: 8, 
                                borderRadius: '50%', 
                                bgcolor: getSeverityColor(report.severity),
                                display: 'inline-block',
                                mr: 1
                              }}/>
                              Priority: {report.priority}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" fontWeight="500">
                              {formatTimestamp(report.createdAt)}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < aiReports.slice(0, 5).length - 1 && (
                      <Divider variant="inset" component="li" sx={{ opacity: 0.5 }} />
                    )}
                  </React.Fragment>
                ))}
              </List>
            )}
          </GlassCard>
        </Grid>

        {/* Recent Damage Reports Section */}
        <Grid item xs={12}>
          <GlassCard sx={{ p: { xs: 2, md: 3 } }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
              <Typography variant="h6" fontWeight="700" color="text.primary">
                Recent Damage Reports
              </Typography>
              <Button 
                endIcon={<MoreHorizIcon />}
                sx={{ 
                  textTransform: 'none',
                  fontWeight: 600,
                  borderRadius: 2
                }}
              >
                View All
              </Button>
            </Box>
            <List sx={{ px: { xs: 0, md: 1 } }}>
              {dashboardData.recentReports.map((report, index) => (
                <React.Fragment key={report.id}>
                  <ListItem 
                    sx={{ 
                      py: 2,
                      px: { xs: 1, md: 2 },
                      borderRadius: 2,
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        bgcolor: alpha(theme.palette.action.hover, 0.5),
                        transform: 'translateX(4px)'
                      }
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar 
                        sx={{ 
                          bgcolor: alpha(getSeverityColor(report.severity), 0.15),
                          color: getSeverityColor(report.severity),
                          fontWeight: 'bold'
                        }}
                      >
                        {report.severity[0]}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="subtitle1" fontWeight="600" color="text.primary">
                              {report.title}
                            </Typography>
                            <Chip 
                              label={report.id} 
                              size="small" 
                              variant="outlined"
                              sx={{ 
                                height: 20, 
                                fontSize: '0.65rem',
                                fontWeight: 600,
                                borderRadius: 1
                              }} 
                            />
                          </Box>
                          <Box display="flex" gap={1}>
                            <Button 
                              size="small" 
                              variant="outlined" 
                              startIcon={<BuildIcon />} 
                              onClick={() => handleOpenRepairDialog(report)}
                              sx={{ borderRadius: 1 }}
                            >
                              Manage Repair
                            </Button>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Chip 
                                icon={<PersonIcon fontSize="small" />}
                                label={report.assignedTo !== 'Unassigned' ? report.assignedTo : 'Not Assigned'}
                                color={report.assignedTo !== 'Unassigned' ? "success" : "default"}
                                variant={report.assignedTo !== 'Unassigned' ? "filled" : "outlined"}
                                size="small"
                                sx={{ mr: 1, height: 28 }}
                              />
                              <Button 
                                size="small" 
                                variant="outlined" 
                                startIcon={<PersonIcon />} 
                                onClick={() => handleOpenAssignDialog(report)}
                                sx={{ borderRadius: 1 }}
                              >
                                {report.assignedTo !== 'Unassigned' ? 'Reassign' : 'Assign'}
                              </Button>
                            </Box>
                          </Box>
                        </Box>
                      }
                      secondary={
                        <Box display="flex" justifyContent="space-between" mt={1}>
                          <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box component="span" sx={{ 
                              width: 8, 
                              height: 8, 
                              borderRadius: '50%', 
                              bgcolor: getSeverityColor(report.severity),
                              display: 'inline-block',
                              mr: 1
                            }}/>
                            {report.location}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" fontWeight="500">
                            {report.timestamp}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                  {index < dashboardData.recentReports.length - 1 && (
                    <Divider variant="inset" component="li" sx={{ opacity: 0.5 }} />
                  )}
                </React.Fragment>
              ))}
            </List>
          </GlassCard>
        </Grid>
      </Grid>

      {/* Dialog for generating damage report from AI analysis */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Generate Damage Report</DialogTitle>
        <DialogContent>
          {selectedReport && (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle1">
                    AI Analysis Results:
                  </Typography>
                  <Typography variant="body2">
                    Damage Type: {selectedReport.damageType}
                  </Typography>
                  <Typography variant="body2">
                    Severity: {selectedReport.severity}
                  </Typography>
                  <Typography variant="body2" mb={2}>
                    Priority: {selectedReport.priority}
                  </Typography>
                  
                  {selectedReport.annotatedImage && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" mb={1}>Annotated Image:</Typography>
                      <img 
                        src={selectedReport.annotatedImage} 
                        alt="Damage" 
                        style={{ 
                          width: '100%', 
                          maxHeight: '200px',
                          objectFit: 'contain',
                          borderRadius: 8
                        }}
                      />
                    </Box>
                  )}
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    name="region"
                    label="Region"
                    fullWidth
                    value={reportFormData.region}
                    onChange={handleInputChange}
                    margin="dense"
                    required
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    name="location"
                    label="Exact Location"
                    fullWidth
                    value={reportFormData.location}
                    onChange={handleInputChange}
                    margin="dense"
                    required
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    name="description"
                    label="Description"
                    fullWidth
                    multiline
                    rows={3}
                    value={reportFormData.description}
                    onChange={handleInputChange}
                    margin="dense"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    name="action"
                    label="Required Action"
                    fullWidth
                    value={reportFormData.action}
                    onChange={handleInputChange}
                    margin="dense"
                    required
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <FormControl fullWidth margin="dense">
                    <InputLabel id="assign-worker-label">Assign to Field Worker (Optional)</InputLabel>
                    <Select
                      labelId="assign-worker-label"
                      name="assignTo"
                      value={reportFormData.assignTo}
                      label="Assign to Field Worker (Optional)"
                      onChange={handleInputChange}
                    >
                      <MenuItem value="">
                        <em>Unassigned</em>
                      </MenuItem>
                      {fieldWorkers.map(worker => (
                        <MenuItem key={worker.id} value={worker.id}>
                          {worker.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={createDamageReport} variant="contained" color="primary">
            Create Report
          </Button>
        </DialogActions>
      </Dialog>

      {/* Dialog for managing repair tasks */}
      <Dialog open={repairDialog} onClose={handleCloseDialog} maxWidth="xs" fullWidth>
        <DialogTitle>Manage Repair Task</DialogTitle>
        <DialogContent>
          {selectedReport && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                Report ID: {selectedReport.id}
              </Typography>
              <Typography variant="body2" mb={1}>
                Currently {selectedReport.assignedTo && selectedReport.assignedTo !== 'Unassigned' ? 
                  `Assigned to: ${selectedReport.assignedTo}` : 'Unassigned'}
              </Typography>
              <Typography variant="body2" mb={2}>
                Status: {selectedReport.status}
              </Typography>
              
              {selectedReport.assignedTo && selectedReport.assignedTo !== 'Unassigned' && (
                <Button 
                  variant="outlined" 
                  color="error"
                  fullWidth
                  onClick={unassignRepair}
                >
                  Unassign Repair Task
                </Button>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Dialog for assigning repair tasks */}
      <Dialog open={assignDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Assign Repair Task</DialogTitle>
        <DialogContent>
          {selectedReport && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                Report ID: {selectedReport.reportId || selectedReport._id || selectedReport.id}
              </Typography>
              
              <Typography variant="body2" color="text.secondary" mb={2}>
                Region: {selectedReport.region} | Severity: {selectedReport.severity}
              </Typography>
              
              <FormControl fullWidth margin="dense">
                <InputLabel>Select Field Worker</InputLabel>
                <Select
                  value={selectedWorker}
                  label="Select Field Worker"
                  onChange={(e) => {
                    console.log('Selected worker:', e.target.value);
                    setSelectedWorker(e.target.value);
                  }}
                >
                  <MenuItem value="">
                    <em>Select a worker</em>
                  </MenuItem>
                  {fieldWorkers.map(worker => (
                    <MenuItem key={worker.id} value={worker.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Avatar sx={{ width: 24, height: 24, mr: 1 }}>
                          {worker.name ? worker.name[0] : '?'}
                        </Avatar>
                        {worker.name} - {worker.specialization || 'General'} ({worker.region || 'All regions'})
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText>
                  {fieldWorkers.length ? 'Select a field worker to assign this task' : 'No field workers available'}
                </FormHelperText>
              </FormControl>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button 
            onClick={assignRepair} 
            variant="contained" 
            color="primary"
            disabled={!selectedWorker}
          >
            Assign Task
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity} 
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default Dashboard;