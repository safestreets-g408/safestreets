import React, { useState, useEffect, useCallback } from 'react';
import { 
  Grid, 
  Box, 
  Typography, 
  IconButton, 
  Button,
  Stack,
  Tooltip
} from '@mui/material';

// Icons
import DashboardIcon from '@mui/icons-material/Dashboard';
import ReportIcon from '@mui/icons-material/Report';
import BuildIcon from '@mui/icons-material/Build';
import CheckCircleIcon from '@mui/icons-material/CheckCircle'; 
import WarningIcon from '@mui/icons-material/Warning';
import RefreshIcon from '@mui/icons-material/Refresh';

// Custom Components
import StatCard from '../components/dashboard/StatCard';
import RecentReports from '../components/dashboard/RecentReports';
import ActivityFeed from '../components/dashboard/ActivityFeed';
import QuickActions from '../components/dashboard/QuickActions';
import AiReportsDialog from '../components/dashboard/AiReportsDialog';
import CreateDamageReportDialog from '../components/dashboard/CreateDamageReportDialog';

// API
import { api } from '../utils/api';
import { API_ENDPOINTS } from '../config/constants';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState({
    totalReports: 0,
    pendingRepairs: 0,
    completedRepairs: 0,
    criticalDamages: 0,
    recentReports: []
  });
  
  const [aiReports, setAiReports] = useState([]);
  const [aiReportsLoading, setAiReportsLoading] = useState(false);
  const [aiReportsError, setAiReportsError] = useState(null);
  const [aiReportsOpen, setAiReportsOpen] = useState(false);
  const [selectedAiReport, setSelectedAiReport] = useState(null);
  
  const [fieldWorkers, setFieldWorkers] = useState([]);
  const [selectedFieldWorker, setSelectedFieldWorker] = useState('');
  
  const [createReportOpen, setCreateReportOpen] = useState(false);
  const [reportFormData, setReportFormData] = useState({
    region: '',
    location: '',
    damageType: '',
    severity: '',
    priority: '',
    description: '',
    reporter: 'admin@example.com', 
    aiReportId: null, 
    assignToWorker: false
  });
  const [createReportLoading, setCreateReportLoading] = useState(false);
  const [createReportError, setCreateReportError] = useState(null);
  
  const [recentActivity, setRecentActivity] = useState([]);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      console.log('Fetching dashboard data...');
      
      const reportsData = await api.get(`${API_ENDPOINTS.DAMAGE_REPORTS}/reports`);
      console.log('Reports data fetched:', reportsData.length || 0, 'reports');
      
      const totalReports = reportsData.length || 0;
      const pendingRepairs = reportsData.filter(r => r.status === 'Pending').length;
      const completedRepairs = reportsData.filter(r => r.status === 'Completed').length;
      const criticalDamages = reportsData.filter(r => r.severity === 'HIGH' || r.severity === 'Critical').length;
      
      let recentReports = [];
      if (Array.isArray(reportsData) && reportsData.length > 0) {
        const sortedReports = [...reportsData].sort(
          (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
        );
        
        recentReports = sortedReports.slice(0, 3).map(report => ({
          id: report._id,
          title: `${report.damageType} at ${report.location}`,
          severity: report.severity,
          location: report.location,
          timestamp: new Date(report.createdAt).toLocaleDateString(),
          status: report.status
        }));
      }
      
      
      setDashboardData({
        totalReports,
        pendingRepairs,
        completedRepairs,
        criticalDamages,
        recentReports
      });
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setDashboardData({
        totalReports: 0,
        pendingRepairs: 0,
        completedRepairs: 0,
        criticalDamages: 0,
        recentReports: []
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);


  const fetchAiReports = async () => {
    try {
      setAiReportsLoading(true);
      setAiReportsError(null);
      const response = await api.get(`${API_ENDPOINTS.IMAGES}/reports`);
      
      let reportsData = [];
      if (Array.isArray(response)) {
        reportsData = response;
      } else if (response?.reports && Array.isArray(response.reports)) {
        reportsData = response.reports;
      } else if (response && typeof response === 'object') {
        const possibleReports = Object.values(response).find(val => Array.isArray(val));
        reportsData = Array.isArray(possibleReports) ? possibleReports : [];
      }
      
      
      if (reportsData.length > 0) {
        const sample = {...reportsData[0]};
        if (sample.annotatedImageBase64) {
          sample.annotatedImageBase64 = '[Base64 image data truncated]';
        }
      }
      
      setAiReports(reportsData || []);
    } catch (error) {
      console.error('Error fetching AI reports:', error);
      setAiReportsError(error.message || 'Failed to fetch AI reports');
      setAiReports([]);
    } finally {
      setAiReportsLoading(false);
    }
  };

  const fetchFieldWorkers = async () => {
    try {
      const workers = await api.get(`${API_ENDPOINTS.FIELD_WORKERS}`);
      const workersData = Array.isArray(workers) ? workers : 
                         (workers.fieldWorkers || workers.workers || []);
      
      setFieldWorkers(workersData);
    } catch (error) {
      console.error('Error fetching field workers:', error);
      setFieldWorkers([]);
    }
  };

  const handleViewAnalytics = async () => {
    try {
      
      setAiReportsLoading(true);
      
      await Promise.all([
        fetchAiReports(),
        fetchFieldWorkers()
      ]);
      
      setAiReportsOpen(true);
    } catch (error) {
      console.error('Error in View Analytics handler:', error);
      setAiReportsOpen(true);
    }
  };

  const handleSelectAiReport = (report) => {
    if (report.damageReportGenerated) {
      alert('Damage report already generated for this AI report.');
      return;
    }
    setSelectedAiReport(report);

    setReportFormData({
      ...reportFormData,
      region: reportFormData.region || '',
      location: reportFormData.location || '',
      damageType: report.damageType || '',
      severity: report.severity || '',
      priority: report.priority?.toString() || '',
      aiReportId: report.id,
      reporter: 'admin@example.com',
    });


    setCreateReportOpen(true);
  };

  const resetFormData = () => {
    setReportFormData({
      region: '',
      location: '',
      damageType: '',
      severity: '',
      priority: '',
      description: '',
      reporter: 'admin@example.com',
      aiReportId: null,
      assignToWorker: false
    });
    setSelectedFieldWorker('');
    setSelectedAiReport(null);
    setCreateReportError(null);
  };

  const handleDialogClose = () => {
    resetFormData();
    setCreateReportOpen(false);
  };

  const handleFormInputChange = (e) => {
    const { name, value } = e.target;
    setReportFormData({
      ...reportFormData,
      [name]: value
    });
  };

  const handleFieldWorkerChange = (e) => {
    setSelectedFieldWorker(e.target.value);
    setReportFormData({
      ...reportFormData,
      assignToWorker: !!e.target.value
    });
  };

  const handleCreateReport = async () => {
    try {
      setCreateReportLoading(true);
      setCreateReportError(null);
      
      const aiReportId = reportFormData.aiReportId || (selectedAiReport?.id);
      
      if (!aiReportId) {
        setCreateReportError('AI Report ID is missing. Please select an AI report first.');
        setCreateReportLoading(false);
        return;
      }
      

      const requiredFields = ['region', 'location', 'damageType', 'severity', 'priority'];
      const missingFields = requiredFields.filter(field => !reportFormData[field]);
      
      if (missingFields.length > 0) {
        setCreateReportError(`Please fill in all required fields: ${missingFields.join(', ')}`);
        setCreateReportLoading(false);
        return;
      }

      let endpoint = `${API_ENDPOINTS.DAMAGE_REPORTS}/reports/create-from-ai`;
      let payload = { 
        ...reportFormData,
        aiReportId
      };



      if (reportFormData.assignToWorker && selectedFieldWorker) {
        endpoint = `${API_ENDPOINTS.DAMAGE_REPORTS}/reports/create-and-assign`;
        payload.workerId = selectedFieldWorker;
      }

      const response = await api.post(endpoint, payload);
      
      const isSuccess = response.success || response.report || 
                       (response.message && response.message.includes('success'));
      
      if (isSuccess) {
        resetFormData();
        
        setCreateReportOpen(false);
        setAiReportsOpen(false);
        
        setTimeout(() => {
          fetchDashboardData();
        }, 1000); 
      } else {
        throw new Error(response.message || 'Failed to create damage report');
      }
    } catch (error) {
      console.error('Error creating damage report:', error);
      setCreateReportError(error.message || 'Failed to create damage report');
    } finally {
      setCreateReportLoading(false);
    }
  };

  const handleAiReportsClose = () => {
    setAiReportsOpen(false);
  };

  const updateRecentActivity = useCallback(() => {
    const aiActivity = aiReports.map(report => ({
      id: report.id,
      type: 'ai-report',
      title: `AI Report: ${report.damageType}`,
      description: `Predicted damage type: ${report.damageType}, Severity: ${report.severity}`,
      time: new Date(report.createdAt).toLocaleTimeString(),
      severity: report.severity,
      location: report.location || 'Unknown'
    }));

    const damageActivity = dashboardData.recentReports.map(report => ({
      id: report.id,
      type: 'damage-report',
      title: `Damage Report: ${report.title}`,
      description: `Reported damage type: ${report.severity}, Status: ${report.status}`,
      time: report.timestamp,
      severity: report.severity,
      location: report.location
    }));

    const combinedActivity = [...aiActivity, ...damageActivity];
    setRecentActivity(combinedActivity);
  }, [aiReports, dashboardData.recentReports]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    updateRecentActivity();
  }, [aiReports, dashboardData.recentReports, updateRecentActivity]);


  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ mb: 4 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
            <Box>
              <Typography 
                variant="h4" 
                sx={{ 
                  fontWeight: 700,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 1,
                }}
              >
                Welcome back, Admin! 👋
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ fontWeight: 500 }}>
                Here's what's happening with SafeStreets today
              </Typography>
            </Box>
            
            <Stack direction="row" spacing={2}>
              <Tooltip title="Refresh Dashboard">
                <IconButton 
                  onClick={fetchDashboardData}
                  disabled={loading}
                  sx={{
                    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%)',
                      transform: 'rotate(180deg)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              
              <Button
                variant="contained"
                startIcon={<DashboardIcon />}
                onClick={handleViewAnalytics}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  borderRadius: 3,
                  px: 3,
                  '&:hover': {
                    background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
                    transform: 'translateY(-2px)',
                  },
                  transition: 'all 0.3s ease',
                }}
              >
                View Analytics
              </Button>
            </Stack>
          </Stack>
        </Box>

      {/* Stats Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<ReportIcon />}
            title="Total Reports"
            value={dashboardData.totalReports}
            change={12}
            color="primary"
            loading={loading}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<BuildIcon />}
            title="Pending Repairs"
            value={dashboardData.pendingRepairs}
            change={-8}
            color="warning"
            loading={loading}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<CheckCircleIcon />}
            title="Completed Repairs"
            value={dashboardData.completedRepairs}
            change={15}
            color="success"
            loading={loading}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<WarningIcon />}
            title="Critical Issues"
            value={dashboardData.criticalDamages}
            change={-5}
            color="error"
            loading={loading}
          />
        </Grid>
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Recent Reports */}
        <Grid item xs={12} lg={8}>
          <RecentReports reports={dashboardData.recentReports} loading={loading} />
        </Grid>

        {/* Activity Feed */}
        <Grid item xs={12} lg={4}>
          <ActivityFeed activities={recentActivity} />
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <QuickActions />
        </Grid>
      </Grid>

      {/* AI Reports Dialog */}
      <AiReportsDialog
        open={aiReportsOpen}
        onClose={handleAiReportsClose}
        reports={aiReports}
        loading={aiReportsLoading}
        error={aiReportsError}
        onSelectReport={handleSelectAiReport}
      />

      {/* Create Damage Report Dialog */}
      <CreateDamageReportDialog 
        open={createReportOpen}
        onClose={handleDialogClose}
        onSubmit={handleCreateReport}
        selectedAiReport={selectedAiReport}
        formData={reportFormData}
        onFormChange={handleFormInputChange}
        fieldWorkers={fieldWorkers}
        onFieldWorkerChange={handleFieldWorkerChange}
        selectedFieldWorker={selectedFieldWorker}
        loading={createReportLoading}
        error={createReportError}
      />
    </Box>
  );
};

export default Dashboard;
