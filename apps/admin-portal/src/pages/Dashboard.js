import React, { useState, useEffect } from 'react';
import { 
  Grid, Paper, Typography, Box, Card, CardContent, 
  LinearProgress, IconButton, Avatar, List, ListItem,
  ListItemText, ListItemAvatar, Divider, Chip, Button,
  useTheme, alpha, CircularProgress
} from '@mui/material';
import ReportIcon from '@mui/icons-material/Report';
import BuildIcon from '@mui/icons-material/Build';
import CheckCircleIcon from '@mui/icons-material/CheckCircle'; 
import WarningIcon from '@mui/icons-material/Warning';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import RefreshIcon from '@mui/icons-material/Refresh';
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

const StatusChip = styled(Chip)(({ theme, status }) => ({
  borderRadius: theme.shape.borderRadius,
  fontWeight: 600,
  fontSize: '0.75rem',
  height: 24,
  boxShadow: status !== 'default' ? `0 2px 8px ${alpha(theme.palette[status].main, 0.3)}` : 'none',
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
          id: report.id,
          title: report.description || 'No description',
          severity: report.severity || 'Low',
          location: report.region || 'Unknown Location',
          timestamp: formatTimestamp(report.createdAt),
          status: report.status || 'Pending'
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

  useEffect(() => {
    fetchDashboardData();
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
                          <StatusChip 
                            label={report.status}
                            size="small"
                            status={
                              report.status === 'Completed' ? 'success' :
                              report.status === 'In Progress' ? 'warning' : 'default'
                            }
                            color={
                              report.status === 'Completed' ? 'success' :
                              report.status === 'In Progress' ? 'warning' : 'default'
                            }
                          />
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
    </Box>
  );
}

export default Dashboard;
