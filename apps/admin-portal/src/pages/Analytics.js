import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Grid, FormControl, InputLabel, 
  Select, MenuItem,
  useTheme, alpha, Card, CardContent,
  CardHeader, Avatar, IconButton, Fade, Zoom,
  CircularProgress
} from '@mui/material';
import { 
  BarChart, Bar, PieChart, Pie, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Cell, AreaChart, Area
} from 'recharts';
import DownloadIcon from '@mui/icons-material/Download';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import PieChartIcon from '@mui/icons-material/PieChart';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import BuildIcon from '@mui/icons-material/Build';
import AssessmentIcon from '@mui/icons-material/Assessment';
import RefreshIcon from '@mui/icons-material/Refresh';
import { api } from '../utils/api';

function Analytics() {
  const theme = useTheme();
  const [timeframe, setTimeframe] = useState('7days');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analyticsData, setAnalyticsData] = useState({
    dailyTrends: [],
    severityData: [],
    affectedZones: [],
    damageTypes: [],
    repairStatus: []
  });

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        setLoading(true);
        setError(null);

        const endDate = new Date();
        const startDate = new Date();
        switch (timeframe) {
          case '7days':
            startDate.setDate(endDate.getDate() - 7);
            break;
          case '30days':
            startDate.setDate(endDate.getDate() - 30);
            break;
          case '90days':
            startDate.setDate(endDate.getDate() - 90);
            break;
          case 'year':
            startDate.setFullYear(endDate.getFullYear() - 1);
            break;
          default:
            startDate.setDate(endDate.getDate() - 7);
        }

        // Fetch reports within date range
        const response = await api.get('/damage/reports', {
          params: {
            startDate: startDate.toISOString(),
            endDate: endDate.toISOString()
          }
        });

        // Process data for different charts
        const processedData = processReportsData(response, startDate, endDate);
        setAnalyticsData(processedData);
      } catch (err) {
        console.error('Error fetching analytics:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeframe]);

  const processReportsData = (reports, startDate, endDate) => {
    // Daily trends
    const dailyTrends = getDailyTrends(reports, startDate, endDate);

    // Severity distribution
    const severityData = reports.reduce((acc, report) => {
      const severity = report.severity || 'Unknown';
      const existingEntry = acc.find(item => item.name === severity);
      if (existingEntry) {
        existingEntry.value++;
      } else {
        acc.push({
          name: severity,
          value: 1,
          color: getSeverityColor(severity)
        });
      }
      return acc;
    }, []);

    // Affected zones
    const affectedZones = reports.reduce((acc, report) => {
      const region = report.region || 'Unknown';
      const existingEntry = acc.find(item => item.name === region);
      if (existingEntry) {
        existingEntry.reports++;
      } else {
        acc.push({ name: region, reports: 1 });
      }
      return acc;
    }, []);

    // Damage types
    const damageTypes = reports.reduce((acc, report) => {
      const type = report.damageType || 'Other';
      const existingEntry = acc.find(item => item.name === type);
      if (existingEntry) {
        existingEntry.value++;
      } else {
        acc.push({ name: type, value: 1 });
      }
      return acc;
    }, []);

    // Repair status
    const repairStatus = reports.reduce((acc, report) => {
      const status = report.status || 'Pending';
      const existingEntry = acc.find(item => item.name === status);
      if (existingEntry) {
        existingEntry.value++;
      } else {
        acc.push({
          name: status,
          value: 1,
          color: getStatusColor(status)
        });
      }
      return acc;
    }, []);

    // Convert counts to percentages for pie charts
    const convertToPercentages = (data) => {
      const total = data.reduce((sum, item) => sum + item.value, 0);
      return data.map(item => ({
        ...item,
        value: Number(((item.value / total) * 100).toFixed(1))
      }));
    };

    return {
      dailyTrends,
      severityData: convertToPercentages(severityData),
      affectedZones: affectedZones.sort((a, b) => b.reports - a.reports),
      damageTypes: damageTypes.sort((a, b) => b.value - a.value),
      repairStatus: convertToPercentages(repairStatus)
    };
  };

  const getDailyTrends = (reports, startDate, endDate) => {
    const dates = {};
    let currentDate = new Date(startDate);

    // Initialize all dates with 0
    while (currentDate <= endDate) {
      dates[currentDate.toISOString().split('T')[0]] = 0;
      currentDate.setDate(currentDate.getDate() + 1);
    }

    // Count reports per day
    reports.forEach(report => {
      const reportDate = new Date(report.createdAt).toISOString().split('T')[0];
      if (dates[reportDate] !== undefined) {
        dates[reportDate]++;
      }
    });

    // Convert to array format for chart
    return Object.entries(dates).map(([date, reports]) => ({
      date,
      reports
    }));
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'Pending': return theme.palette.warning.main;
      case 'Assigned': return theme.palette.info.main;
      case 'In-Progress': return theme.palette.primary.main;
      case 'Resolved': return theme.palette.success.main;
      case 'Rejected': return theme.palette.error.main;
      default: return theme.palette.grey[400];
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Low': return theme.palette.info.main;
      case 'Medium': return theme.palette.warning.main;
      case 'High': return theme.palette.error.main;
      case 'Critical': return theme.palette.secondary.main;
      default: return theme.palette.grey[400];
    }
  };

  const handleTimeframeChange = (event) => {
    setTimeframe(event.target.value);
  };
  
  const exportData = (dataType) => {
    console.log(`Exporting ${dataType} data...`);
    
    let dataToExport;
    switch(dataType) {
      case 'trends': dataToExport = analyticsData.dailyTrends; break;
      case 'severity': dataToExport = analyticsData.severityData; break;
      case 'zones': dataToExport = analyticsData.affectedZones; break;
      case 'types': dataToExport = analyticsData.damageTypes; break;
      case 'status': dataToExport = analyticsData.repairStatus; break;
      default: dataToExport = [];
    }
    
    const csvContent = "data:text/csv;charset=utf-8," 
      + Object.keys(dataToExport[0]).join(",") + "\n"
      + dataToExport.map(item => Object.values(item).join(",")).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${dataType}_data.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const ChartCard = ({ title, icon, chartType, children, onExport }) => (
    <Card 
      elevation={0}
      sx={{ 
        height: '100%', 
        borderRadius: 3, 
        boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.1)}`,
        transition: 'transform 0.3s, box-shadow 0.3s',
        overflow: 'hidden',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: `0 12px 28px ${alpha(theme.palette.primary.main, 0.15)}`
        }
      }}
    >
      <CardHeader
        avatar={
          <Avatar sx={{ bgcolor: alpha(theme.palette.primary.main, 0.1) }}>
            {icon}
          </Avatar>
        }
        action={
          <Box>
            <IconButton size="small" sx={{ mr: 1 }}>
              <RefreshIcon fontSize="small" />
            </IconButton>
            <IconButton 
              size="small"
              onClick={onExport}
              sx={{ 
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                '&:hover': {
                  bgcolor: alpha(theme.palette.primary.main, 0.2),
                }
              }}
            >
              <DownloadIcon fontSize="small" />
            </IconButton>
          </Box>
        }
        title={<Typography variant="h6" fontWeight="600">{title}</Typography>}
        sx={{ 
          pb: 0,
          '& .MuiCardHeader-title': { fontSize: '1.1rem' },
          borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`
        }}
      />
      <CardContent sx={{ pt: 2, height: 'calc(100% - 72px)' }}>
        <Zoom in={true} style={{ transitionDelay: '150ms' }}>
          <Box sx={{ height: '100%' }}>
            {children}
          </Box>
        </Zoom>
      </CardContent>
    </Card>
  );

  return (
    <Fade in={true}>
      <Box>
        <Typography 
          variant="h4" 
          gutterBottom 
          sx={{
            fontWeight: 'semi-bold'
          }}
        >
          Analytics & Insights
        </Typography>
        
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'flex-end' }}>
          <FormControl 
            sx={{ 
              minWidth: 200,
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                '&:hover fieldset': {
                  borderColor: theme.palette.primary.main,
                },
              }
            }}
          >
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={timeframe}
              label="Timeframe"
              onChange={handleTimeframeChange}
            >
              <MenuItem value="7days">Last 7 Days</MenuItem>
              <MenuItem value="30days">Last 30 Days</MenuItem>
              <MenuItem value="90days">Last 90 Days</MenuItem>
              <MenuItem value="year">Last Year</MenuItem>
              <MenuItem value="custom">Custom Range</MenuItem>
            </Select>
          </FormControl>
        </Box>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box sx={{ textAlign: 'center', color: 'error.main', p: 3 }}>
            <Typography>{error}</Typography>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {/* Daily/Weekly Trends */}
            <Grid item xs={12} md={8}>
              <ChartCard 
                title="Daily Report Trends" 
                icon={<TrendingUpIcon color="primary" />}
                chartType="line"
                onExport={() => exportData('trends')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={analyticsData.dailyTrends}>
                    <defs>
                      <linearGradient id="colorReports" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.8}/>
                        <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.text.secondary, 0.1)} />
                    <XAxis dataKey="date" stroke={theme.palette.text.secondary} />
                    <YAxis stroke={theme.palette.text.secondary} />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 8, 
                        boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                        border: 'none'
                      }} 
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="reports" 
                      stroke={theme.palette.primary.main} 
                      fillOpacity={1}
                      fill="url(#colorReports)"
                      strokeWidth={3}
                      activeDot={{ r: 8, strokeWidth: 0, fill: theme.palette.primary.dark }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>
            </Grid>
            
            {/* Severity Distribution */}
            <Grid item xs={12} md={4}>
              <ChartCard 
                title="Severity Distribution" 
                icon={<PieChartIcon color="primary" />}
                chartType="pie"
                onExport={() => exportData('severity')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={analyticsData.severityData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={90}
                      innerRadius={40}
                      paddingAngle={2}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {analyticsData.severityData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.color} 
                          stroke={theme.palette.background.paper}
                          strokeWidth={2}
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 8, 
                        boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                        border: 'none'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
            </Grid>
            
            {/* Most Affected Zones */}
            <Grid item xs={12} md={6}>
              <ChartCard 
                title="Most Affected Zones" 
                icon={<LocationOnIcon color="primary" />}
                chartType="bar"
                onExport={() => exportData('zones')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.affectedZones}>
                    <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.text.secondary, 0.1)} />
                    <XAxis dataKey="name" stroke={theme.palette.text.secondary} />
                    <YAxis stroke={theme.palette.text.secondary} />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 8, 
                        boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                        border: 'none'
                      }}
                    />
                    <Legend />
                    <Bar 
                      dataKey="reports" 
                      name="Number of Reports"
                      radius={[4, 4, 0, 0]}
                    >
                      {analyticsData.affectedZones.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={`${theme.palette.primary.main}${80 + index * 20}`} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </Grid>
            
            {/* Damage Type Breakdown */}
            <Grid item xs={12} md={6}>
              <ChartCard 
                title="Damage Type Breakdown" 
                icon={<AssessmentIcon color="primary" />}
                chartType="bar"
                onExport={() => exportData('types')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.damageTypes} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.text.secondary, 0.1)} />
                    <XAxis type="number" stroke={theme.palette.text.secondary} />
                    <YAxis dataKey="name" type="category" width={100} stroke={theme.palette.text.secondary} />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 8, 
                        boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                        border: 'none'
                      }}
                    />
                    <Legend />
                    <Bar 
                      dataKey="value" 
                      name="Number of Reports"
                      radius={[0, 4, 4, 0]}
                    >
                      {analyticsData.damageTypes.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={`${theme.palette.secondary.main}${80 + index * 20}`} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </Grid>
            
            {/* Repair Status Overview */}
            <Grid item xs={12}>
              <ChartCard 
                title="Repair Status Overview" 
                icon={<BuildIcon color="primary" />}
                chartType="pie"
                onExport={() => exportData('status')}
              >
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={analyticsData.repairStatus}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={90}
                          innerRadius={40}
                          paddingAngle={2}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {analyticsData.repairStatus.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={entry.color} 
                              stroke={theme.palette.background.paper}
                              strokeWidth={2}
                            />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ 
                            borderRadius: 8, 
                            boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                            border: 'none'
                          }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      {analyticsData.repairStatus.map((status, index) => (
                        <Fade 
                          in={true} 
                          key={status.name}
                          style={{ transitionDelay: `${index * 100}ms` }}
                        >
                          <Box sx={{ mb: 2.5 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="body1" fontWeight="500" color="text.primary">
                                {status.name}
                              </Typography>
                              <Typography variant="body1" fontWeight="bold">
                                {status.value}%
                              </Typography>
                            </Box>
                            <Box 
                              sx={{ 
                                width: '100%', 
                                height: 10, 
                                bgcolor: alpha(status.color, 0.15), 
                                borderRadius: 2,
                                overflow: 'hidden'
                              }}
                            >
                              <Box 
                                sx={{ 
                                  width: `${status.value}%`, 
                                  height: '100%', 
                                  bgcolor: status.color, 
                                  borderRadius: 2,
                                  boxShadow: `0 2px 8px ${alpha(status.color, 0.4)}`
                                }} 
                              />
                            </Box>
                          </Box>
                        </Fade>
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              </ChartCard>
            </Grid>
          </Grid>
        )}
      </Box>
    </Fade>
  );
}

export default Analytics;
