import React, { useState, useEffect } from 'react';
import { useTheme } from '@mui/material/styles';
import { 
  Box, Typography, Grid, FormControl, InputLabel, 
  Select, MenuItem, Card, CardContent,
  CardHeader, Avatar, IconButton,
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
  
  // Define colors based on the theme
  const colors = {
    primary: theme.palette.primary.main,
    primaryDark: theme.palette.primary.dark,
    secondary: theme.palette.secondary.main,
    success: theme.palette.success.main,
    warning: theme.palette.warning.main,
    error: theme.palette.error.main,
    surface: theme.palette.background.paper,
    border: theme.palette.divider,
    text: {
      primary: theme.palette.text.primary,
      secondary: theme.palette.text.secondary
    }
  };
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
      case 'Pending': return colors.warning;
      case 'Assigned': return colors.primary;
      case 'In-Progress': return colors.primaryDark;
      case 'Resolved': return colors.success;
      case 'Rejected': return colors.error;
      default: return colors.text.secondary;
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Low': return colors.primary;
      case 'Medium': return colors.warning;
      case 'High': return colors.error;
      case 'Critical': return colors.error;
      default: return colors.text.secondary;
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
        borderRadius: 1, 
        overflow: 'hidden',
        border: `1px solid ${colors.border}`,
        backgroundColor: colors.surface
      }}
    >
      <CardHeader
        avatar={
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {icon && (
              <Box 
                component="span" 
                sx={{ 
                  color: 'text.secondary',
                  mr: 1.5,
                  display: 'flex'
                }}
              >
                {icon}
              </Box>
            )}
          </Box>
        }
        action={
          <Box>
            <IconButton size="small" onClick={onExport}>
              <DownloadIcon fontSize="small" />
            </IconButton>
          </Box>
        }
        title={
          <Typography 
            variant="subtitle1" 
            fontWeight="500"
            color={colors.text.primary}
            sx={{ fontSize: '0.95rem' }}
          >
            {title}
          </Typography>
        }
        sx={{ 
          py: 1.5,
          px: 2,
          borderBottom: `1px solid ${colors.border}`
        }}
      />
      <CardContent sx={{ pt: 2, px: 2, height: 'calc(100% - 56px)' }}>
        <Box sx={{ height: '100%' }}>
          {children}
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ py: 1 }}>
        
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography 
            variant="subtitle1"
            sx={{ 
              fontWeight: 500,
              fontSize: '1rem',
              color: 'text.primary'
            }}
          >
            Analytics Overview
          </Typography>
          <FormControl 
            size="small"
            sx={{ 
              minWidth: 180,
              '& .MuiOutlinedInput-root': {
                borderRadius: 1,
                fontSize: '0.9rem',
                '&:hover fieldset': {
                  borderColor: colors.primary,
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
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
            <CircularProgress size={30} thickness={4} sx={{ color: 'primary.main' }} />
          </Box>
        ) : error ? (
          <Box sx={{ 
            textAlign: 'center', 
            color: 'error.main', 
            p: 3, 
            border: '1px solid', 
            borderColor: 'error.light', 
            borderRadius: 1, 
            bgcolor: 'rgba(244, 67, 54, 0.05)'
          }}>
            <Typography variant="body2">{error}</Typography>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {/* Daily/Weekly Trends */}
            <Grid item xs={12} md={8}>
              <ChartCard 
                title="Daily Report Trends" 
                icon={<TrendingUpIcon />}
                chartType="line"
                onExport={() => exportData('trends')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={analyticsData.dailyTrends}>
                    <defs>
                      <linearGradient id="colorReports" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={colors.primary} stopOpacity={0.3}/>
                        <stop offset="95%" stopColor={colors.primary} stopOpacity={0.05}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.border} vertical={false} />
                    <XAxis dataKey="date" stroke={colors.text.secondary} fontSize={11} tickMargin={8} />
                    <YAxis stroke={colors.text.secondary} fontSize={11} tickMargin={8} />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 2, 
                        boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                        border: '1px solid',
                        borderColor: colors.border,
                        fontSize: '0.85rem'
                      }} 
                    />
                    <Legend wrapperStyle={{ fontSize: '0.85rem' }} />
                    <Area 
                      type="monotone" 
                      dataKey="reports" 
                      stroke={colors.primary} 
                      fillOpacity={1}
                      fill="url(#colorReports)"
                      strokeWidth={2}
                      activeDot={{ r: 6, strokeWidth: 0, fill: colors.primaryDark }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>
            </Grid>
            
            {/* Severity Distribution */}
            <Grid item xs={12} md={4}>
              <ChartCard 
                title="Severity Distribution" 
                icon={<PieChartIcon />}
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
                      outerRadius={80}
                      innerRadius={40}
                      paddingAngle={1}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {analyticsData.severityData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.color} 
                          stroke={colors.surface}
                          strokeWidth={1}
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 2, 
                        boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                        border: '1px solid',
                        borderColor: colors.border,
                        fontSize: '0.85rem'
                      }}
                      formatter={(value) => [`${value}%`]}
                    />
                    <Legend
                      layout="horizontal"
                      verticalAlign="bottom"
                      align="center"
                      wrapperStyle={{ fontSize: '0.85rem', paddingTop: '10px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
            </Grid>
            
            {/* Most Affected Zones */}
            <Grid item xs={12} md={6}>
              <ChartCard 
                title="Most Affected Zones" 
                icon={<LocationOnIcon />}
                chartType="bar"
                onExport={() => exportData('zones')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.affectedZones} barSize={20}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.border} vertical={false} />
                    <XAxis 
                      dataKey="name" 
                      stroke={colors.text.secondary} 
                      fontSize={11} 
                      tickMargin={8}
                    />
                    <YAxis 
                      stroke={colors.text.secondary} 
                      fontSize={11}
                      tickMargin={8}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 2, 
                        boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                        border: '1px solid',
                        borderColor: colors.border,
                        fontSize: '0.85rem'
                      }}
                    />
                    <Legend 
                      wrapperStyle={{ 
                        fontSize: '0.85rem',
                        paddingTop: '10px'
                      }} 
                    />
                    <Bar 
                      dataKey="reports" 
                      name="Number of Reports"
                      radius={[2, 2, 0, 0]}
                    >
                      {analyticsData.affectedZones.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`}
                          fill={colors.primary} 
                          fillOpacity={0.7}
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
                icon={<AssessmentIcon />}
                chartType="bar"
                onExport={() => exportData('types')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.damageTypes} layout="vertical" barSize={20}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.border} horizontal={false} />
                    <XAxis 
                      type="number" 
                      stroke={colors.text.secondary} 
                      fontSize={11}
                      tickMargin={8}
                    />
                    <YAxis 
                      dataKey="name" 
                      type="category" 
                      width={100} 
                      stroke={colors.text.secondary} 
                      fontSize={11}
                      tickMargin={8}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: 2, 
                        boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                        border: '1px solid',
                        borderColor: colors.border,
                        fontSize: '0.85rem'
                      }}
                    />
                    <Legend 
                      wrapperStyle={{ 
                        fontSize: '0.85rem',
                        paddingTop: '10px'
                      }} 
                    />
                    <Bar 
                      dataKey="value" 
                      name="Number of Reports"
                      radius={[0, 2, 2, 0]}
                    >
                      {analyticsData.damageTypes.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={colors.secondary}
                          fillOpacity={0.7}
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
                icon={<BuildIcon />}
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
                          outerRadius={80}
                          innerRadius={40}
                          paddingAngle={1}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {analyticsData.repairStatus.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={entry.color} 
                              stroke={colors.surface}
                              strokeWidth={1}
                            />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ 
                            borderRadius: 2, 
                            boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                            border: '1px solid',
                            borderColor: colors.border,
                            fontSize: '0.85rem'
                          }}
                          formatter={(value) => [`${value}%`]}
                        />
                        <Legend
                          layout="horizontal"
                          verticalAlign="bottom"
                          align="center"
                          wrapperStyle={{ fontSize: '0.85rem', paddingTop: '10px' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      {analyticsData.repairStatus.map((status, index) => (
                        <Box sx={{ mb: 2 }} key={status.name}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                color: 'text.primary',
                                fontSize: '0.85rem'
                              }}
                            >
                              {status.name}
                            </Typography>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontWeight: 500,
                                fontSize: '0.85rem'
                              }}
                            >
                              {status.value}%
                            </Typography>
                          </Box>
                          <Box 
                            sx={{ 
                              width: '100%', 
                              height: 6, 
                              bgcolor: 'background.default', 
                              borderRadius: 0.5,
                              overflow: 'hidden'
                            }}
                          >
                            <Box 
                              sx={{ 
                                width: `${status.value}%`, 
                                height: '100%', 
                                bgcolor: status.color
                              }} 
                            />
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              </ChartCard>
            </Grid>
          </Grid>
        )}
      </Box>
  );
}

export default Analytics;
