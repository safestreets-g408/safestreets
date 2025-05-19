import React, { useState } from 'react';
import { 
  Box, Typography, Grid, FormControl, InputLabel, 
  Select, MenuItem,
  useTheme, alpha, Card, CardContent,
  CardHeader, Avatar, IconButton, Fade, Zoom
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

function Analytics() {
  const theme = useTheme();
  const [timeframe, setTimeframe] = useState('7days');
  
  const dailyTrendsData = [
    { date: '2023-06-15', reports: 12 },
    { date: '2023-06-16', reports: 19 },
    { date: '2023-06-17', reports: 15 },
    { date: '2023-06-18', reports: 25 },
    { date: '2023-06-19', reports: 32 },
    { date: '2023-06-20', reports: 18 },
    { date: '2023-06-21', reports: 23 },
  ];
  
  const severityData = [
    { name: 'Low', value: 35, color: theme.palette.info.main },
    { name: 'Medium', value: 45, color: theme.palette.warning.main },
    { name: 'High', value: 15, color: theme.palette.error.main },
    { name: 'Critical', value: 5, color: theme.palette.secondary.main },
  ];
  
  const affectedZonesData = [
    { name: 'North', reports: 45 },
    { name: 'South', reports: 28 },
    { name: 'East', reports: 37 },
    { name: 'West', reports: 19 },
    { name: 'Central', reports: 52 },
  ];
  
  const damageTypeData = [
    { name: 'Structural', value: 38 },
    { name: 'Electrical', value: 22 },
    { name: 'Plumbing', value: 15 },
    { name: 'Flooding', value: 25 },
    { name: 'Fire', value: 8 },
    { name: 'Other', value: 12 },
  ];
  
  const repairStatusData = [
    { name: 'Pending', value: 30, color: theme.palette.warning.main },
    { name: 'Assigned', value: 15, color: theme.palette.info.main },
    { name: 'In-Progress', value: 25, color: theme.palette.primary.main },
    { name: 'Resolved', value: 25, color: theme.palette.success.main },
    { name: 'Rejected', value: 5, color: theme.palette.error.main },
  ];
  
  const handleTimeframeChange = (event) => {
    setTimeframe(event.target.value);
  };
  
  const exportData = (dataType) => {
    console.log(`Exporting ${dataType} data...`);
    
    let dataToExport;
    switch(dataType) {
      case 'trends': dataToExport = dailyTrendsData; break;
      case 'severity': dataToExport = severityData; break;
      case 'zones': dataToExport = affectedZonesData; break;
      case 'types': dataToExport = damageTypeData; break;
      case 'status': dataToExport = repairStatusData; break;
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
                <AreaChart data={dailyTrendsData}>
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
                    data={severityData}
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
                    {severityData.map((entry, index) => (
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
                <BarChart data={affectedZonesData}>
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
                    {affectedZonesData.map((entry, index) => (
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
                <BarChart data={damageTypeData} layout="vertical">
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
                    {damageTypeData.map((entry, index) => (
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
                        data={repairStatusData}
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
                        {repairStatusData.map((entry, index) => (
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
                    {repairStatusData.map((status, index) => (
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
      </Box>
    </Fade>
  );
}

export default Analytics;
