import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Paper, 
  TextField, 
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
  Divider,
  Chip,
  IconButton,
  Tooltip as MuiTooltip
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

const Historical = () => {
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [region, setRegion] = useState('all');
  const [damageType, setDamageType] = useState('all');
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [view, setView] = useState('reports'); // 'reports', 'trends'
  
  // Mock data - replace with actual API calls
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      // Simulate API call
      setTimeout(() => {
        const mockReports = [
          { 
            id: 1, 
            date: '2023-01-15', 
            location: 'North Wing', 
            region: 'North',
            damageType: 'Water Damage', 
            status: 'Repaired',
            severity: 'High',
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Ceiling leak causing water damage to floor and walls'
          },
          { 
            id: 2, 
            date: '2023-02-20', 
            location: 'South Entrance', 
            region: 'South',
            damageType: 'Structural', 
            status: 'Repaired',
            severity: 'Medium',
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Cracks in support column'
          },
          { 
            id: 3, 
            date: '2023-03-10', 
            location: 'East Corridor', 
            region: 'East',
            damageType: 'Electrical', 
            status: 'Repaired',
            severity: 'Low',
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Exposed wiring in wall socket'
          },
          { 
            id: 4, 
            date: '2023-04-05', 
            location: 'West Building', 
            region: 'West',
            damageType: 'Water Damage', 
            status: 'Pending',
            severity: 'Medium',
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Water leakage from pipes in the basement area affecting storage'
          },
          { 
            id: 5, 
            date: '2023-05-12', 
            location: 'North Conference Room', 
            region: 'North',
            damageType: 'Other', 
            status: 'Repaired',
            severity: 'Low',
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Damage to ceiling tiles from HVAC malfunction'
          },
          { 
            id: 6, 
            date: '2023-06-18', 
            location: 'East Wing Bathroom', 
            region: 'East',
            damageType: 'Water Damage', 
            status: 'In Progress',
            severity: 'High',
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Severe flooding from broken water main affecting multiple rooms'
          },
        ];
        setReports(mockReports);
        setLoading(false);
      }, 1000);
    };
    
    fetchData();
  }, []);

  const handleFilter = () => {
    setLoading(true);
    // Simulate filtering API call
    setTimeout(() => {
      // In a real app, this would be an API call with the filter parameters
      setLoading(false);
    }, 800);
  };

  const handleExportData = () => {
    // Create CSV content
    const headers = ['ID', 'Date', 'Location', 'Region', 'Damage Type', 'Status', 'Severity', 'Description'];
    const csvContent = [
      headers.join(','),
      ...reports.map(report => [
        report.id,
        report.date,
        report.location,
        report.region,
        report.damageType,
        report.status,
        report.severity,
        `"${report.description.replace(/"/g, '""')}"`
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
  };

  // Mock data for visualizations
  const trendData = [
    { month: 'Jan', waterDamage: 4, structural: 2, electrical: 1, other: 1 },
    { month: 'Feb', waterDamage: 3, structural: 3, electrical: 2, other: 0 },
    { month: 'Mar', waterDamage: 2, structural: 1, electrical: 3, other: 2 },
    { month: 'Apr', waterDamage: 5, structural: 2, electrical: 1, other: 1 },
    { month: 'May', waterDamage: 3, structural: 4, electrical: 2, other: 0 },
    { month: 'Jun', waterDamage: 2, structural: 3, electrical: 4, other: 1 },
  ];

  const regionData = [
    { name: 'North', value: 12 },
    { name: 'South', value: 8 },
    { name: 'East', value: 15 },
    { name: 'West', value: 10 },
  ];

  const severityData = [
    { month: 'Jan', high: 3, medium: 4, low: 1 },
    { month: 'Feb', high: 2, medium: 3, low: 3 },
    { month: 'Mar', high: 1, medium: 4, low: 3 },
    { month: 'Apr', high: 4, medium: 2, low: 3 },
    { month: 'May', high: 2, medium: 5, low: 2 },
    { month: 'Jun', high: 3, medium: 3, low: 4 },
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
  const STATUS_COLORS = {
    'Repaired': '#4caf50',
    'Pending': '#ff9800',
    'In Progress': '#2196f3'
  };
  
  const SEVERITY_COLORS = {
    'High': '#f44336',
    'Medium': '#ff9800',
    'Low': '#4caf50'
  };

  const getStatusChip = (status) => {
    return (
      <Chip 
        label={status} 
        size="small" 
        sx={{ 
          backgroundColor: STATUS_COLORS[status] || '#757575',
          color: 'white',
          fontWeight: 'bold'
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
          backgroundColor: SEVERITY_COLORS[severity] || '#757575',
          color: 'white',
          fontWeight: 'bold'
        }} 
      />
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      
      <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <FilterAltIcon sx={{ mr: 1 }} />
          Filter Reports
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="Start Date"
                value={startDate}
                onChange={(newValue) => setStartDate(newValue)}
                renderInput={(params) => <TextField {...params} fullWidth />}
                slotProps={{ textField: { fullWidth: true, variant: 'outlined' } }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} md={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="End Date"
                value={endDate}
                onChange={(newValue) => setEndDate(newValue)}
                renderInput={(params) => <TextField {...params} fullWidth />}
                slotProps={{ textField: { fullWidth: true, variant: 'outlined' } }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Region</InputLabel>
              <Select
                value={region}
                label="Region"
                onChange={(e) => setRegion(e.target.value)}
              >
                <MenuItem value="all">All Regions</MenuItem>
                <MenuItem value="north">North</MenuItem>
                <MenuItem value="south">South</MenuItem>
                <MenuItem value="east">East</MenuItem>
                <MenuItem value="west">West</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Damage Type</InputLabel>
              <Select
                value={damageType}
                label="Damage Type"
                onChange={(e) => setDamageType(e.target.value)}
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="water">Water Damage</MenuItem>
                <MenuItem value="structural">Structural</MenuItem>
                <MenuItem value="electrical">Electrical</MenuItem>
                <MenuItem value="other">Other</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button 
              variant="contained" 
              color="primary" 
              fullWidth 
              onClick={handleFilter}
              disabled={loading}
              sx={{ height: '56px', borderRadius: 2 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Apply Filters'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      <Box sx={{ mb: 3 }}>
        <Tabs 
          value={view} 
          onChange={(e, newValue) => setView(newValue)}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
          sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Archived Reports" value="reports" />
          <Tab label="Trend Analysis" value="trends" />
        </Tabs>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && view === 'reports' && (
        <>
          {selectedReport ? (
            <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3 }}>
              <Button 
                variant="outlined" 
                sx={{ mb: 3 }}
                onClick={() => setSelectedReport(null)}
                startIcon={<ArrowBackIcon />}
              >
                Back to Reports
              </Button>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                Report #{selectedReport.id} - {selectedReport.location}
              </Typography>
              <Divider sx={{ mb: 3 }} />
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ mb: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="subtitle2" color="textSecondary">Date:</Typography>
                        <Typography variant="body1" sx={{ fontWeight: 'medium' }}>{selectedReport.date}</Typography>
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
                    </Grid>
                  </Box>
                  <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Description:</Typography>
                  <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
                    <Typography paragraph>{selectedReport.description}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Before/After Comparison</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card sx={{ boxShadow: 3, borderRadius: 2, overflow: 'hidden' }}>
                        <CardMedia
                          component="img"
                          height="200"
                          image={selectedReport.beforeImage}
                          alt="Before repair"
                          sx={{ objectFit: 'cover' }}
                        />
                        <CardContent sx={{ bgcolor: '#f5f5f5' }}>
                          <Typography variant="body2" sx={{ fontWeight: 'medium', textAlign: 'center' }}>Before Repair</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card sx={{ boxShadow: 3, borderRadius: 2, overflow: 'hidden' }}>
                        <CardMedia
                          component="img"
                          height="200"
                          image={selectedReport.afterImage}
                          alt="After repair"
                          sx={{ objectFit: 'cover' }}
                        />
                        <CardContent sx={{ bgcolor: '#f5f5f5' }}>
                          <Typography variant="body2" sx={{ fontWeight: 'medium', textAlign: 'center' }}>After Repair</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </Paper>
          ) : (
            <Paper sx={{ p: 3, borderRadius: 2, boxShadow: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                Archived Reports
              </Typography>
              <Grid container spacing={3}>
                {reports.map((report) => (
                  <Grid item xs={12} md={4} key={report.id}>
                    <Card 
                      sx={{ 
                        cursor: 'pointer', 
                        transition: 'transform 0.2s, box-shadow 0.2s',
                        borderRadius: 2,
                        overflow: 'hidden',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 6
                        }
                      }} 
                      onClick={() => setSelectedReport(report)}
                    >
                      <CardMedia
                        component="img"
                        height="140"
                        image={report.beforeImage}
                        alt={report.location}
                        sx={{ objectFit: 'cover' }}
                      />
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                            {report.location}
                          </Typography>
                          <MuiTooltip title="View details">
                            <IconButton size="small" color="primary">
                              <VisibilityIcon />
                            </IconButton>
                          </MuiTooltip>
                        </Box>
                        <Typography color="textSecondary" variant="body2" sx={{ mb: 1 }}>
                          ID: #{report.id} | {report.date}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                          <Chip 
                            label={report.damageType} 
                            size="small" 
                            sx={{ bgcolor: '#e3f2fd', color: '#1976d2' }} 
                          />
                          {getStatusChip(report.status)}
                          {getSeverityChip(report.severity)}
                        </Box>
                        <Typography variant="body2" sx={{ 
                          mt: 1, 
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                        }}>
                          {report.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          )}
        </>
      )}

      {!loading && view === 'trends' && (
        <Paper sx={{ p: 3, borderRadius: 2, boxShadow: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2', mb: 3 }}>
            Damage Analysis Dashboard (2023)
          </Typography>
          <Grid container spacing={4}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 1, mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                  Damage Type Trends by Month
                </Typography>
                <Box sx={{ height: 350, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={trendData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
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
              </Paper>
              
              <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 1 }}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                  Severity Trends Over Time
                </Typography>
                <Box sx={{ height: 300, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={severityData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
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
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 1, mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                  Repairs by Region
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
              </Paper>
              
              <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 1 }}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                  Key Statistics
                </Typography>
                <Box sx={{ p: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Paper sx={{ p: 2, bgcolor: '#e3f2fd', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>45</Typography>
                        <Typography variant="body2">Total Reports</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6}>
                      <Paper sx={{ p: 2, bgcolor: '#e8f5e9', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#4caf50' }}>38</Typography>
                        <Typography variant="body2">Repaired</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6}>
                      <Paper sx={{ p: 2, bgcolor: '#fff3e0', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#ff9800' }}>5</Typography>
                        <Typography variant="body2">In Progress</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6}>
                      <Paper sx={{ p: 2, bgcolor: '#ffebee', borderRadius: 2, textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#f44336' }}>12</Typography>
                        <Typography variant="body2">High Severity</Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Container>
  );
};

export default Historical;
