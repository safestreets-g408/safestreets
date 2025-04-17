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
  CircularProgress 
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
  Cell
} from 'recharts';

const Historical = () => {
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [region, setRegion] = useState('all');
  const [damageType, setDamageType] = useState('all');
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [view, setView] = useState('reports'); // 'reports', 'heatmap', 'trends'
  
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
            beforeImage: 'https://via.placeholder.com/300x200?text=Before',
            afterImage: 'https://via.placeholder.com/300x200?text=After',
            description: 'Exposed wiring in wall socket'
          },
          // Add more mock data as needed
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
    const headers = ['ID', 'Date', 'Location', 'Region', 'Damage Type', 'Status', 'Description'];
    const csvContent = [
      headers.join(','),
      ...reports.map(report => [
        report.id,
        report.date,
        report.location,
        report.region,
        report.damageType,
        report.status,
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

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom component="h1">
        Historical Analysis
      </Typography>
      
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
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
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
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
            <FormControl fullWidth>
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
            >
              {loading ? <CircularProgress size={24} /> : 'Apply Filters'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      <Box sx={{ mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item>
            <Button 
              variant={view === 'reports' ? 'contained' : 'outlined'} 
              onClick={() => setView('reports')}
            >
              Archived Reports
            </Button>
          </Grid>
          <Grid item>
            <Button 
              variant={view === 'trends' ? 'contained' : 'outlined'} 
              onClick={() => setView('trends')}
            >
              Trend Analysis
            </Button>
          </Grid>
          <Grid item>
            <Button 
              variant="outlined" 
              color="secondary" 
              onClick={handleExportData}
            >
              Export Data
            </Button>
          </Grid>
        </Grid>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && view === 'reports' && (
        <>
          {selectedReport ? (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Button 
                variant="outlined" 
                sx={{ mb: 2 }}
                onClick={() => setSelectedReport(null)}
              >
                Back to Reports
              </Button>
              <Typography variant="h5" gutterBottom>
                Report #{selectedReport.id} - {selectedReport.location}
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1">Date: {selectedReport.date}</Typography>
                  <Typography variant="subtitle1">Region: {selectedReport.region}</Typography>
                  <Typography variant="subtitle1">Damage Type: {selectedReport.damageType}</Typography>
                  <Typography variant="subtitle1">Status: {selectedReport.status}</Typography>
                  <Typography variant="subtitle1" sx={{ mt: 2 }}>Description:</Typography>
                  <Typography paragraph>{selectedReport.description}</Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Before/After Comparison</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card>
                        <CardMedia
                          component="img"
                          height="200"
                          image={selectedReport.beforeImage}
                          alt="Before repair"
                        />
                        <CardContent>
                          <Typography variant="body2">Before Repair</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card>
                        <CardMedia
                          component="img"
                          height="200"
                          image={selectedReport.afterImage}
                          alt="After repair"
                        />
                        <CardContent>
                          <Typography variant="body2">After Repair</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </Paper>
          ) : (
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Archived Reports
              </Typography>
              <Grid container spacing={3}>
                {reports.map((report) => (
                  <Grid item xs={12} md={4} key={report.id}>
                    <Card sx={{ cursor: 'pointer' }} onClick={() => setSelectedReport(report)}>
                      <CardContent>
                        <Typography variant="h6">
                          {report.location} - {report.damageType}
                        </Typography>
                        <Typography color="textSecondary">
                          Date: {report.date}
                        </Typography>
                        <Typography color="textSecondary">
                          Status: {report.status}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          {report.description.substring(0, 100)}...
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
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Damage Type Trends (2023)
          </Typography>
          <Grid container spacing={4}>
            <Grid item xs={12} md={8}>
              <Box sx={{ height: 400, width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={trendData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="waterDamage" name="Water Damage" stackId="a" fill="#8884d8" />
                    <Bar dataKey="structural" name="Structural" stackId="a" fill="#82ca9d" />
                    <Bar dataKey="electrical" name="Electrical" stackId="a" fill="#ffc658" />
                    <Bar dataKey="other" name="Other" stackId="a" fill="#ff8042" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle1" gutterBottom>
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
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {regionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Container>
  );
};

export default Historical;
