import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Paper,
  Grid
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { api } from '../utils/api';
import { TOKEN_KEY } from '../config/constants';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const AiAnalysis = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [reports, setReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    if (newValue === 1) {
      fetchReports();
    }
  };

  const fetchReports = async () => {
    setReportsLoading(true);
    try {
      const data = await api.get('/images/reports');
      setReports(data.reports);
    } catch (err) {
      setError(err.message);
    } finally {
      setReportsLoading(false);
    }
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    try {
      console.log('Processing image:', {
        name: selectedImage.name,
        type: selectedImage.type,
        size: selectedImage.size
      });

      // Convert image to base64
      const base64Promise = new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            // Send only the base64 data without the prefix
            const base64String = reader.result.split(',')[1];
            resolve(base64String);
          } catch (err) {
            console.error('Error processing FileReader result:', err);
            reject(err);
          }
        };
        reader.onerror = (err) => {
          console.error('FileReader error:', err);
          reject(err);
        };
        reader.readAsDataURL(selectedImage);
      });

      const base64Image = await base64Promise;
      console.log('Base64 conversion successful, length:', base64Image.length);

      console.log('Making request to server...');
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:5030'}/api/images/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem(TOKEN_KEY)}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: selectedImage.name,
          email: 'admin@example.com',
          image: base64Image,
          contentType: selectedImage.type
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Server response error:', errorData);
        throw new Error(errorData.message || errorData.error || 'Failed to analyze image');
      }

      const data = await response.json();
      console.log('Server response:', data);
      setPrediction(data);
      
      // Refresh reports list if we're successful
      if (data.reportId) {
        fetchReports();
      }
    } catch (err) {
      console.error('Upload error details:', err);
      setError(err.message || 'Failed to upload image');
    } finally {
      setLoading(false);
    }
  };

  const renderReports = () => {
    if (reportsLoading) {
      return <CircularProgress />;
    }

    if (!reports.length) {
      return (
        <Alert severity="info">
          No reports found
        </Alert>
      );
    }

    return (
      <Grid container spacing={3}>
        {reports.map((report) => (
          <Grid item xs={12} md={6} key={report.id}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Damage Report
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Uploaded by: {report.userEmail}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Date: {new Date(report.createdAt).toLocaleDateString()}
              </Typography>
              <Typography variant="body1" gutterBottom>
                Damage Type: {report.damageType}
              </Typography>
              <Typography variant="body1" gutterBottom>
                Severity: {report.severity}
              </Typography>
              <Typography variant="body1" gutterBottom>
                Priority: {report.priority}
              </Typography>
              {report.annotatedImage && (
                <Box sx={{ mt: 2 }}>
                  <img
                    src={report.annotatedImage}
                    alt="Annotated damage"
                    style={{
                      width: '100%',
                      maxHeight: 200,
                      objectFit: 'contain',
                      borderRadius: 4,
                      border: '1px solid rgba(0,0,0,0.1)'
                    }}
                  />
                </Box>
              )}
            </Paper>
          </Grid>
        ))}
      </Grid>
    );
  };

  return (
    <Box sx={{ p: 3 }}>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Upload Image" />
          <Tab label="View Reports" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <Button
                component="label"
                variant="contained"
                startIcon={<CloudUploadIcon />}
                disabled={loading}
              >
                Upload Image
                <VisuallyHiddenInput
                  type="file"
                  accept="image/jpeg,image/png"
                  onChange={handleImageSelect}
                />
              </Button>

              {previewUrl && (
                <Box sx={{ mt: 2, width: '100%', maxHeight: 300, overflow: 'hidden' }}>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                </Box>
              )}

              {selectedImage && (
                <Button
                  variant="contained"
                  color="primary"
                  onClick={analyzeImage}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Analyze Image'}
                </Button>
              )}

              {error && (
                <Alert severity="error" sx={{ width: '100%', mt: 2 }}>
                  {error}
                </Alert>
              )}

              {prediction && (
                <Box sx={{ width: '100%' }}>
                  <Alert severity="success">
                    <Typography variant="h6">Analysis Results:</Typography>
                    <Typography variant="body1" sx={{ mt: 1 }}>
                      Damage Type: {prediction.prediction.damageType}
                    </Typography>
                    <Typography variant="body1">
                      Severity: {prediction.prediction.severity}
                    </Typography>
                    <Typography variant="body1">
                      Priority: {prediction.prediction.priority}
                    </Typography>
                  </Alert>
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {renderReports()}
      </TabPanel>
    </Box>
  );
};

export default AiAnalysis;