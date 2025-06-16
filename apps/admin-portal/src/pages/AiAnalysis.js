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
  Grid,
  Divider,
  Container,
  Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import AssessmentIcon from '@mui/icons-material/Assessment';
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

  // Reports are now rendered directly in the JSX of TabPanel with index 1

  return (
    <Box sx={{ p: 3 }}>

      <Paper 
        elevation={2} 
        sx={{ 
          borderRadius: 2,
          overflow: 'hidden',
          mb: 4
        }}
      >
        <Box sx={{ px: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ 
              '& .MuiTab-root': {
                minHeight: '64px',
                textTransform: 'none',
                fontSize: '1rem',
                fontWeight: 500,
              }
            }}
          >
            <Tab 
              label="Upload Image" 
              icon={<CloudUploadIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="View Reports" 
              icon={<AssessmentIcon />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Card sx={{ boxShadow: 'none' }}>
            <CardContent sx={{ px: { xs: 2, sm: 4 }, py: 4 }}>
              <Grid container spacing={4}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    gap: 3,
                    height: '100%',
                    justifyContent: 'center'
                  }}>
                    <Paper 
                      elevation={0} 
                      sx={{ 
                        width: '100%', 
                        height: 300, 
                        borderRadius: 2,
                        border: '2px dashed #ccc',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: '#f8f9fa',
                        position: 'relative',
                        overflow: 'hidden'
                      }}
                    >
                      {previewUrl ? (
                        <img
                          src={previewUrl}
                          alt="Preview"
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            padding: '16px'
                          }}
                        />
                      ) : (
                        <>
                          <ImageIcon sx={{ fontSize: 60, color: 'text.secondary', opacity: 0.5, mb: 2 }} />
                          <Typography variant="body1" color="text.secondary">
                            No image selected
                          </Typography>
                        </>
                      )}
                    </Paper>
                    
                    <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'center' }}>
                      <Button
                        component="label"
                        variant="contained"
                        startIcon={<CloudUploadIcon />}
                        disabled={loading}
                        size="large"
                        sx={{ px: 3, py: 1, borderRadius: 2 }}
                      >
                        Select Image
                        <VisuallyHiddenInput
                          type="file"
                          accept="image/jpeg,image/png"
                          onChange={handleImageSelect}
                        />
                      </Button>

                      {selectedImage && (
                        <Button
                          variant="contained"
                          color="secondary"
                          onClick={analyzeImage}
                          disabled={loading}
                          size="large"
                          sx={{ px: 3, py: 1, borderRadius: 2 }}
                        >
                          {loading ? (
                            <>
                              <CircularProgress size={20} sx={{ mr: 1, color: 'white' }} />
                              Analyzing...
                            </>
                          ) : 'Analyze Image'}
                        </Button>
                      )}
                    </Box>
                  </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper 
                    elevation={1} 
                    sx={{ 
                      p: 3, 
                      height: '100%', 
                      borderRadius: 2,
                      backgroundColor: '#f8f8ff'
                    }}
                  >
                    <Typography variant="h6" gutterBottom sx={{ mb: 2, fontWeight: 500 }}>
                      Analysis Results
                    </Typography>
                    
                    {error && (
                      <Alert severity="error" sx={{ width: '100%', mb: 2 }}>
                        {error}
                      </Alert>
                    )}

                    {loading && (
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center',
                        justifyContent: 'center', 
                        my: 6,
                        gap: 2
                      }}>
                        <CircularProgress />
                        <Typography variant="body2" color="text.secondary">
                          Processing image with AI...
                        </Typography>
                      </Box>
                    )}

                    {!loading && !error && prediction ? (
                      <Box sx={{ pt: 2 }}>
                        <Grid container spacing={2}>
                          <Grid item xs={12}>
                            <Paper sx={{ p: 2, borderRadius: 2, bgcolor: '#e3f2fd', mb: 2 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="subtitle1" fontWeight="medium">Damage Type</Typography>
                                <Chip 
                                  label={prediction.prediction.damageType} 
                                  color="primary" 
                                  variant="filled" 
                                  size="medium"
                                />
                              </Box>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} sm={6}>
                            <Paper sx={{ p: 2, borderRadius: 2, bgcolor: '#fff8e1', height: '100%' }}>
                              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                                <Typography variant="subtitle2" color="text.secondary">Severity</Typography>
                                <Chip 
                                  label={prediction.prediction.severity} 
                                  color={prediction.prediction.severity && String(prediction.prediction.severity).toLowerCase() === 'high' ? 'error' : 'warning'} 
                                  sx={{ fontWeight: 'bold' }}
                                />
                              </Box>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} sm={6}>
                            <Paper sx={{ p: 2, borderRadius: 2, bgcolor: '#f9fbe7', height: '100%' }}>
                              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                                <Typography variant="subtitle2" color="text.secondary">Priority</Typography>
                                <Chip 
                                  label={prediction.prediction.priority} 
                                  color={prediction.prediction.priority && String(prediction.prediction.priority).toLowerCase() === 'high' ? 'error' : 'info'} 
                                  sx={{ fontWeight: 'bold' }}
                                />
                              </Box>
                            </Paper>
                          </Grid>
                        </Grid>
                      </Box>
                    ) : !loading && !error && (
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center',
                        justifyContent: 'center', 
                        my: 6,
                        gap: 2
                      }}>
                        <Typography variant="body1" color="text.secondary" align="center">
                          Select and analyze an image to view AI assessment results
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ px: { xs: 1, sm: 3 }, py: 3 }}>
            {reportsLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : reports.length === 0 ? (
              <Paper sx={{ p: 4, borderRadius: 2, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                <AssessmentIcon sx={{ fontSize: 60, color: 'text.secondary', opacity: 0.5, mb: 2 }} />
                <Typography variant="h6" gutterBottom>No Reports Available</Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload and analyze images to create damage reports
                </Typography>
              </Paper>
            ) : (
              <Grid container spacing={3}>
                {reports.map((report) => (
                  <Grid item xs={12} md={6} lg={4} key={report.id}>
                    <Paper 
                      elevation={2} 
                      sx={{ 
                        p: 0,
                        borderRadius: 2,
                        overflow: 'hidden',
                        transition: 'transform 0.2s, box-shadow 0.2s',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 4
                        }
                      }}
                    >
                      {report.annotatedImage && (
                        <Box sx={{ 
                          width: '100%',
                          height: 200, 
                          overflow: 'hidden',
                          backgroundColor: '#f0f0f0'
                        }}>
                          <img
                            src={report.annotatedImage}
                            alt="Annotated damage"
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover'
                            }}
                          />
                        </Box>
                      )}
                      
                      <Box sx={{ p: 2.5 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Typography variant="h6" sx={{ fontWeight: 500 }}>
                            Damage Report
                          </Typography>
                          <Chip 
                            size="small" 
                            label={report.severity} 
                            color={report.severity === 'High' ? 'error' : report.severity === 'Medium' ? 'warning' : 'success'} 
                          />
                        </Box>
                        
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" color="text.secondary">Type</Typography>
                            <Typography variant="body2" fontWeight="medium">{report.damageType}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" color="text.secondary">Priority</Typography>
                            <Typography variant="body2" fontWeight="medium">{report.priority}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2" color="text.secondary">Date</Typography>
                            <Typography variant="body2">{new Date(report.createdAt).toLocaleDateString()}</Typography>
                          </Box>
                        </Box>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            )}
          </Box>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default AiAnalysis;