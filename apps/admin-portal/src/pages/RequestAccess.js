import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  Box,
  useMediaQuery,

  IconButton,
  Alert,
  Snackbar,
  MenuItem,
  Grid,
  CircularProgress
} from '@mui/material';
import SecurityIcon from '@mui/icons-material/Security';
import HomeIcon from '@mui/icons-material/Home';
import axios from 'axios';
import { API_BASE_URL } from '../config/constants';

const regions = [
  'North America',
  'South America',
  'Europe',
  'Africa',
  'Asia',
  'Oceania',
  'Middle East'
];

const RequestAccess = () => {
  const [formData, setFormData] = useState({
    organizationName: '',
    contactName: '',
    email: '',
    phone: '',
    region: '',
    reason: ''
  });
  
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showAlert, setShowAlert] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const isMobile = useMediaQuery('(max-width:768px)');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Basic validation
    for (const key in formData) {
      if (!formData[key]) {
        setError(`Please fill in all fields`);
        setShowAlert(true);
        return;
      }
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError('Please enter a valid email address');
      setShowAlert(true);
      return;
    }

    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/access-requests`, formData);
      
      setSuccess('Your request has been submitted successfully! We will review it and get back to you soon.');
      setShowAlert(true);
      setFormData({
        organizationName: '',
        contactName: '',
        email: '',
        phone: '',
        region: '',
        reason: ''
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to submit your request. Please try again later.');
      setShowAlert(true);
    } finally {
      setLoading(false);
    }
  };

  const handleCloseAlert = () => {
    setShowAlert(false);
  };

  return (
    <Container maxWidth="lg" sx={{ 
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      py: 4
    }}>
      <IconButton 
        onClick={() => navigate('/landing')}
        sx={{ 
          position: 'absolute',
          top: 20,
          left: 20,
          color: 'primary.main'
        }}
      >
        <HomeIcon sx={{ fontSize: 32 }} />
      </IconButton>
      <Paper
        elevation={0}
        sx={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          borderRadius: '8px',
          overflow: 'hidden',
          minHeight: '600px',
          width: '100%',
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}
      >
        {/* Left Side - Information */}
        <Box
          sx={{
            flex: '1 1 40%',
            p: 6,
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
            <SecurityIcon sx={{ fontSize: 48, mr: 2 }} />
            <Typography variant="h3" sx={{ fontWeight: 600 }}>
              SafeStreets
            </Typography>
          </Box>
          
          <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
            Request Access
          </Typography>
          
          <Typography variant="body1" sx={{ mb: 4, opacity: 0.9 }}>
            SafeStreets provides road management and maintenance solutions for government 
            agencies and municipalities. Fill out this form to request access to our platform.
          </Typography>

          <Box sx={{ mb: 4 }}>
            <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
              ✓ AI-powered damage detection and analysis
            </Typography>
            <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
              ✓ Mobile app for field workers
            </Typography>
            <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
              ✓ Comprehensive admin dashboard
            </Typography>
            <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
              ✓ Advanced analytics and reporting
            </Typography>
          </Box>
        </Box>

        {/* Right Side - Request Form */}
        <Box
          sx={{
            flex: '1 1 60%',
            p: 6,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
        >
          <Typography variant="h4" sx={{ mb: 4, fontWeight: 600, color: 'text.primary' }}>
            Authority Access Request
          </Typography>

          <Box component="form" onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  required
                  fullWidth
                  id="organizationName"
                  label="Organization Name"
                  name="organizationName"
                  value={formData.organizationName}
                  onChange={handleChange}
                  variant="outlined"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
                    }
                  }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  required
                  fullWidth
                  id="contactName"
                  label="Contact Person Name"
                  name="contactName"
                  value={formData.contactName}
                  onChange={handleChange}
                  variant="outlined"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
                    }
                  }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  required
                  fullWidth
                  id="email"
                  label="Email Address"
                  name="email"
                  autoComplete="email"
                  value={formData.email}
                  onChange={handleChange}
                  variant="outlined"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
                    }
                  }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  required
                  fullWidth
                  id="phone"
                  label="Phone Number"
                  name="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  variant="outlined"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
                    }
                  }}
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  required
                  fullWidth
                  id="region"
                  select
                  label="Region"
                  name="region"
                  value={formData.region}
                  onChange={handleChange}
                  variant="outlined"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
                    }
                  }}
                >
                  {regions.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
                </TextField>
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  required
                  fullWidth
                  id="reason"
                  label="Reason for Access"
                  name="reason"
                  value={formData.reason}
                  onChange={handleChange}
                  variant="outlined"
                  multiline
                  rows={4}
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
                    }
                  }}
                />
              </Grid>
            </Grid>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{
                mt: 4,
                py: 1.8,
                borderRadius: 1,
                fontSize: '1rem',
                textTransform: 'none',
                fontWeight: 600,
                bgcolor: 'primary.main',
                '&:hover': {
                  bgcolor: 'primary.dark',
                },
                '&:disabled': {
                  bgcolor: 'action.disabledBackground',
                }
              }}
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                'Submit Request'
              )}
            </Button>
          </Box>
        </Box>
      </Paper>

      <Snackbar 
        open={showAlert} 
        autoHideDuration={6000} 
        onClose={handleCloseAlert}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseAlert} 
          severity={error ? 'error' : 'success'} 
          sx={{ width: '100%' }}
        >
          {error || success}
        </Alert>
      </Snackbar>

    </Container>
  );
};

export default RequestAccess;
