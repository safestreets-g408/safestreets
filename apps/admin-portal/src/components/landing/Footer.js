import React, { useState } from 'react';
import { useTheme } from '@mui/material/styles';
import {
  Box,
  Container,
  Typography,
  Grid,
  Stack,
  IconButton,
  Divider,
  alpha,
  TextField,
  Button,
  Snackbar,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  LinkedIn,
  GitHub,
  Email,
  Send as SendIcon
} from '@mui/icons-material';
import axios from 'axios';
import config from '../../config';

const Footer = () => {
  const theme = useTheme();
  const currentYear = new Date().getFullYear();
  
  // State for newsletter subscription
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  
  // Handle form submission
  const handleSubscribe = async (e) => {
    e.preventDefault();
    
    // Simple email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setSnackbar({
        open: true,
        message: 'Please enter a valid email address',
        severity: 'error'
      });
      return;
    }
    
    setLoading(true);
    
    try {
      // Get the API base URL from config
      const API_BASE_URL = config.backend.baseURL;
      
      // Send subscription request
      const response = await axios.post(`${API_BASE_URL}/api/admin/subscribe-newsletter`, { email });
      
      setSnackbar({
        open: true,
        message: response.data.message || 'Successfully subscribed to newsletter!',
        severity: 'success'
      });
      
      // Clear the form
      setEmail('');
    } catch (error) {
      console.error('Newsletter subscription error:', error);
      setSnackbar({
        open: true,
        message: error.response?.data?.message || 'Failed to subscribe. Please try again later.',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Handle snackbar close
  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };
  
  return (
    <Box
      sx={{
        py: { xs: 6, md: 8 },
        bgcolor: alpha(theme.palette.primary.main, 0.04),
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Stack spacing={2}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 900,
                    background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  SafeStreets
                </Typography>
              </Box>
              
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ lineHeight: 1.6, maxWidth: 300 }}
              >
                Advanced AI-powered road maintenance system for modern infrastructure management.
                Built with the latest technologies for maximum efficiency.
              </Typography>
              
              <Stack direction="row" spacing={1}>
                <IconButton
                  aria-label="LinkedIn"
                  sx={{
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    color: 'primary.main',
                    '&:hover': {
                      bgcolor: 'primary.main',
                      color: 'white',
                    },
                  }}
                  size="small"
                >
                  <LinkedIn fontSize="small" />
                </IconButton>
                <IconButton
                  aria-label="GitHub"
                  sx={{
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    color: 'primary.main',
                    '&:hover': {
                      bgcolor: 'primary.main',
                      color: 'white',
                    },
                  }}
                  size="small"
                >
                  <GitHub fontSize="small" />
                </IconButton>
                <IconButton
                  aria-label="Email"
                  sx={{
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    color: 'primary.main',
                    '&:hover': {
                      bgcolor: 'primary.main',
                      color: 'white',
                    },
                  }}
                  size="small"
                >
                  <Email fontSize="small" />
                </IconButton>
              </Stack>
            </Stack>
          </Grid>
          
          <Grid item xs={6} md={2}>
            <Typography
              variant="subtitle1"
              fontWeight={700}
              gutterBottom
              sx={{ mb: 2 }}
            >
              Product
            </Typography>
            
            <Stack spacing={1.5}>
              {['Features', 'Benefits', 'Pricing', 'Demo'].map((item) => (
                <Typography
                  key={item}
                  variant="body2"
                  component="a"
                  href="#"
                  sx={{
                    color: 'text.secondary',
                    textDecoration: 'none',
                    transition: 'color 0.2s',
                    '&:hover': {
                      color: 'primary.main',
                    },
                  }}
                >
                  {item}
                </Typography>
              ))}
              <Typography
                variant="body2"
                component="a"
                href="https://safestreets.gitbook.io/safestreets"
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  color: 'text.secondary',
                  textDecoration: 'none',
                  transition: 'color 0.2s',
                  '&:hover': {
                    color: 'primary.main',
                  },
                }}
              >
                Documentation
              </Typography>
            </Stack>
          </Grid>
          
          <Grid item xs={6} md={2}>
            <Typography
              variant="subtitle1"
              fontWeight={700}
              gutterBottom
              sx={{ mb: 2 }}
            >
              Company
            </Typography>
            
            <Stack spacing={1.5}>
              {['About Us', 'Careers', 'Blog', 'Press', 'Contact'].map((item) => (
                <Typography
                  key={item}
                  variant="body2"
                  component="a"
                  href="#"
                  sx={{
                    color: 'text.secondary',
                    textDecoration: 'none',
                    transition: 'color 0.2s',
                    '&:hover': {
                      color: 'primary.main',
                    },
                  }}
                >
                  {item}
                </Typography>
              ))}
            </Stack>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography
              variant="subtitle1"
              fontWeight={700}
              gutterBottom
              sx={{ mb: 2 }}
            >
              Stay Updated
            </Typography>
            
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 3, lineHeight: 1.6 }}
            >
              Subscribe to our newsletter for the latest updates on product features,
              AI advancements, and road maintenance best practices.
            </Typography>
            
            <Box
              component="form"
              onSubmit={handleSubscribe}
              sx={{
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
              }}
            >
              <TextField
                fullWidth
                placeholder="Enter your email"
                variant="outlined"
                type="email"
                size="small"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={loading}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: '8px',
                    pr: '110px',
                  },
                }}
              />
              <Button
                type="submit"
                variant="contained"
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                sx={{
                  position: 'absolute',
                  right: 5,
                  height: 'calc(100% - 10px)',
                  borderRadius: '6px',
                  fontWeight: 600,
                  textTransform: 'none',
                  whiteSpace: 'nowrap',
                }}
              >
                {loading ? 'Subscribing...' : 'Subscribe'}
              </Button>
            </Box>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 4, opacity: 0.2 }} />
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography
              variant="caption"
              color="text.secondary"
            >
              © {currentYear} SafeStreets AI. All rights reserved.
            </Typography>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Stack
              direction="row"
              spacing={3} 
              justifyContent={{ xs: 'flex-start', md: 'flex-end' }}
              sx={{ flexWrap: 'wrap' }}
            >
              <Typography variant="caption" color="text.secondary" component="a" href="#" sx={{ textDecoration: 'none' }}>
                Privacy Policy
              </Typography>
              <Typography variant="caption" color="text.secondary" component="a" href="#" sx={{ textDecoration: 'none' }}>
                Terms of Service
              </Typography>
              <Typography variant="caption" color="text.secondary" component="a" href="#" sx={{ textDecoration: 'none' }}>
                Security
              </Typography>
              <Typography variant="caption" color="text.secondary" component="a" href="#" sx={{ textDecoration: 'none' }}>
                Compliance
              </Typography>
            </Stack>
          </Grid>
        </Grid>
      </Container>
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Footer;
