import React from 'react';
import { useTheme } from '@mui/material/styles';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  alpha,
  Fade,
  Stack,
} from '@mui/material';
import { motion } from 'framer-motion';
import { ArrowForward } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Hero = ({ isAuthenticated }) => {
  const theme = useTheme();
  const navigate = useNavigate();

  const handleGetStarted = () => {
    if (isAuthenticated) {
      navigate('/dashboard');
    } else {
      navigate('/login');
    }
  };

  return (
    <Box
      sx={{
        position: 'relative',
        pt: { xs: 12, sm: 15, md: 18 },
        pb: { xs: 8, sm: 10, md: 12 },
        overflow: 'hidden',
      }}
    >
      {/* Background gradient */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '100%',
          background: `radial-gradient(80% 80% at 50% -10%, ${alpha(theme.palette.primary.main, 0.15)}, transparent)`,
          zIndex: -1,
        }}
      />
      
      <Container maxWidth="lg">
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Fade in={true} timeout={1000}>
              <Box>
                <Box 
                  component={motion.div}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7 }}
                >
                  <Stack
                    direction="row"
                    alignItems="center"
                    spacing={1}
                    sx={{
                      mb: 2,
                      p: 1.2,
                      px: 2,
                      border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                      borderRadius: '100px',
                      width: 'fit-content',
                      bgcolor: alpha(theme.palette.primary.main, 0.08),
                    }}
                  >
                    <Box
                      sx={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        bgcolor: 'primary.main',
                      }}
                    />
                    <Typography
                      variant="subtitle2"
                      fontWeight={600}
                      color="primary"
                      fontSize="0.875rem"
                    >
                      AI-Powered Road Management System
                    </Typography>
                  </Stack>
                </Box>

                <Typography
                  component={motion.h1}
                  variant="h1"
                  sx={{
                    fontSize: { xs: '2.5rem', md: '3.5rem', lg: '3.75rem' },
                    fontWeight: 800,
                    lineHeight: 1.2,
                    mb: 2,
                    letterSpacing: '-0.02em',
                  }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7, delay: 0.1 }}
                >
                  Transform Road Maintenance with{' '}
                  <Box
                    component="span"
                    sx={{
                      position: 'relative',
                      display: 'inline-block',
                      background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        bottom: '0.125rem',
                        left: 0,
                        right: 0,
                        height: '0.3rem',
                        background: `linear-gradient(90deg, ${alpha(theme.palette.primary.main, 0.4)}, ${alpha(theme.palette.primary.dark, 0.4)})`,
                        borderRadius: '100px',
                        zIndex: -1,
                      },
                    }}
                  >
                    AI-Powered
                  </Box>{' '}
                  Intelligence
                </Typography>

                <Box
                  component={motion.div}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7, delay: 0.2 }}
                >
                  <Typography
                    variant="h6"
                    color="text.secondary"
                    sx={{
                      mb: 4,
                      fontWeight: 500,
                      lineHeight: 1.6,
                      maxWidth: '600px',
                    }}
                  >
                    Deploy advanced AI technology to detect, analyze, and prioritize road damages. Streamline your maintenance workflows with our integrated mobile and web platform designed for modern infrastructure management.
                  </Typography>
                </Box>

                <Box
                  component={motion.div}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7, delay: 0.3 }}
                  sx={{
                    display: 'flex',
                    flexDirection: { xs: 'column', sm: 'row' },
                    gap: 2,
                  }}
                >
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleGetStarted}
                    endIcon={<ArrowForward />}
                    sx={{
                      px: 4,
                      py: 1.5,
                      fontWeight: 600,
                      borderRadius: '12px',
                      boxShadow: `0 8px 20px ${alpha(theme.palette.primary.main, 0.3)}`,
                      background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-3px)',
                        boxShadow: `0 10px 25px ${alpha(theme.palette.primary.main, 0.5)}`,
                      },
                    }}
                  >
                    {isAuthenticated ? 'Go to Dashboard' : 'Get Started Free'}
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => navigate('/request-access')}
                    sx={{
                      px: 4,
                      py: 1.5,
                      fontWeight: 600,
                      borderRadius: '12px',
                      borderColor: alpha(theme.palette.primary.main, 0.3),
                      color: 'text.primary',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        borderColor: theme.palette.primary.main,
                        bgcolor: alpha(theme.palette.primary.main, 0.05),
                      },
                    }}
                  >
                    Request Access
                  </Button>
                </Box>
              </Box>
            </Fade>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box
              component={motion.div}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              sx={{
                position: 'relative',
                height: { xs: '300px', md: '500px' },
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Box
                component="img"
                src="/assets/images/dashboard.png" 
                alt="SafeStreets Dashboard"
                sx={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain',
                  borderRadius: '24px',
                  boxShadow: `0 16px 50px ${alpha(theme.palette.common.black, 0.2)}`,
                }}
              />
              
              {/* Optional floating elements */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: { xs: -20, md: 40 },
                  right: { xs: 0, md: -40 },
                  p: 2,
                  borderRadius: '16px',
                  bgcolor: 'background.paper',
                  boxShadow: `0 10px 30px ${alpha(theme.palette.common.black, 0.1)}`,
                  width: { xs: '180px', md: '220px' },
                  zIndex: 2,
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
                component={motion.div}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.8 }}
              >
                <Typography variant="subtitle2" fontWeight={700} mb={1}>
                  AI Detection Accuracy
                </Typography>
                <Typography variant="h4" fontWeight={800} color="primary.main" mb={1}>
                  94.7%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Our Vision Transformer model outperforms traditional methods
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Hero;
