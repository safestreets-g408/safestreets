import React from 'react';
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
} from '@mui/material';
import {
  LinkedIn,
  GitHub,
  Email,
} from '@mui/icons-material';

const Footer = () => {
  const theme = useTheme();
  
  const currentYear = new Date().getFullYear();
  
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
              {['Features', 'Benefits', 'Pricing', 'Demo', 'Documentation'].map((item) => (
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
              sx={{
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
              }}
            >
              <Box
                component="input"
                type="email"
                placeholder="Enter your email"
                sx={{
                  width: '100%',
                  p: 1.5,
                  pl: 2,
                  pr: '110px',
                  borderRadius: '10px',
                  border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                  bgcolor: 'background.paper',
                  fontSize: '0.875rem',
                  transition: 'all 0.2s',
                  '&:focus': {
                    outline: 'none',
                    borderColor: theme.palette.primary.main,
                    boxShadow: `0 0 0 3px ${alpha(theme.palette.primary.main, 0.2)}`,
                  },
                }}
              />
              <Box
                component="button"
                type="submit"
                sx={{
                  position: 'absolute',
                  right: 5,
                  py: 0.9,
                  px: 2,
                  borderRadius: '8px',
                  bgcolor: 'primary.main',
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '0.8125rem',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  '&:hover': {
                    bgcolor: 'primary.dark',
                  },
                }}
              >
                Subscribe
              </Box>
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
    </Box>
  );
};

export default Footer;
