import React from 'react';
import { useTheme } from '@mui/material/styles';
import {
  AppBar,
  Toolbar,
  Container,
  Box,
  Typography,
  Button,
  IconButton,
  Stack,
  alpha,
  Drawer,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import ThemeToggle from '../theme/ThemeToggle';
import {
  RocketLaunch,
  Menu as MenuIcon,
  Close,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Navigation = ({ 
  isAuthenticated, 
  mobileMenuOpen, 
  setMobileMenuOpen, 
  scrollToSection,
  featuresRef, 
  benefitsRef 
}) => {
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
    <>
      {/* Modern Navigation */}
      <AppBar position="fixed" elevation={2} sx={{ 
        bgcolor: theme.palette.mode === 'dark'
          ? alpha(theme.palette.background.appBar, 0.97)
          : alpha(theme.palette.background.paper, 0.97),
        backdropFilter: 'blur(20px)',
        borderBottom: theme.palette.mode === 'dark'
          ? `1px solid ${alpha(theme.palette.divider, 0.2)}`
          : `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.08)}`,
        transition: 'all 0.3s ease'
      }}>
        <Container maxWidth="lg">
          <Toolbar sx={{ py: { xs: 1.4, md: 1.6 } }}>
            <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Box sx={{
                  background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                  borderRadius: '12px',
                  p: 0.9,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.25)}`,
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: `0 6px 15px ${alpha(theme.palette.primary.main, 0.35)}`,
                  }
                }}>
                  <RocketLaunch sx={{ color: 'white', fontSize: 24 }} />
                </Box>
                <Typography variant="h5" sx={{ 
                  fontWeight: 900, 
                  letterSpacing: -0.5,
                  display: 'flex',
                  alignItems: 'center',
                  background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  SafeStreets
                  <sup style={{
                    fontSize: '40%', 
                    verticalAlign: 'super', 
                    marginLeft: '0.5rem',
                    WebkitTextFillColor: theme.palette.primary.dark
                  }}>®</sup>
                </Typography>
              </Stack>
            </Box>
            
            <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 3, mx: 4 }}>
              <Button 
                color="inherit" 
                onClick={() => scrollToSection(featuresRef)}
                sx={{ 
                  fontWeight: 600, 
                  position: 'relative',
                  px: 1.5,
                  color: 'text.primary',
                  '&:hover': {
                    color: 'primary.main',
                    background: 'transparent'
                  },
                  '&:hover:after': {
                    width: '80%',
                    opacity: 1
                  },
                  '&:after': {
                    content: '""',
                    position: 'absolute',
                    bottom: '0.2rem',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '0%',
                    height: '2px',
                    bgcolor: 'primary.main',
                    transition: 'all 0.3s ease',
                    opacity: 0,
                  }
                }}
              >
                Features
              </Button>
              <Button 
                color="inherit" 
                onClick={() => scrollToSection(benefitsRef)}
                sx={{ 
                  fontWeight: 600, 
                  position: 'relative',
                  px: 1.5,
                  color: 'text.primary',
                  '&:hover': {
                    color: 'primary.main',
                    background: 'transparent'
                  },
                  '&:hover:after': {
                    width: '80%',
                    opacity: 1
                  },
                  '&:after': {
                    content: '""',
                    position: 'absolute',
                    bottom: '0.2rem',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '0%',
                    height: '2px',
                    bgcolor: 'primary.main',
                    transition: 'all 0.3s ease',
                    opacity: 0,
                  }
                }}
              >
                Benefits
              </Button>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {/* Theme Toggle */}
              <ThemeToggle />
              
              {/* Get Started Button */}
              <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                <Button
                  variant="contained"
                  onClick={handleGetStarted}
                  sx={{
                    px: 3,
                    py: 1.2,
                    fontWeight: 600,
                    borderRadius: '10px',
                    boxShadow: `0 4px 14px ${alpha(theme.palette.primary.main, 0.25)}`,
                    background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                    }
                  }}
                >
                  {isAuthenticated ? 'Go to Dashboard' : 'Get Started'}
                </Button>
              </Box>
            </Box>
            
            <IconButton
              color="inherit"
              sx={{ display: { md: 'none' } }}
              onClick={() => setMobileMenuOpen(true)}
            >
              <MenuIcon />
            </IconButton>
          </Toolbar>
        </Container>
      </AppBar>
      
      {/* Mobile menu */}
      <Drawer
        anchor="right"
        open={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        sx={{
          '& .MuiDrawer-paper': { 
            width: '100%',
            maxWidth: '300px',
            bgcolor: 'background.paper',
            backgroundImage: `linear-gradient(to bottom, ${alpha(theme.palette.primary.light, 0.05)}, ${alpha(theme.palette.background.paper, 0.9)})`,
            p: 2
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>SafeStreets</Typography>
          <IconButton onClick={() => setMobileMenuOpen(false)}>
            <Close />
          </IconButton>
        </Box>
        
        <List>
          <ListItem 
            button 
            onClick={() => {
              scrollToSection(featuresRef);
              setMobileMenuOpen(false);
            }}
          >
            <ListItemText primary="Features" />
          </ListItem>
          <ListItem 
            button 
            onClick={() => {
              scrollToSection(benefitsRef);
              setMobileMenuOpen(false);
            }}
          >
            <ListItemText primary="Benefits" />
          </ListItem>
          <ListItem 
            button 
            onClick={() => {
              handleGetStarted();
              setMobileMenuOpen(false);
            }}
          >
            <ListItemText 
              primary={isAuthenticated ? 'Go to Dashboard' : 'Get Started'} 
              primaryTypographyProps={{
                color: 'primary',
                fontWeight: 600
              }}
            />
          </ListItem>
        </List>
      </Drawer>
    </>
  );
};

export default Navigation;
