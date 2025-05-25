import React, { useState,  } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Toolbar,
  useTheme,
  alpha
} from '@mui/material';
//import { useAuth } from '../../hooks/useAuth';
import { DRAWER_WIDTH } from '../../config/constants';
import Sidebar from './Sidebar';
import Header from './Header';

const MainLayout = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const navigate = useNavigate();
  // Uncomment if using authentication
  //const { logout } = useAuth();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleLogout = () => {
    navigate('/')
  }

  return (
    <Box 
      sx={{ 
        display: 'flex',
        minHeight: '100vh',
        bgcolor: theme.palette.background.default
      }}
    >
      <Header onDrawerToggle={handleDrawerToggle} />
      <Sidebar 
        mobileOpen={mobileOpen} 
        onDrawerToggle={handleDrawerToggle}
        onLogout={handleLogout}
      />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, sm: 3 },
          width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
          minHeight: '100vh',
          bgcolor: theme.palette.background.default,
          borderRadius: { xs: 0, sm: '12px 0 0 0' },
          position: 'relative',
          '&:before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '200px',
            background: `linear-gradient(180deg, ${alpha(theme.palette.primary.main, 0.03)} 0%, transparent 100%)`,
            borderRadius: 'inherit',
            pointerEvents: 'none',
            zIndex: 0
          }
        }}
      >
        <Toolbar />
        <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};

export default MainLayout;