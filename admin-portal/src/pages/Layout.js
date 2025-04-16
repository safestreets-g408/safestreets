import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, Box, Drawer, Toolbar, Typography, Divider, 
  List, ListItem, ListItemIcon, ListItemText, IconButton,
  useTheme, useMediaQuery, Avatar, ListItemButton
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ReportIcon from '@mui/icons-material/Report';
import MapIcon from '@mui/icons-material/Map';
import AnalyticsIcon from '@mui/icons-material/BarChart';
import RepairIcon from '@mui/icons-material/Build';
import HistoryIcon from '@mui/icons-material/History';
import LogoutIcon from '@mui/icons-material/ExitToApp';

const drawerWidth = 260;

const Layout = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNav = (path) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const handleLogout = async () => {
    try {
      localStorage.removeItem('token');
      navigate('/login', { replace: true });
    } catch (error) {
      console.error("Failed to log out", error);
    }
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Damage Reports', icon: <ReportIcon />, path: '/reports' },
    { text: 'Map View', icon: <MapIcon />, path: '/map' },
    { text: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
    { text: 'Repair Management', icon: <RepairIcon />, path: '/repairs' },
    { text: 'Historical Analysis', icon: <HistoryIcon />, path: '/historical' },
  ];

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 2,
        px: 2,
        minHeight: 64
      }}>
        <Avatar sx={{ 
          bgcolor: 'primary.main',
          width: 40,
          height: 40
        }}>
          DC
        </Avatar>
        <Typography variant="h6" noWrap sx={{ fontWeight: 600 }}>
          Damage Control
        </Typography>
      </Toolbar>
      <Divider />
      <List sx={{ flex: 1, px: 2 }}>
        {menuItems.map((item) => (
          <ListItem 
            key={item.text} 
            disablePadding
            sx={{ mb: 1 }}
          >
            <ListItemButton
              onClick={() => handleNav(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'white'
                  }
                }
              }}
            >
              <ListItemIcon sx={{ minWidth: 40, color: location.pathname === item.path ? 'white' : 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: location.pathname === item.path ? 600 : 400
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List sx={{ px: 2, pb: 2 }}>
        <ListItem disablePadding>
          <ListItemButton 
            onClick={handleLogout}
            sx={{ 
              borderRadius: 1,
              '&:hover': {
                backgroundColor: 'error.light',
                color: 'white',
                '& .MuiListItemIcon-root': {
                  color: 'white'
                }
              }
            }}
          >
            <ListItemIcon sx={{ minWidth: 40 }}><LogoutIcon /></ListItemIcon>
            <ListItemText 
              primary="Logout"
              primaryTypographyProps={{
                fontSize: '0.9rem'
              }}
            />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar 
        position="fixed" 
        sx={{ 
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          backgroundColor: 'white',
          color: 'text.primary',
          boxShadow: 1
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600 }}>
            Damage Control Admin Dashboard
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              backgroundColor: 'background.default',
            },
          }}
        >
          {drawer}
        </Drawer>
        
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              backgroundColor: 'background.default',
              borderRight: '1px solid',
              borderColor: 'divider'
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      
      <Box
        component="main"
        sx={{ 
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: '64px',
          minHeight: 'calc(100vh - 64px)'
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}

export default Layout;
