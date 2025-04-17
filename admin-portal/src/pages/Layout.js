import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, Box, Drawer, Toolbar, Typography, Divider, 
  List, ListItem, ListItemIcon, ListItemText, IconButton,
  useTheme, useMediaQuery, Avatar, ListItemButton,
  Badge, Tooltip, Menu, MenuItem, alpha
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ReportIcon from '@mui/icons-material/Report';
import MapIcon from '@mui/icons-material/Map';
import AnalyticsIcon from '@mui/icons-material/BarChart';
import RepairIcon from '@mui/icons-material/Build';
import HistoryIcon from '@mui/icons-material/History';
import LogoutIcon from '@mui/icons-material/ExitToApp';
import NotificationsIcon from '@mui/icons-material/Notifications';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

const drawerWidth = 280;

const Layout = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchor, setNotificationAnchor] = useState(null);
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

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationOpen = (event) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationAnchor(null);
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

  // Mock notifications
  const notifications = [
    { id: 1, message: "New critical damage report submitted", time: "10 minutes ago" },
    { id: 2, message: "Repair team assigned to DR-2023-003", time: "1 hour ago" },
    { id: 3, message: "Weekly analytics report ready", time: "3 hours ago" }
  ];

  const drawer = (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      background: `linear-gradient(to bottom, ${alpha(theme.palette.primary.main, 0.05)}, ${alpha(theme.palette.background.default, 1)} 30%)`,
    }}>
      <Toolbar sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 2,
        px: 3,
        minHeight: 70
      }}>
        <Avatar sx={{ 
          bgcolor: theme.palette.primary.main,
          width: 42,
          height: 42,
          boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
        }}>
          DC
        </Avatar>
        <Typography variant="h5" noWrap sx={{ 
          fontWeight: 700, 
          background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.primary.light} 90%)`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          Damage Control
        </Typography>
      </Toolbar>
      <Divider sx={{ mx: 2 }} />
      <List sx={{ flex: 1, px: 2, py: 2 }}>
        {menuItems.map((item) => {
          const isSelected = location.pathname === item.path;
          return (
            <ListItem 
              key={item.text} 
              disablePadding
              sx={{ mb: 1.5 }}
            >
              <ListItemButton
                onClick={() => handleNav(item.path)}
                selected={isSelected}
                sx={{
                  borderRadius: 2,
                  py: 1.2,
                  transition: 'all 0.2s ease-in-out',
                  '&.Mui-selected': {
                    backgroundColor: theme.palette.primary.main,
                    color: 'white',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                    '&:hover': {
                      backgroundColor: theme.palette.primary.dark,
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'white'
                    }
                  },
                  '&:hover': {
                    backgroundColor: isSelected ? theme.palette.primary.dark : alpha(theme.palette.primary.main, 0.1),
                    transform: 'translateY(-2px)',
                    boxShadow: isSelected ? '0 6px 12px rgba(0,0,0,0.2)' : '0 4px 8px rgba(0,0,0,0.05)',
                  }
                }}
              >
                <ListItemIcon sx={{ 
                  minWidth: 45, 
                  color: isSelected ? 'white' : theme.palette.primary.main 
                }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text}
                  primaryTypographyProps={{
                    fontSize: '0.95rem',
                    fontWeight: isSelected ? 600 : 500,
                    letterSpacing: '0.02em'
                  }}
                />
                {item.text === 'Damage Reports' && (
                  <Box
                    sx={{
                      bgcolor: theme.palette.error.main,
                      color: 'white',
                      borderRadius: '50%',
                      width: 22,
                      height: 22,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                      ml: 1
                    }}
                  >
                    3
                  </Box>
                )}
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
      <Box sx={{ p: 2, mb: 2, mx: 2, bgcolor: alpha(theme.palette.primary.light, 0.1), borderRadius: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: theme.palette.text.secondary, fontWeight: 600 }}>
          System Status
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ 
            width: 10, 
            height: 10, 
            borderRadius: '50%', 
            bgcolor: theme.palette.success.main,
            boxShadow: `0 0 0 3px ${alpha(theme.palette.success.main, 0.2)}`
          }} />
          <Typography variant="body2" sx={{ color: theme.palette.text.secondary }}>
            All systems operational
          </Typography>
        </Box>
      </Box>
      <Divider sx={{ mx: 2 }} />
      <List sx={{ px: 2, pb: 2, pt: 1 }}>
        <ListItem disablePadding>
          <ListItemButton 
            onClick={handleLogout}
            sx={{ 
              borderRadius: 2,
              py: 1.2,
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                backgroundColor: alpha(theme.palette.error.main, 0.1),
                color: theme.palette.error.main,
                transform: 'translateY(-2px)',
                '& .MuiListItemIcon-root': {
                  color: theme.palette.error.main
                }
              }
            }}
          >
            <ListItemIcon sx={{ minWidth: 45, color: theme.palette.error.main }}>
              <LogoutIcon />
            </ListItemIcon>
            <ListItemText 
              primary="Logout"
              primaryTypographyProps={{
                fontSize: '0.95rem',
                fontWeight: 500,
                letterSpacing: '0.02em'
              }}
            />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: alpha(theme.palette.background.default, 0.98) }}>
      <AppBar 
        position="fixed" 
        elevation={0}
        sx={{ 
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          backgroundColor: 'white',
          color: 'text.primary',
          borderBottom: `1px solid ${theme.palette.divider}`
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <IconButton
              color="inherit"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { sm: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600, display: { xs: 'none', sm: 'block' } }}>
              {menuItems.find(item => item.path === location.pathname)?.text || 'Dashboard'}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Help">
              <IconButton color="inherit" size="large" sx={{ borderRadius: 2 }}>
                <HelpOutlineIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Notifications">
              <IconButton 
                color="inherit" 
                size="large" 
                sx={{ borderRadius: 2 }}
                onClick={handleNotificationOpen}
              >
                <Badge badgeContent={notifications.length} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>
            
            <Menu
              anchorEl={notificationAnchor}
              open={Boolean(notificationAnchor)}
              onClose={handleNotificationClose}
              PaperProps={{
                sx: { 
                  width: 320,
                  maxHeight: 400,
                  boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                  borderRadius: 2,
                  mt: 1
                }
              }}
            >
              <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                <Typography variant="subtitle1" fontWeight={600}>Notifications</Typography>
              </Box>
              {notifications.map(notification => (
                <MenuItem key={notification.id} onClick={handleNotificationClose} sx={{ py: 1.5 }}>
                  <Box>
                    <Typography variant="body2">{notification.message}</Typography>
                    <Typography variant="caption" color="text.secondary">{notification.time}</Typography>
                  </Box>
                </MenuItem>
              ))}
              <Box sx={{ p: 1.5, borderTop: `1px solid ${theme.palette.divider}`, textAlign: 'center' }}>
                <Typography 
                  variant="body2" 
                  color="primary" 
                  sx={{ cursor: 'pointer', fontWeight: 500 }}
                >
                  View all notifications
                </Typography>
              </Box>
            </Menu>
            
            <Tooltip title="Account">
              <IconButton 
                edge="end" 
                color="inherit" 
                size="large" 
                onClick={handleProfileMenuOpen}
                sx={{ 
                  borderRadius: 2,
                  ml: 1
                }}
              >
                <AccountCircleIcon />
              </IconButton>
            </Tooltip>
            
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleProfileMenuClose}
              PaperProps={{
                sx: { 
                  width: 200,
                  boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                  borderRadius: 2,
                  mt: 1
                }
              }}
            >
              <Box sx={{ p: 2, textAlign: 'center' }}>
                <Avatar 
                  sx={{ 
                    width: 60, 
                    height: 60, 
                    margin: '0 auto 8px',
                    bgcolor: theme.palette.primary.main
                  }}
                >
                  AD
                </Avatar>
                <Typography variant="subtitle1" fontWeight={600}>Admin User</Typography>
                <Typography variant="body2" color="text.secondary">admin@example.com</Typography>
              </Box>
              <Divider />
              <MenuItem onClick={handleProfileMenuClose}>
                <ListItemIcon>
                  <AccountCircleIcon fontSize="small" />
                </ListItemIcon>
                <Typography variant="body2">My Profile</Typography>
              </MenuItem>
              <MenuItem onClick={handleProfileMenuClose}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                <Typography variant="body2">Settings</Typography>
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" color="error" />
                </ListItemIcon>
                <Typography variant="body2" color="error">Logout</Typography>
              </MenuItem>
            </Menu>
          </Box>
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
              boxShadow: '0 0 20px rgba(0,0,0,0.1)'
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
          minHeight: 'calc(100vh - 64px)',
          transition: 'all 0.3s ease-in-out',
          background: `linear-gradient(to bottom right, ${alpha(theme.palette.background.default, 0.9)}, ${alpha(theme.palette.background.paper, 0.9)})`,
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}

export default Layout;
