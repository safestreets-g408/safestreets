import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import {
  Drawer,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Typography,
  Avatar,
  useTheme,
  alpha,
  useMediaQuery,
  Button,
  Chip,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ReportIcon from '@mui/icons-material/Report';
import MapIcon from '@mui/icons-material/Map';
import AnalyticsIcon from '@mui/icons-material/BarChart';
import RepairIcon from '@mui/icons-material/Build';
import HistoryIcon from '@mui/icons-material/History';
import LogoutIcon from '@mui/icons-material/ExitToApp';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import BusinessIcon from '@mui/icons-material/Business';
import RequestPageIcon from '@mui/icons-material/RequestPage';
import { DRAWER_WIDTH } from '../../config/constants';

// Super admin menu items
const superAdminMenuItems = [
  { 
    text: 'Dashboard', 
    icon: <DashboardIcon />, 
    path: '/dashboard', 
    badge: null,
  },
  {
    text: 'Tenant Management',
    icon: <BusinessIcon />,
    path: '/tenants',
    badge: null,
  },
  {
    text: 'Access Requests',
    icon: <RequestPageIcon />,
    path: '/access-requests',
    badge: 'New',
  },
  { 
    text: 'All Reports', 
    icon: <ReportIcon />, 
    path: '/reports', 
    badge: null,
  },
  { 
    text: 'AI Analysis', 
    icon: <AutoAwesomeIcon />, 
    path: '/ai-analysis', 
    badge: 'AI',
  },
  {
    text: 'AI Assistant',
    icon: <AutoAwesomeIcon />,
    path: '/ai-chat',
    badge: 'Gemini',
  }
];

// Regular admin menu items
const regularMenuItems = [
  { 
    text: 'Dashboard', 
    icon: <DashboardIcon />, 
    path: '/dashboard', 
    badge: null,
  },
  { 
    text: 'Damage Reports', 
    icon: <ReportIcon />, 
    path: '/reports', 
    badge: 12,
  },
  { 
    text: 'Map View', 
    icon: <MapIcon />, 
    path: '/map', 
    badge: null,
  },
  { 
    text: 'Analytics', 
    icon: <AnalyticsIcon />, 
    path: '/analytics', 
    badge: null,
  },
  { 
    text: 'Repair Management', 
    icon: <RepairIcon />, 
    path: '/repairs', 
    badge: 5,
  },
  { 
    text: 'Historical Analysis', 
    icon: <HistoryIcon />, 
    path: '/historical', 
    badge: null,
  },
  { 
    text: 'AI Analysis', 
    icon: <AutoAwesomeIcon />, 
    path: '/ai-analysis', 
    badge: 'AI',
  },
  {
    text: 'AI Assistant',
    icon: <AutoAwesomeIcon />,
    path: '/ai-chat',
    badge: 'Gemini',
  }
];

const Sidebar = ({ mobileOpen, onDrawerToggle, onLogout }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const {logout} = useAuth();
  
  // Get user from localStorage for role-based rendering
  const userString = localStorage.getItem('admin_data');
  const user = userString ? JSON.parse(userString) : null;
  const userRole = user?.role || 'admin';

  const handleNav = (path) => {
    navigate(path);
    if (isMobile) {
      onDrawerToggle();
    }
  };

  const handleLogout = () => {
    logout();
  };

  const drawer = (
    <Box sx={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: theme.palette.background.paper,
      position: 'relative',
    }}>
      {/* Header Section */}
      <Box sx={{
        py: 1.5,
        px: 2,
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        borderBottom: '1px solid',
        borderColor: 'divider',
        height: '60px',
      }}>
        <Avatar sx={{
          bgcolor: 'primary.main',
          width: 32,
          height: 32,
          fontSize: '0.875rem',
          fontWeight: 500,
        }}>
          SS
        </Avatar>
        <Box>
          <Typography variant="subtitle2" sx={{
            fontWeight: 500,
            fontSize: '0.875rem',
            lineHeight: 1.2,
          }}>
            SafeStreets
          </Typography>
          <Typography variant="caption" sx={{
            color: 'text.secondary',
            fontSize: '0.7rem',
            lineHeight: 1.2,
            display: 'block',
          }}>
            Admin Portal
          </Typography>
        </Box>
      </Box>

      {/* Navigation Section */}
      <Box sx={{ flex: 1, pt: 1.5, pb: 1, overflow: 'auto' }}>
        <Typography 
          variant="caption" 
          sx={{
            color: 'text.secondary',
            px: 2.5,
            mb: 1,
            fontSize: '0.7rem',
            fontWeight: 500,
            letterSpacing: '0.02em',
            display: 'block',
            opacity: 0.8,
          }}
        >
          NAVIGATION
        </Typography>
        
        <List sx={{ p: 0 }}>
          {(userRole === 'super-admin' ? superAdminMenuItems : regularMenuItems)
            .map((item) => {
              const isSelected = location.pathname === item.path;
              return (
              <ListItem key={item.text} disablePadding sx={{ mb: 0.25 }}>
                <ListItemButton
                  onClick={() => handleNav(item.path)}
                  selected={isSelected}
                  sx={{
                    minHeight: 40,
                    px: 2,
                    py: 1,
                    mx: 0.75,
                    borderRadius: 0.75,
                    transition: 'all 0.15s ease',
                    '&:before': {
                      content: '""',
                      position: 'absolute',
                      left: 0,
                      top: isSelected ? '25%' : '50%',
                      height: isSelected ? '50%' : 0,
                      width: 2,
                      bgcolor: 'primary.main',
                      borderTopRightRadius: 2,
                      borderBottomRightRadius: 2,
                      transition: 'all 0.15s ease',
                    },
                    '&.Mui-selected': {
                      bgcolor: alpha(theme.palette.primary.main, 0.06),
                      '&:hover': {
                        bgcolor: alpha(theme.palette.primary.main, 0.08),
                      },
                      '& .MuiListItemIcon-root': {
                        color: 'primary.main',
                      },
                      '& .MuiListItemText-primary': {
                        fontWeight: 500,
                        color: 'primary.main',
                      },
                    },
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.04),
                    }
                  }}
                >
                  <ListItemIcon sx={{
                    minWidth: 36,
                    color: isSelected 
                      ? 'primary.main' 
                      : 'text.secondary',
                    '& .MuiSvgIcon-root': {
                      fontSize: '1.125rem',
                    }
                  }}>
                    {item.icon}
                  </ListItemIcon>
                  
                  <ListItemText 
                    primary={item.text}
                    primaryTypographyProps={{
                      fontSize: '0.8125rem',
                      fontWeight: 400,
                      color: isSelected ? 'primary.main' : 'text.primary',
                    }}
                  />

                  {/* Badge */}
                  {item.badge && (
                    <Box sx={{ ml: 0.5 }}>
                      {typeof item.badge === 'number' ? (
                        <Chip 
                          label={item.badge} 
                          size="small"
                          sx={{
                            height: 16,
                            fontSize: '0.625rem',
                            fontWeight: 500,
                            bgcolor: 'error.main',
                            color: '#fff',
                            '& .MuiChip-label': { px: 0.75 },
                          }}
                        />
                      ) : (
                        <Chip 
                          label={item.badge}
                          size="small"
                          variant="outlined"
                          sx={{
                            height: 16,
                            fontSize: '0.625rem',
                            fontWeight: 500,
                            borderColor: 'primary.main',
                            color: 'primary.main',
                            '& .MuiChip-label': { px: 0.75 },
                          }}
                        />
                      )}
                    </Box>
                  )}
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>
      </Box>

      {/* Footer Section */}
      <Box sx={{ p: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
        <Button
          variant="text"
          onClick={handleLogout}
          startIcon={<LogoutIcon sx={{ fontSize: '1rem' }} />}
          fullWidth
          size="small"
          sx={{
            borderRadius: 0.75,
            py: 0.75,
            justifyContent: 'flex-start',
            textTransform: 'none',
            fontWeight: 400,
            fontSize: '0.8125rem',
            color: 'text.secondary',
            '&:hover': {
              color: 'error.main',
              bgcolor: alpha(theme.palette.error.main, 0.04),
            },
          }}
        >
          Sign Out
        </Button>

        <Typography variant="caption" sx={{ 
          mt: 1.5, 
          color: 'text.disabled', 
          textAlign: 'center',
          display: 'block',
          fontSize: '0.6875rem',
        }}>
          SafeStreets Portal v2.0
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ 
        width: { md: DRAWER_WIDTH }, 
        flexShrink: { md: 0 },
      }}
    >
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onDrawerToggle}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: DRAWER_WIDTH,
            borderRight: 'none',
            boxShadow: 3,
          },
        }}
      >
        {drawer}
      </Drawer>

      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: DRAWER_WIDTH,
            borderRight: '1px solid',
            borderColor: 'divider',
            boxShadow: 'none',
          },
        }}
        open
      >
        {drawer}
      </Drawer>
    </Box>
  );
};

export default Sidebar;