import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Toolbar,
  Typography,
  Avatar,
  Divider,
  useTheme,
  alpha,
  useMediaQuery,
  Button
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ReportIcon from '@mui/icons-material/Report';
import MapIcon from '@mui/icons-material/Map';
import AnalyticsIcon from '@mui/icons-material/BarChart';
import RepairIcon from '@mui/icons-material/Build';
import HistoryIcon from '@mui/icons-material/History';
import LogoutIcon from '@mui/icons-material/ExitToApp';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { DRAWER_WIDTH } from '../../config/constants';

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Damage Reports', icon: <ReportIcon />, path: '/reports' },
  { text: 'Map View', icon: <MapIcon />, path: '/map' },
  { text: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
  { text: 'Repair Management', icon: <RepairIcon />, path: '/repairs' },
  { text: 'Historical Analysis', icon: <HistoryIcon />, path: '/historical' },
  { text: 'AI Analysis', icon: <AutoAwesomeIcon />, path: '/ai-analysis' }
];

const Sidebar = ({ mobileOpen, onDrawerToggle, onLogout }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const handleNav = (path) => {
    navigate(path);
    if (isMobile) {
      onDrawerToggle();
    }
  };

  const handleLogout = () => {
    onLogout();
    navigate('/login');
  }

  const drawer = (
    <Box sx={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: theme.palette.background.paper,
      borderRight: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
      transition: theme.transitions.create(['width', 'background'], {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.enteringScreen,
      }),
      position: 'relative',
      overflow: 'hidden',
      boxShadow: isMobile ? theme.shadows[10] : 'none',
      '&:after': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'linear-gradient(180deg, rgba(51, 102, 255, 0.02) 0%, rgba(16, 185, 129, 0.02) 100%)',
        pointerEvents: 'none',
        zIndex: 0,
      },
    }}>
      <Toolbar sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 3,
        minHeight: { xs: 70, sm: 70 },
        position: 'relative',
        zIndex: 1,
      }}>
        <Avatar sx={{
          background: `linear-gradient(135deg, ${theme.palette.primary.main} 20%, ${theme.palette.secondary.main} 80%)`,
          width: 42,
          height: 42,
          fontSize: '1.2rem',
          fontWeight: 700,
          boxShadow: `0 4px 14px ${alpha(theme.palette.primary.main, 0.25)}`,
          border: `2px solid ${theme.palette.background.paper}`,
        }}>
          SS
        </Avatar>
        <Typography variant="h6" noWrap sx={{
          fontWeight: 800,
          letterSpacing: '-0.5px',
          background: `linear-gradient(135deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          position: 'relative',
          '&:after': {
            content: '""',
            position: 'absolute',
            bottom: -4,
            left: 0,
            width: '40%',
            height: 2,
            background: `linear-gradient(90deg, ${theme.palette.primary.main}, transparent)`,
            borderRadius: '2px',
          }
        }}>
          SafeStreets
        </Typography>
      </Toolbar>

      <Divider sx={{ mx: 2, opacity: 0.4 }} />

      <List sx={{ flex: 1, px: 2, py: 2, position: 'relative', zIndex: 1 }}>
        <Typography 
          variant="overline" 
          sx={{
            color: alpha(theme.palette.text.secondary, 0.7),
            fontWeight: 600,
            fontSize: '0.7rem',
            px: 2,
            mb: 1,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}
        >
          Main Navigation
        </Typography>
        
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 1.2 }}>
            <ListItemButton
              onClick={() => handleNav(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2,
                minHeight: 48,
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.2s ease',
                pl: 2,
                py: 0.8,
                '&:before': {
                  content: '""',
                  position: 'absolute',
                  left: 0,
                  top: location.pathname === item.path ? '20%' : '50%',
                  height: location.pathname === item.path ? '60%' : 0,
                  width: 3,
                  backgroundColor: theme.palette.primary.main,
                  borderRadius: 4,
                  transition: 'all 0.2s ease-in-out',
                },
                '&:hover:before': {
                  height: '60%',
                  top: '20%',
                },
                '&.Mui-selected': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.08),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.12),
                  },
                  '& .MuiListItemIcon-root': {
                    color: theme.palette.primary.main,
                    '& svg': {
                      transform: 'scale(1.1)',
                      filter: `drop-shadow(0 4px 8px ${alpha(theme.palette.primary.main, 0.4)})`,
                    }
                  },
                },
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.06),
                  transform: 'translateX(4px)',
                  '& .MuiListItemIcon-root svg': {
                    transform: 'scale(1.1)',
                  }
                }
              }}
            >
              <ListItemIcon sx={{
                minWidth: 40,
                color: location.pathname === item.path 
                  ? theme.palette.primary.main 
                  : alpha(theme.palette.text.primary, 0.7),
                '& svg': {
                  transition: 'all 0.2s ease-in-out',
                  fontSize: '1.3rem',
                }
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.95rem',
                  fontWeight: location.pathname === item.path ? 700 : 500,
                  color: location.pathname === item.path ? theme.palette.text.primary : undefined,
                  letterSpacing: '0.01em',
                }}
              />
              {location.pathname === item.path && (
                <Box sx={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  backgroundColor: theme.palette.primary.main,
                  mr: 1,
                  boxShadow: `0 0 0 3px ${alpha(theme.palette.primary.main, 0.2)}`,
                }} />
              )}
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ mx: 2, mb: 2, opacity: 0.4 }} />

      <Box sx={{ px: 2, pb: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Button
          variant="outlined"
          color="error"
          onClick={handleLogout}
          startIcon={<LogoutIcon />}
          sx={{
            width: '100%',
            borderRadius: 2,
            py: 1,
            borderWidth: '1.5px',
            fontWeight: 600,
            textTransform: 'none',
            '&:hover': {
              borderWidth: '1.5px',
              backgroundColor: alpha(theme.palette.error.main, 0.08),
              transform: 'translateY(-2px)',
            },
          }}
        >
          Sign Out
        </Button>

        <Typography variant="caption" sx={{ mt: 2, color: alpha(theme.palette.text.secondary, 0.6), textAlign: 'center' }}>
          SafeStreets Admin Portal v1.2
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ 
        width: { sm: DRAWER_WIDTH }, 
        flexShrink: { sm: 0 },
      }}
    >
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onDrawerToggle}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: 'block', sm: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: DRAWER_WIDTH,
            borderRight: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            boxShadow: theme.shadows[8]
          },
        }}
      >
        {drawer}
      </Drawer>

      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', sm: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: DRAWER_WIDTH,
            borderRight: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            boxShadow: 'none',
            background: theme.palette.background.default,
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