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
  useMediaQuery
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
import { te } from 'date-fns/locale';

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
      background: alpha(theme.palette.background.paper, 0.8),
      backdropFilter: 'blur(20px)',
      borderRight: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
    }}>
      <Toolbar sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 3,
        minHeight: { xs: 60, sm: 60 },
        background: alpha(theme.palette.primary.main, 0.03)
      }}>
        <Avatar sx={{
          background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          width: 42,
          height: 42,
          fontSize: '1.2rem',
          fontWeight: 600,
          boxShadow: `0 2px 10px ${alpha(theme.palette.primary.main, 0.2)}`,
        }}>
          SS
        </Avatar>
        <Typography variant="h6" noWrap sx={{
          fontWeight: 700,
          letterSpacing: '-0.5px',
          background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          SafeStreets
        </Typography>
      </Toolbar>

      <Divider sx={{ mx: 2, my: 1 }} />

      <List sx={{ flex: 1, px: 2, py: 2 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
            <ListItemButton
              onClick={() => handleNav(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2,
                minHeight: 48,
                position: 'relative',
                overflow: 'hidden',
                '&:before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.08)}, ${alpha(theme.palette.secondary.main, 0.08)})`,
                  opacity: 0,
                  transition: 'opacity 0.2s ease-in-out',
                },
                '&:hover:before': {
                  opacity: 1,
                },
                '&.Mui-selected': {
                  bgcolor: 'transparent',
                  '&:before': {
                    opacity: 1,
                    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.12)}, ${alpha(theme.palette.secondary.main, 0.12)})`,
                  },
                  '&:hover': {
                    bgcolor: alpha(theme.palette.primary.main, 0.12),
                  },
                  '& .MuiListItemIcon-root': {
                    color: theme.palette.primary.main,
                  },
                  '& .MuiTypography-root': {
                    fontWeight: 600,
                    color: theme.palette.primary.main,
                  }
                },
                '&:hover': {
                  bgcolor: alpha(theme.palette.primary.main, 0.05),
                }
              }}
            >
              <ListItemIcon sx={{
                minWidth: 40,
                color: location.pathname === item.path 
                  ? theme.palette.primary.main 
                  : alpha(theme.palette.text.primary, 0.7)
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: location.pathname === item.path ? 600 : 500
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ mx: 2, mb: 1 }} />

      <List sx={{ px: 2, pb: 2 }}>
        <ListItem disablePadding>
          <ListItemButton
            onClick={handleLogout}
            sx={{
              borderRadius: 2,
              color: theme.palette.error.main,
              '&:hover': {
                bgcolor: alpha(theme.palette.error.main, 0.05),
              }
            }}
          >
            <ListItemIcon sx={{ 
              minWidth: 40,
              color: 'inherit' 
            }}>
              <LogoutIcon />
            </ListItemIcon>
            <ListItemText 
              primary="Logout"
              primaryTypographyProps={{
                fontSize: '0.9rem',
                fontWeight: 500
              }}
            />
          </ListItemButton>
        </ListItem>
      </List>
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