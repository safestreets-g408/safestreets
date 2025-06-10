import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Badge,
  Avatar,
  Menu,
  MenuItem,
  ListItemIcon,
  Box,
  useTheme,
  alpha,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import SettingsIcon from '@mui/icons-material/Settings';
import LogoutIcon from '@mui/icons-material/Logout';
import RefreshIcon from '@mui/icons-material/Refresh';
import { DRAWER_WIDTH } from '../../config/constants';
//import { useAuth } from '../../hooks/useAuth';

const Header = ({ onDrawerToggle }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const navigate = useNavigate();
  const theme = useTheme();
  //const { logout } = useAuth();

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

  const notifications = [
    { id: 1, message: "New critical damage report submitted", time: "10 minutes ago" },
    { id: 2, message: "Repair team assigned to DR-2023-003", time: "1 hour ago" },
    { id: 3, message: "Weekly analytics report ready", time: "3 hours ago" }
  ];

  return (
    <AppBar
      position="fixed"
      elevation={0}
      sx={{
        width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
        ml: { sm: `${DRAWER_WIDTH}px` },
        boxShadow: 'none',
        backdropFilter: 'blur(20px)',
        backgroundColor: alpha(theme.palette.background.default, 0.9),
        borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        transition: theme.transitions.create(['width', 'margin'], {
          easing: theme.transitions.easing.sharp,
          duration: theme.transitions.duration.leavingScreen,
        }),
      }}
    >
      <Toolbar 
        sx={{ 
          justifyContent: 'space-between',
          minHeight: { xs: 64, sm: 70 },
          px: { xs: 2, sm: 3, md: 4 },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton
            color="primary"
            edge="start"
            onClick={onDrawerToggle}
            sx={{ 
              mr: 2, 
              display: { sm: 'none' },
              bgcolor: alpha(theme.palette.primary.main, 0.08),
              '&:hover': {
                bgcolor: alpha(theme.palette.primary.main, 0.12),
              }
            }}
          >
            <MenuIcon />
          </IconButton>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton 
            color="primary"
            onClick={() => window.location.reload()}
            sx={{ 
              width: 40,
              height: 40,
              bgcolor: alpha(theme.palette.primary.main, 0.08),
              '&:hover': {
                bgcolor: alpha(theme.palette.primary.main, 0.12),
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            <RefreshIcon />
          </IconButton>

          <IconButton 
            color="primary"
            onClick={handleNotificationOpen}
            sx={{ 
              width: 40,
              height: 40,
              bgcolor: alpha(theme.palette.primary.main, 0.08),
              '&:hover': {
                bgcolor: alpha(theme.palette.primary.main, 0.12),
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            <Badge badgeContent={notifications.length} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>

          <IconButton
            onClick={handleProfileMenuOpen}
            sx={{ 
              ml: 1,
              bgcolor: alpha(theme.palette.primary.main, 0.1),
              '&:hover': {
                bgcolor: alpha(theme.palette.primary.main, 0.2),
              }
            }}
          >
            <Avatar 
              sx={{ 
                bgcolor: theme.palette.primary.main,
                width: 32,
                height: 32,
                fontSize: '0.9rem',
                fontWeight: 600
              }}
            >
              AD
            </Avatar>
          </IconButton>
        </Box>

        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleProfileMenuClose}
          PaperProps={{
            sx: {
              width: 200,
              boxShadow: theme.shadows[8],
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
                bgcolor: theme.palette.primary.main,
                fontSize: '1.2rem',
                fontWeight: 600
              }}
            >
              AD
            </Avatar>
            <Typography variant="subtitle1" fontWeight={600}>Admin User</Typography>
            <Typography variant="body2" color="text.secondary">admin@example.com</Typography>
          </Box>
          <MenuItem onClick={() => {
            handleProfileMenuClose();
            navigate('/profile');
          }}>
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
          <MenuItem onClick={() => {
            handleProfileMenuClose();
          }}>
            <ListItemIcon>
              <LogoutIcon fontSize="small" color="error" />
            </ListItemIcon>
            <Typography variant="body2" color="error">Logout</Typography>
          </MenuItem>
        </Menu>

        <Menu
          anchorEl={notificationAnchor}
          open={Boolean(notificationAnchor)}
          onClose={handleNotificationClose}
          PaperProps={{
            sx: {
              width: 320,
              maxHeight: 400,
              boxShadow: theme.shadows[8],
              borderRadius: 2,
              mt: 1
            }
          }}
        >
          <Box sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Notifications</Typography>
            {notifications.map((notification) => (
              <Box 
                key={notification.id} 
                sx={{ 
                  mb: 2,
                  p: 1.5,
                  borderRadius: 1,
                  bgcolor: alpha(theme.palette.primary.main, 0.05),
                  '&:last-child': { mb: 0 }
                }}
              >
                <Typography variant="body2">{notification.message}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {notification.time}
                </Typography>
              </Box>
            ))}
          </Box>
        </Menu>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 