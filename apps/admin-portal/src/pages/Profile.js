import React, { useState, useEffect } from "react";
import { useTheme, alpha } from '@mui/material/styles';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Avatar,
  Button,
  Divider,
  TextField,
  CardContent,
  IconButton,
  Tabs,
  Tab,
  Badge,
  Tooltip,
  Chip as MuiChip,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Notifications,
  Email,
  Phone,
  LocationOn,
  Security,
  History,
  Settings,
  Person,
  Dashboard,
} from "@mui/icons-material";
import { useAuth } from "../hooks/useAuth";
import { API_BASE_URL, API_ENDPOINTS, TOKEN_KEY } from "../config/constants";

// Professional color palette - will be used with theme
const getColors = (theme) => ({
  primary: theme.palette.primary.main,
  primaryDark: theme.palette.primary.dark,
  secondary: theme.palette.secondary.main,
  success: theme.palette.success.main,
  warning: theme.palette.warning.main,
  error: theme.palette.error.main,
  surface: theme.palette.background.paper,
  border: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : '#e2e8f0',
  text: {
    primary: theme.palette.text.primary,
    secondary: theme.palette.text.secondary
  }
});

const Profile = () => {
  const theme = useTheme();
  const colors = getColors(theme);
  const [tabValue, setTabValue] = useState(0);
  const [editMode, setEditMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [activityData, setActivityData] = useState([]);
  const [settings, setSettings] = useState({
    emailNotifications: true,
    damageAlerts: true,
    systemUpdates: false,
    twoFactorAuth: false
  });
  const [passwordDialog, setPasswordDialog] = useState(false);
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const { user, updateUser } = useAuth();
  const [userData, setUserData] = useState({
    name: user?.name || "",
    email: user?.email || "",
    role: user?.role || "admin",
    position: user?.profile?.position || "Administrator",
    phone: user?.profile?.phone || "",
    location: user?.profile?.location || "",
    department: user?.profile?.department || "",
    joinDate: user?.profile?.joinDate ? new Date(user.profile.joinDate).toLocaleDateString('en-US', { month: 'long', year: 'numeric' }) : "Recently joined",
    bio: user?.profile?.bio || "No bio available",
    skills: user?.profile?.skills || []
  });

  // Load user profile data
  useEffect(() => {
    if (user) {
      setUserData({
        name: user.name || "",
        email: user.email || "",
        role: user.role || "admin",
        position: user.profile?.position || "Administrator",
        phone: user.profile?.phone || "",
        location: user.profile?.location || "",
        department: user.profile?.department || "",
        joinDate: user.profile?.joinDate ? new Date(user.profile.joinDate).toLocaleDateString('en-US', { month: 'long', year: 'numeric' }) : "Recently joined",
        bio: user.profile?.bio || "No bio available",
        skills: user.profile?.skills || []
      });
    }
  }, [user]);

  // Load activity data
  useEffect(() => {
    fetchActivityData();
    fetchNotifications();
  }, []);

  const fetchActivityData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.DAMAGE_REPORTS}?limit=5&sortBy=updatedAt&sortOrder=desc`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem(TOKEN_KEY)}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        const formattedActivity = data.reports?.map(report => ({
          id: report._id,
          title: `${report.status === 'approved' ? 'Approved' : report.status === 'assigned' ? 'Assigned' : 'Reviewed'} ${report.damageType} report`,
          time: new Date(report.updatedAt).toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          }),
          description: `${report.description?.substring(0, 100) || 'No description'}${report.description?.length > 100 ? '...' : ''} - ${report.location}`,
          type: report.status
        })) || [];
        setActivityData(formattedActivity);
      }
    } catch (error) {
      console.error('Error fetching activity data:', error);
    }
  };

  const fetchNotifications = async () => {
    // Mock notifications for now - in a real app, this would fetch from an API
    const mockNotifications = [
      { id: 1, type: 'damage_alert', message: 'New critical damage report', unread: true },
      { id: 2, type: 'system', message: 'System maintenance scheduled', unread: true },
      { id: 3, type: 'email', message: 'Weekly report ready', unread: false }
    ];
    setNotifications(mockNotifications);
  };

  const handleSettingsToggle = (setting) => {
    setSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
    // In a real app, this would make an API call to save settings
    setSuccess(`${setting} setting updated`);
  };

  const handlePasswordChange = () => {
    setPasswordDialog(true);
  };

  const handlePasswordSubmit = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setError('New passwords do not match');
      return;
    }
    
    if (passwordData.newPassword.length < 6) {
      setError('Password must be at least 6 characters long');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.AUTH}/change-password`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem(TOKEN_KEY)}`
        },
        body: JSON.stringify({
          currentPassword: passwordData.currentPassword,
          newPassword: passwordData.newPassword
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to change password');
      }

      setSuccess('Password changed successfully');
      setPasswordDialog(false);
      setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
    } catch (err) {
      setError(err.message || 'An error occurred while changing password');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasswordDialogClose = () => {
    setPasswordDialog(false);
    setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
  };

  const handleTwoFactorAuth = () => {
    handleSettingsToggle('twoFactorAuth');
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleEditToggle = () => {
    setEditMode(!editMode);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUserData({
      ...userData,
      [name]: value,
    });
  };

  const handleSave = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.PROFILE}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem(TOKEN_KEY)}`
        },
        body: JSON.stringify({
          name: userData.name,
          email: userData.email,
          profile: {
            position: userData.position,
            phone: userData.phone,
            location: userData.location,
            department: userData.department,
            bio: userData.bio,
            skills: userData.skills
          }
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to update profile');
      }

      const updatedUser = await response.json();
      
      // Update local user state
      updateUser(updatedUser);
      setSuccess('Profile updated successfully');
      setEditMode(false);
    } catch (err) {
      setError(err.message || 'An error occurred while updating profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setError(null);
    setSuccess(null);
  };

  return (
    <Box sx={{ 
      flexGrow: 1, 
      minHeight: "100vh", 
      py: 4,
      bgcolor: theme.palette.mode === 'dark' ? 'background.default' : 'grey.50'
    }}>
      <Container maxWidth="lg">
        {/* Error Snackbar */}
        <Snackbar 
          open={!!error} 
          autoHideDuration={6000} 
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        </Snackbar>
        
        {/* Success Snackbar */}
        <Snackbar 
          open={!!success} 
          autoHideDuration={6000} 
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
            {success}
          </Alert>
        </Snackbar>
        
        <Box
          sx={{
            borderRadius: 2,
            overflow: "hidden",
          }}
        >
          <Grid container spacing={4}>
            {/* Header */}
            <Grid item xs={12}>
              <Paper 
                elevation={0} 
                sx={{ 
                  p: 3, 
                  borderRadius: 2,
                  backgroundColor: colors.surface,
                  border: `1px solid ${colors.border}`,
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: 2
                  }}
                >
                  <Box sx={{ display: "flex", alignItems: "center" }}>
                    <Badge
                      overlap="circular"
                      anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                      variant="dot"
                      color="success"
                    >
                      <Avatar
                        src="/static/images/avatar/sarah.jpg"
                        sx={{ 
                          width: 64, 
                          height: 64, 
                          mr: 2,
                          backgroundColor: colors.primary,
                          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.15)" 
                        }}
                      >
                        {userData.name.split(' ').map(n => n[0]).join('').toUpperCase() || 'AD'}
                      </Avatar>
                    </Badge>
                    <Box>
                      <Typography variant="h5" fontWeight="600">
                        {userData.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {userData.position}
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <Tooltip title="Notifications">
                      <IconButton
                        color="primary"
                        sx={{
                          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : colors.border,
                          "&:hover": {
                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : alpha(theme.palette.primary.main, 0.1),
                          },
                        }}
                      >
                        <Badge badgeContent={notifications.filter(n => n.unread).length} color="error">
                          <Notifications />
                        </Badge>
                      </IconButton>
                    </Tooltip>
                    <Button
                      variant="contained"
                      startIcon={editMode ? <SaveIcon /> : <EditIcon />}
                      onClick={editMode ? handleSave : handleEditToggle}
                      disabled={isLoading}
                      sx={{
                        borderRadius: 1.5,
                        px: 3,
                        py: 1,
                        backgroundColor: colors.primary,
                        "&:hover": {
                          backgroundColor: colors.primaryDark,
                        },
                        boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
                      }}
                    >
                      {isLoading 
                        ? "Saving..." 
                        : editMode 
                          ? "Save Changes" 
                          : "Edit Profile"}
                    </Button>
                  </Box>
                </Box>
              </Paper>
            </Grid>

            {/* Main Content */}
            <Grid item xs={12} md={4}>
              <Paper
                elevation={0}
                sx={{
                  borderRadius: 2,
                  overflow: "hidden",
                  height: "100%",
                  border: `1px solid ${colors.border}`,
                }}
              >
                <Box
                  sx={{
                    p: 3,
                    textAlign: "center",
                    background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
                    color: "white",
                  }}
                >
                  <Avatar
                    src="/static/images/avatar/sarah.jpg"
                    sx={{ 
                      width: 100, 
                      height: 100, 
                      mx: "auto", 
                      border: "4px solid white",
                      boxShadow: "0 4px 10px rgba(0, 0, 0, 0.2)"
                    }}
                  >
                    {userData.name.split(' ').map(n => n[0]).join('').toUpperCase() || 'AD'}
                  </Avatar>
                  <Typography variant="h5" sx={{ mt: 2, fontWeight: "bold" }}>
                    {userData.name}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9, mb: 1 }}>
                    {userData.position}
                  </Typography>
                </Box>
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <Email sx={{ mr: 2, color: theme.palette.primary.main }} />
                    <Typography variant="body2">{userData.email}</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <Phone sx={{ mr: 2, color: theme.palette.primary.main }} />
                    <Typography variant="body2">{userData.phone || 'Not provided'}</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <LocationOn sx={{ mr: 2, color: theme.palette.primary.main }} />
                    <Typography variant="body2">{userData.location || 'Not provided'}</Typography>
                  </Box>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
                    Department
                  </Typography>
                  <Typography variant="body2" paragraph>
                    {userData.department || 'Not specified'}
                  </Typography>
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
                    Joined
                  </Typography>
                  <Typography variant="body2">{userData.joinDate}</Typography>
                </CardContent>
              </Paper>
            </Grid>

            <Grid item xs={12} md={8}>
                <Paper
                  elevation={0}
                  sx={{
                    borderRadius: 2,
                    overflow: "hidden",
                    height: "100%",
                    border: `1px solid ${colors.border}`,
                  }}
                >
                  <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    variant="fullWidth"
                    sx={{ 
                      borderBottom: 1, 
                      borderColor: "divider",
                      bgcolor: theme.palette.background.paper,
                    }}
                    indicatorColor="primary"
                    textColor="primary"
                  >
                    <Tab
                      icon={<Person fontSize="small" />}
                      label="Profile"
                      iconPosition="start"
                      sx={{ textTransform: "none", py: 1.5 }}
                    />
                    <Tab
                      icon={<History fontSize="small" />}
                      label="Activity"
                      iconPosition="start"
                      sx={{ textTransform: "none", py: 1.5 }}
                    />
                    <Tab
                      icon={<Settings fontSize="small" />}
                      label="Settings"
                      iconPosition="start"
                      sx={{ textTransform: "none", py: 1.5 }}
                    />
                  </Tabs>

                  {tabValue === 0 && (
                    <Box sx={{ p: 3 }}>
                      <Typography variant="h6" fontWeight="bold" gutterBottom color="primary">
                        About Me
                      </Typography>
                      {editMode ? (
                        <TextField
                          name="bio"
                          value={userData.bio}
                          onChange={handleInputChange}
                          multiline
                          rows={4}
                          fullWidth
                          variant="outlined"
                          sx={{ 
                            mb: 3,
                            '& .MuiOutlinedInput-root': {
                              bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                            }
                          }}
                        />
                      ) : (
                        <Typography variant="body2" paragraph sx={{ mb: 3 }}>
                          {userData.bio}
                        </Typography>
                      )}

                      <Typography variant="h6" fontWeight="bold" gutterBottom color="primary">
                        Personal Information
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Full Name"
                            name="name"
                            value={userData.name}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ 
                              mb: 2,
                              '& .MuiOutlinedInput-root': {
                                bgcolor: editMode ? (theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)') : 'transparent',
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Job Title"
                            name="position"
                            value={userData.position}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ 
                              mb: 2,
                              '& .MuiOutlinedInput-root': {
                                bgcolor: editMode ? (theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)') : 'transparent',
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Email"
                            name="email"
                            value={userData.email}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ 
                              mb: 2,
                              '& .MuiOutlinedInput-root': {
                                bgcolor: editMode ? (theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)') : 'transparent',
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Phone"
                            name="phone"
                            value={userData.phone}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ 
                              mb: 2,
                              '& .MuiOutlinedInput-root': {
                                bgcolor: editMode ? (theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)') : 'transparent',
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Location"
                            name="location"
                            value={userData.location}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ 
                              mb: 2,
                              '& .MuiOutlinedInput-root': {
                                bgcolor: editMode ? (theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)') : 'transparent',
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Department"
                            name="department"
                            value={userData.department}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ 
                              mb: 2,
                              '& .MuiOutlinedInput-root': {
                                bgcolor: editMode ? (theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)') : 'transparent',
                              }
                            }}
                          />
                        </Grid>
                      </Grid>

                      <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ mt: 2 }} color="primary">
                        Skills & Expertise
                      </Typography>
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                        {(userData.skills && userData.skills.length > 0)
                          ? userData.skills.map((skill) => (
                              <MuiChip 
                                key={skill} 
                                label={skill} 
                                color="primary" 
                                variant="outlined" 
                                size="small"
                                onDelete={editMode ? () => {
                                  setUserData({
                                    ...userData,
                                    skills: userData.skills.filter(s => s !== skill)
                                  });
                                } : undefined}
                              />
                            ))
                          : !editMode && 
                            <Typography variant="body2" color="text.secondary">
                              No skills listed yet.
                            </Typography>
                        }
                        
                      {editMode && (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                            <TextField
                              placeholder="Add skill"
                              size="small"
                              sx={{ 
                                minWidth: 150,
                                '& .MuiOutlinedInput-root': {
                                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                                }
                              }}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' && e.target.value.trim()) {
                                  const newSkill = e.target.value.trim();
                                  if (!userData.skills.includes(newSkill)) {
                                    setUserData({
                                      ...userData,
                                      skills: [...(userData.skills || []), newSkill]
                                    });
                                  }
                                  e.target.value = '';
                                  e.preventDefault();
                                }
                              }}
                            />
                            <Typography variant="caption" color="text.secondary">
                              Press Enter to add
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    </Box>
                  )}

                  {tabValue === 1 && (
                    <Box sx={{ p: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6" fontWeight="bold" color="primary">
                          Recent Activity
                        </Typography>
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={fetchActivityData}
                          sx={{ 
                            borderRadius: 1.5,
                            borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.23)' : 'rgba(0, 0, 0, 0.23)',
                            color: theme.palette.text.primary,
                            '&:hover': {
                              borderColor: theme.palette.primary.main,
                              bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)',
                            }
                          }}
                        >
                          Refresh
                        </Button>
                      </Box>
                      {activityData.length > 0 ? (
                        <Timeline>
                          {activityData.map((activity, index) => (
                            <TimelineItem
                              key={activity.id || index}
                              title={activity.title}
                              time={activity.time}
                              description={activity.description}
                              type={activity.type}
                            />
                          ))}
                        </Timeline>
                      ) : (
                        <Box sx={{ textAlign: 'center', py: 4 }}>
                          <Typography variant="body2" color="text.secondary">
                            No recent activity found. Start by reviewing damage reports or updating your profile.
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  )}

                  {tabValue === 2 && (
                    <Box sx={{ p: 3 }}>
                      <Grid container spacing={3}>
                        <Grid item xs={12}>
                          <Typography variant="h6" fontWeight="bold" gutterBottom color="primary">
                            Account Settings
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Paper
                            elevation={0}
                            sx={{ 
                              borderRadius: 2,
                              border: `1px solid ${colors.border}`,
                            }}
                          >
                            <CardContent>
                              <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="primary">
                                Security
                              </Typography>
                              <Grid container spacing={2}>
                                <Grid item xs={12}>
                                  <Button
                                    variant="outlined"
                                    fullWidth
                                    startIcon={<Security />}
                                    onClick={handlePasswordChange}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      mb: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Change Password
                                  </Button>
                                </Grid>
                                <Grid item xs={12}>
                                  <Button
                                    variant={settings.twoFactorAuth ? "contained" : "outlined"}
                                    fullWidth
                                    startIcon={<Security />}
                                    onClick={handleTwoFactorAuth}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Two-Factor Authentication {settings.twoFactorAuth ? '(Enabled)' : '(Disabled)'}
                                  </Button>
                                </Grid>
                              </Grid>
                            </CardContent>
                          </Paper>
                        </Grid>
                        <Grid item xs={12}>
                          <Paper
                            elevation={0}
                            sx={{ 
                              borderRadius: 2,
                              border: `1px solid ${colors.border}`,
                            }}
                          >
                            <CardContent>
                              <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="primary">
                                Notifications
                              </Typography>
                              <Grid container spacing={2}>
                                <Grid item xs={12}>
                                  <Button
                                    variant={settings.emailNotifications ? "contained" : "outlined"}
                                    fullWidth
                                    startIcon={<Email />}
                                    onClick={() => handleSettingsToggle('emailNotifications')}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      mb: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Email Notifications {settings.emailNotifications ? '(On)' : '(Off)'}
                                  </Button>
                                </Grid>
                                <Grid item xs={12}>
                                  <Button
                                    variant={settings.damageAlerts ? "contained" : "outlined"}
                                    fullWidth
                                    startIcon={<Notifications />}
                                    onClick={() => handleSettingsToggle('damageAlerts')}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      mb: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Damage Alerts {settings.damageAlerts ? '(On)' : '(Off)'}
                                  </Button>
                                </Grid>
                                <Grid item xs={12}>
                                  <Button
                                    variant={settings.systemUpdates ? "contained" : "outlined"}
                                    fullWidth
                                    startIcon={<Dashboard />}
                                    onClick={() => handleSettingsToggle('systemUpdates')}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    System Updates {settings.systemUpdates ? '(On)' : '(Off)'}
                                  </Button>
                                </Grid>
                              </Grid>
                            </CardContent>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </Box>
        </Container>

        {/* Password Change Dialog */}
        <Dialog 
          open={passwordDialog} 
          onClose={handlePasswordDialogClose}
          maxWidth="sm"
          fullWidth
          PaperProps={{
            sx: {
              bgcolor: theme.palette.background.paper,
              backgroundImage: 'none',
            }
          }}
        >
          <DialogTitle sx={{ color: theme.palette.text.primary }}>
            Change Password
          </DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 1 }}>
              <TextField
                fullWidth
                type="password"
                label="Current Password"
                value={passwordData.currentPassword}
                onChange={(e) => setPasswordData(prev => ({ ...prev, currentPassword: e.target.value }))}
                sx={{ 
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                  }
                }}
              />
              <TextField
                fullWidth
                type="password"
                label="New Password"
                value={passwordData.newPassword}
                onChange={(e) => setPasswordData(prev => ({ ...prev, newPassword: e.target.value }))}
                sx={{ 
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                  }
                }}
              />
              <TextField
                fullWidth
                type="password"
                label="Confirm New Password"
                value={passwordData.confirmPassword}
                onChange={(e) => setPasswordData(prev => ({ ...prev, confirmPassword: e.target.value }))}
                error={passwordData.confirmPassword && passwordData.newPassword !== passwordData.confirmPassword}
                helperText={passwordData.confirmPassword && passwordData.newPassword !== passwordData.confirmPassword ? 'Passwords do not match' : ''}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                  }
                }}
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={handlePasswordDialogClose}>Cancel</Button>
            <Button 
              onClick={handlePasswordSubmit} 
              variant="contained"
              disabled={isLoading || !passwordData.currentPassword || !passwordData.newPassword || !passwordData.confirmPassword}
            >
              {isLoading ? 'Changing...' : 'Change Password'}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    );
};

// Timeline components
const Timeline = ({ children }) => (
  <Box sx={{ mt: 2 }}>
    {children}
  </Box>
);

const TimelineItem = ({ title, time, description, type }) => {
  const theme = useTheme();
  
  // Color based on activity type
  const getTypeColor = (activityType) => {
    switch (activityType) {
      case 'approved': return theme.palette.success.main;
      case 'assigned': return theme.palette.info.main;
      case 'pending': return theme.palette.warning.main;
      case 'rejected': return theme.palette.error.main;
      default: return theme.palette.primary.main;
    }
  };

  return (
    <Box 
      sx={{ 
        mb: 3, 
        pb: 3, 
        borderBottom: "1px solid", 
        borderColor: theme.palette.divider,
        position: "relative",
        pl: 3,
        "&::before": {
          content: '""',
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: 4,
          borderRadius: 4,
          backgroundColor: getTypeColor(type),
        },
        "&:hover": {
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.02)' : 'rgba(0, 0, 0, 0.02)',
          borderRadius: 1,
        }
      }}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
        <Typography variant="subtitle1" fontWeight="bold" color="text.primary">
          {title}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {time}
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary">
        {description}
      </Typography>
    </Box>
  );
};

export default Profile;
