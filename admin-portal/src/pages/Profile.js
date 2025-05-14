import React, { useState } from "react";
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
  Card,
  CardContent,
  IconButton,
  Fade,
  Tabs,
  Tab,
  Badge,
  Tooltip,
  useTheme,
  useMediaQuery,
  Chip as MuiChip,
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
import { styled } from "@mui/material/styles";
import { motion } from "framer-motion";

const MotionCard = motion(Card);

const StyledBadge = styled(Badge)(({ theme }) => ({
  "& .MuiBadge-badge": {
    backgroundColor: "#44b700",
    color: "#44b700",
    boxShadow: `0 0 0 2px ${theme.palette.background.paper}`,
    "&::after": {
      position: "absolute",
      top: 0,
      left: 0,
      width: "100%",
      height: "100%",
      borderRadius: "50%",
      animation: "ripple 1.2s infinite ease-in-out",
      border: "1px solid currentColor",
      content: '""',
    },
  },
  "@keyframes ripple": {
    "0%": {
      transform: "scale(.8)",
      opacity: 1,
    },
    "100%": {
      transform: "scale(2.4)",
      opacity: 0,
    },
  },
}));

const Profile = () => {
  const theme = useTheme();
  // eslint-disable-next-line no-unused-vars
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const [tabValue, setTabValue] = useState(0);
  const [editMode, setEditMode] = useState(false);
  const [userData, setUserData] = useState({
    name: "Sarah Johnson",
    role: "Senior Road Safety Administrator",
    email: "sarah.johnson@safestreets.org",
    phone: "+1 (555) 123-4567",
    location: "San Francisco, CA",
    department: "Infrastructure Management",
    joinDate: "March 2020",
    bio: "Road safety expert with over 10 years of experience in infrastructure management and damage assessment. Specialized in implementing AI-based solutions for road maintenance prioritization.",
  });

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

  const handleSave = () => {
    // Here you would typically save the data to your backend
    setEditMode(false);
  };

  return (
    <Box sx={{ flexGrow: 1, minHeight: "100vh", bgcolor: "background.default", py: 4 }}>
      <Container maxWidth="lg">
        <Fade in={true} timeout={800}>
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
                  elevation={2} 
                  sx={{ 
                    p: 3, 
                    borderRadius: 2,
                    background: theme.palette.background.paper,
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
                      <StyledBadge
                        overlap="circular"
                        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
                        variant="dot"
                      >
                        <Avatar
                          src="/static/images/avatar/sarah.jpg"
                          sx={{ 
                            width: 64, 
                            height: 64, 
                            mr: 2,
                            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.15)" 
                          }}
                        />
                      </StyledBadge>
                      <Box>
                        <Typography variant="h5" fontWeight="600">
                          {userData.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {userData.role}
                        </Typography>
                      </Box>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                      <Tooltip title="Notifications">
                        <IconButton
                          color="primary"
                          sx={{
                            bgcolor: theme.palette.action.hover,
                            "&:hover": {
                              bgcolor: theme.palette.action.selected,
                            },
                          }}
                        >
                          <Badge badgeContent={3} color="error">
                            <Notifications />
                          </Badge>
                        </IconButton>
                      </Tooltip>
                      <Button
                        variant="contained"
                        startIcon={editMode ? <SaveIcon /> : <EditIcon />}
                        onClick={editMode ? handleSave : handleEditToggle}
                        sx={{
                          borderRadius: 1.5,
                          px: 3,
                          py: 1,
                          boxShadow: 2,
                        }}
                      >
                        {editMode ? "Save Changes" : "Edit Profile"}
                      </Button>
                    </Box>
                  </Box>
                </Paper>
              </Grid>

              {/* Main Content */}
              <Grid item xs={12} md={4}>
                <MotionCard
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    overflow: "hidden",
                    height: "100%",
                  }}
                >
                  <Box
                    sx={{
                      p: 3,
                      textAlign: "center",
                      background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
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
                    />
                    <Typography variant="h5" sx={{ mt: 2, fontWeight: "bold" }}>
                      {userData.name}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9, mb: 1 }}>
                      {userData.role}
                    </Typography>
                  </Box>
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                      <Email sx={{ mr: 2, color: theme.palette.primary.main }} />
                      <Typography variant="body2">{userData.email}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                      <Phone sx={{ mr: 2, color: theme.palette.primary.main }} />
                      <Typography variant="body2">{userData.phone}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                      <LocationOn sx={{ mr: 2, color: theme.palette.primary.main }} />
                      <Typography variant="body2">{userData.location}</Typography>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
                      Department
                    </Typography>
                    <Typography variant="body2" paragraph>
                      {userData.department}
                    </Typography>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
                      Joined
                    </Typography>
                    <Typography variant="body2">{userData.joinDate}</Typography>
                  </CardContent>
                </MotionCard>
              </Grid>

              <Grid item xs={12} md={8}>
                <Paper
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    overflow: "hidden",
                    height: "100%",
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
                          sx={{ mb: 3 }}
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
                            sx={{ mb: 2 }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            label="Job Title"
                            name="role"
                            value={userData.role}
                            onChange={handleInputChange}
                            fullWidth
                            variant="outlined"
                            disabled={!editMode}
                            size="small"
                            sx={{ mb: 2 }}
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
                            sx={{ mb: 2 }}
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
                            sx={{ mb: 2 }}
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
                            sx={{ mb: 2 }}
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
                            sx={{ mb: 2 }}
                          />
                        </Grid>
                      </Grid>

                      <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ mt: 2 }} color="primary">
                        Skills & Expertise
                      </Typography>
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                        {["Road Safety", "Infrastructure Management", "Damage Assessment", "AI Solutions", "Project Management", "Data Analysis"].map((skill) => (
                          <MuiChip 
                            key={skill} 
                            label={skill} 
                            color="primary" 
                            variant="outlined" 
                            size="small"
                          />
                        ))}
                      </Box>
                    </Box>
                  )}

                  {tabValue === 1 && (
                    <Box sx={{ p: 3 }}>
                      <Typography variant="h6" fontWeight="bold" gutterBottom color="primary">
                        Recent Activity
                      </Typography>
                      <Timeline>
                        <TimelineItem
                          title="Approved road repair request"
                          time="Today, 10:30 AM"
                          description="Approved emergency repair for pothole damage on Main Street and 5th Avenue"
                        />
                        <TimelineItem
                          title="Updated damage assessment protocol"
                          time="Yesterday, 2:15 PM"
                          description="Revised the severity classification system for road surface cracks"
                        />
                        <TimelineItem
                          title="Reviewed AI damage detection results"
                          time="Oct 15, 2023"
                          description="Validated AI-detected road damages in the downtown area with 94% accuracy"
                        />
                        <TimelineItem
                          title="Assigned repair crew"
                          time="Oct 12, 2023"
                          description="Dispatched maintenance team to address critical infrastructure issues on Highway 101"
                        />
                        <TimelineItem
                          title="Generated monthly report"
                          time="Oct 1, 2023"
                          description="Created comprehensive analysis of road conditions across the city"
                        />
                      </Timeline>
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
                          <MotionCard
                            whileHover={{ y: -5 }}
                            transition={{ duration: 0.3 }}
                            elevation={1}
                            sx={{ borderRadius: 2 }}
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
                                    variant="outlined"
                                    fullWidth
                                    startIcon={<Security />}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Two-Factor Authentication
                                  </Button>
                                </Grid>
                              </Grid>
                            </CardContent>
                          </MotionCard>
                        </Grid>
                        <Grid item xs={12}>
                          <MotionCard
                            whileHover={{ y: -5 }}
                            transition={{ duration: 0.3 }}
                            elevation={1}
                            sx={{ borderRadius: 2 }}
                          >
                            <CardContent>
                              <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="primary">
                                Notifications
                              </Typography>
                              <Grid container spacing={2}>
                                <Grid item xs={12}>
                                  <Button
                                    variant="outlined"
                                    fullWidth
                                    startIcon={<Email />}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      mb: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Email Notifications
                                  </Button>
                                </Grid>
                                <Grid item xs={12}>
                                  <Button
                                    variant="outlined"
                                    fullWidth
                                    startIcon={<Notifications />}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      mb: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    Damage Alerts
                                  </Button>
                                </Grid>
                                <Grid item xs={12}>
                                  <Button
                                    variant="outlined"
                                    fullWidth
                                    startIcon={<Dashboard />}
                                    sx={{
                                      justifyContent: "flex-start",
                                      textTransform: "none",
                                      py: 1,
                                      borderRadius: 1.5,
                                    }}
                                  >
                                    System Updates
                                  </Button>
                                </Grid>
                              </Grid>
                            </CardContent>
                          </MotionCard>
                        </Grid>
                      </Grid>
                    </Box>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </Box>
        </Fade>
      </Container>
    </Box>
  );
};

// Timeline components
const Timeline = ({ children }) => (
  <Box sx={{ mt: 2 }}>
    {children}
  </Box>
);

const TimelineItem = ({ title, time, description }) => (
  <Box 
    sx={{ 
      mb: 3, 
      pb: 3, 
      borderBottom: "1px solid", 
      borderColor: "divider",
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
        backgroundColor: "primary.light",
      }
    }}
  >
    <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
      <Typography variant="subtitle1" fontWeight="bold">
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

export default Profile;
