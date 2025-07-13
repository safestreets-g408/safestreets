import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { useTheme, alpha } from '@mui/material/styles';
import { Box } from '@mui/material';
import { 
  Navigation,
  Hero,
  Features,
  Statistics,
  Innovations,
  Benefits,
  SystemComponents,
  Footer,
  ScrollToTop
} from '../components/landing';
import {
  Analytics,
  Security,
  Dashboard,
  Assignment,
  MapOutlined,
  SmartToy,
  Visibility,
  Speed,
  Psychology,
  Storage,
  CameraAlt,
  NotificationsActive,
  PhoneAndroid,
  Computer,
} from '@mui/icons-material';

const Landing = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  
  // State for interactive features
  const [demoOpen, setDemoOpen] = useState(false);
  const [statsVisible, setStatsVisible] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Reference for scrolling sections
  const featuresRef = useRef(null);
  const benefitsRef = useRef(null);

  // Animate stats when they come into view
  useEffect(() => {
    const timer = setTimeout(() => setStatsVisible(true), 1000);
    return () => clearTimeout(timer);
  }, []);
  
  // Handle navigation to features section
  const scrollToSection = (ref) => {
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Statistics data
  const stats = [
    { label: 'Roads Monitored', value: 1500, suffix: '+', icon: <MapOutlined /> },
    { label: 'Damages Detected', value: 250, suffix: '+', icon: <Visibility /> },
    { label: 'AI Accuracy', value: 94, suffix: '%', icon: <SmartToy /> },
    { label: 'Response Time', value: 2, suffix: 'hrs', icon: <Speed /> },
  ];

  // Innovation highlights and tech advancements
  const innovations = [
    {
      title: 'Neural Damage Detection',
      description: 'Our advanced computer vision system utilizes a custom-trained Vision Transformer model to identify and classify road damage with unprecedented accuracy.',
      icon: <SmartToy />,
      metric: '94.7%',
      metricLabel: 'Detection Accuracy'
    },
    {
      title: 'LLM-Powered Reports',
      description: 'Large Language Models analyze damage patterns to generate detailed assessment reports, severity classifications, and repair recommendations.',
      icon: <Psychology />,
      metric: '3.2x',
      metricLabel: 'Faster Processing'
    },
    {
      title: 'Edge AI Computing',
      description: 'On-device neural processing allows for real-time damage analysis even in areas with limited connectivity, optimized for mobile field operations.',
      icon: <Speed />,
      metric: '<2s',
      metricLabel: 'Analysis Time'
    }
  ];

  // ROI and benefits data
  const benefits = [
    { title: 'Cost Reduction', value: '35%', description: 'Lower maintenance costs through predictive analytics' },
    { title: 'Time Savings', value: '60%', description: 'Faster damage detection and response times' },
    { title: 'Accuracy Improvement', value: '94%', description: 'AI-powered precision in damage classification' },
    { title: 'Operational Efficiency', value: '45%', description: 'Streamlined workflows and automated reporting' }
  ];

  // Generate feature icon colors with enhanced dark mode support
  const getFeatureColors = () => {
    const mode = theme.palette.mode;
    return {
      green: mode === 'dark' ? '#66BB6A' : '#4CAF50',
      blue: mode === 'dark' ? '#42A5F5' : '#2196F3',
      orange: mode === 'dark' ? '#FFA726' : '#FF9800',
      purple: mode === 'dark' ? '#AB47BC' : '#9C27B0',
      red: mode === 'dark' ? '#EF5350' : '#F44336',
      cyan: mode === 'dark' ? '#26C6DA' : '#00BCD4',
      lightGreen: mode === 'dark' ? '#9CCC65' : '#8BC34A',
      deepOrange: mode === 'dark' ? '#FF7043' : '#FF5722',
      blueGrey: mode === 'dark' ? '#78909C' : '#607D8B',
      indigo: mode === 'dark' ? '#7986CB' : '#3F51B5',
      teal: mode === 'dark' ? '#4DB6AC' : '#009688',
      pink: mode === 'dark' ? '#F06292' : '#E91E63',
    };
  };

  const featureColors = getFeatureColors();

  const features = [
    {
      icon: <SmartToy />,
      title: 'AI-Powered Detection',
      description: 'Advanced Vision Transformer (ViT) models detect and classify road damages including potholes, cracks, and erosion with high accuracy.',
      color: featureColors.green
    },
    {
      icon: <CameraAlt />,
      title: 'Mobile Image Capture',
      description: 'Field workers can capture road damage images with automatic GPS tagging using our React Native mobile application.',
      color: featureColors.blue
    },
    {
      icon: <Psychology />,
      title: 'AI Report Generation',
      description: 'Google Gemini integration automatically generates professional damage report summaries with severity assessment and priority rating.',
      color: featureColors.orange
    },
    {
      icon: <Dashboard />,
      title: 'Admin Dashboard',
      description: 'Comprehensive React-based admin portal with Material-UI for managing reports, analytics, and repair assignments.',
      color: featureColors.purple
    },
    {
      icon: <MapOutlined />,
      title: 'Interactive Map View',
      description: 'Location-based heatmaps and visual representation of damage reports with filtering and search capabilities.',
      color: featureColors.red
    },
    {
      icon: <Analytics />,
      title: 'Advanced Analytics',
      description: 'Real-time data processing with trends, severity distribution, and insights to identify most-affected zones.',
      color: featureColors.cyan
    },
    {
      icon: <Assignment />,
      title: 'Task Management',
      description: 'Assign repair tasks to field teams, track progress, and manage the entire repair workflow efficiently.',
      color: featureColors.lightGreen
    },
    {
      icon: <NotificationsActive />,
      title: 'Smart Notifications',
      description: 'Email alerts for high-priority damages, push notifications for field teams, and automated repair reminders.',
      color: featureColors.deepOrange
    },
    {
      icon: <Security />,
      title: 'Secure Authentication',
      description: 'JWT-based authentication with role-based access control for administrators and field workers.',
      color: featureColors.blueGrey
    }
  ];

  const systemComponents = [
    {
      title: 'Mobile Application',
      description: 'React Native app for field workers with offline support',
      features: ['Image capture with GPS', 'Status tracking', 'Offline synchronization'],
      icon: <PhoneAndroid sx={{ fontSize: 40 }} />,
      type: 'mobile'
    },
    {
      title: 'AI Model Server',
      description: 'Flask-based Vision Transformer inference server',
      features: ['ViT classification', 'CNN road validation', 'Damage severity assessment'],
      icon: <SmartToy sx={{ fontSize: 40 }} />,
      type: 'ai'
    },
    {
      title: 'Backend API',
      description: 'Node.js/Express REST API with MongoDB',
      features: ['Secure authentication', 'Data management', 'Task assignment'],
      icon: <Storage sx={{ fontSize: 40 }} />,
      type: 'backend'
    },
    {
      title: 'Admin Portal',
      description: 'React-based web dashboard with Material-UI',
      features: ['Analytics dashboard', 'Map visualization', 'Repair management'],
      icon: <Computer sx={{ fontSize: 40 }} />,
      type: 'frontend'
    }
  ];

  return (
    <Box 
      sx={{ 
        minHeight: '100vh', 
        bgcolor: 'background.default',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: theme.palette.mode === 'dark'
            ? `radial-gradient(circle at 20% 80%, ${alpha(theme.palette.primary.main, 0.1)} 0%, transparent 50%),
               radial-gradient(circle at 80% 20%, ${alpha(theme.palette.secondary.main, 0.1)} 0%, transparent 50%),
               radial-gradient(circle at 40% 40%, ${alpha(theme.palette.primary.dark, 0.05)} 0%, transparent 50%)`
            : `radial-gradient(circle at 20% 80%, ${alpha(theme.palette.primary.main, 0.05)} 0%, transparent 50%),
               radial-gradient(circle at 80% 20%, ${alpha(theme.palette.secondary.main, 0.05)} 0%, transparent 50%)`,
          pointerEvents: 'none',
          zIndex: -1,
          animation: 'backgroundMove 20s ease-in-out infinite',
        },
        '@keyframes backgroundMove': {
          '0%, 100%': { transform: 'translate(0, 0) scale(1)' },
          '33%': { transform: 'translate(30px, -30px) scale(1.1)' },
          '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
        },
      }}
    >
      {/* Navigation */}
      <Navigation 
        isAuthenticated={isAuthenticated}
        mobileMenuOpen={mobileMenuOpen}
        setMobileMenuOpen={setMobileMenuOpen}
        scrollToSection={scrollToSection}
        featuresRef={featuresRef}
        benefitsRef={benefitsRef}
      />
      
      {/* Hero Section */}
      <Hero isAuthenticated={isAuthenticated} />
      
      {/* Statistics Section */}
      <Statistics stats={stats} statsVisible={statsVisible} />
      
      {/* Features Section */}
      <Features features={features} featuresRef={featuresRef} />
      
      {/* Innovations Section */}
      <Innovations innovations={innovations} />
      
      {/* Benefits Section */}
      <Benefits benefits={benefits} benefitsRef={benefitsRef} />
      
      {/* System Components Section */}
      <SystemComponents systemComponents={systemComponents} />
      
      {/* Footer */}
      <Footer />
      
      {/* Scroll to Top Button */}
      <ScrollToTop />
    </Box>
  );
};

export default Landing;
