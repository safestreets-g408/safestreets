import React, { useState } from 'react';
import { useTheme } from '@mui/material/styles';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  alpha,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Zoom,
  Chip,
} from '@mui/material';
import { CheckCircle, Code, Api, Smartphone, Psychology } from '@mui/icons-material';
import { motion } from 'framer-motion';

const ComponentCard = ({ component, index }) => {
  const theme = useTheme();
  const [isHovered, setIsHovered] = useState(false);
  
  // Define tech stack colors
  const getTechColors = () => ({
    mobile: '#61DAFB', // React Native blue
    ai: '#FF6B35', // AI orange
    backend: '#68A063', // Node.js green
    frontend: '#61DAFB', // React blue
  });
  
  const techColors = getTechColors();
  const techColor = techColors[component.type] || theme.palette.primary.main;
  
  return (
    <Zoom in={true} style={{ transitionDelay: `${index * 150}ms` }}>
      <Card
        component={motion.div}
        whileHover={{ scale: 1.03, rotateY: 3 }}
        elevation={0}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        sx={{
          height: '100%',
          borderRadius: '20px',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          background: theme.palette.mode === 'dark'
            ? `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)}, ${alpha(techColor, 0.05)})`
            : `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.8)}, ${alpha(techColor, 0.03)})`,
          border: theme.palette.mode === 'dark'
            ? `1px solid ${alpha(techColor, 0.3)}`
            : `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          backdropFilter: 'blur(10px)',
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            boxShadow: theme.palette.mode === 'dark'
              ? `0 20px 50px ${alpha(techColor, 0.3)}`
              : `0 20px 50px ${alpha(theme.palette.common.black, 0.12)}`,
            transform: 'translateY(-10px)',
            borderColor: alpha(techColor, 0.5),
            '& .component-icon': {
              transform: 'scale(1.2) rotate(10deg)',
              color: techColor,
            },
            '& .tech-chip': {
              transform: 'scale(1.05)',
              backgroundColor: alpha(techColor, 0.2),
            },
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: '-50%',
            left: '-50%',
            width: '200%',
            height: '200%',
            background: `conic-gradient(from 0deg, transparent, ${alpha(techColor, 0.1)}, transparent)`,
            animation: isHovered ? 'rotate 3s linear infinite' : 'none',
          },
          '@keyframes rotate': {
            '0%': { transform: 'rotate(0deg)' },
            '100%': { transform: 'rotate(360deg)' },
          },
        }}
      >
        <CardContent sx={{ p: 4, position: 'relative', zIndex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box 
                className="component-icon"
                sx={{ 
                  color: techColor,
                  mr: 2,
                  transition: 'all 0.3s ease',
                  filter: `drop-shadow(0 4px 8px ${alpha(techColor, 0.3)})`,
                }}
              >
                {component.icon}
              </Box>
              
              <Typography
                variant="h6"
                fontWeight={700}
                sx={{
                  background: theme.palette.mode === 'dark'
                    ? `linear-gradient(135deg, ${theme.palette.text.primary}, ${techColor})`
                    : 'inherit',
                  backgroundClip: theme.palette.mode === 'dark' ? 'text' : 'inherit',
                  WebkitBackgroundClip: theme.palette.mode === 'dark' ? 'text' : 'inherit',
                  WebkitTextFillColor: theme.palette.mode === 'dark' ? 'transparent' : 'inherit',
                }}
              >
                {component.title}
              </Typography>
            </Box>
            
            <Chip
              className="tech-chip"
              label={component.type}
              size="small"
              sx={{
                backgroundColor: alpha(techColor, 0.1),
                color: techColor,
                fontWeight: 600,
                fontSize: '0.7rem',
                transition: 'all 0.3s ease',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            />
          </Box>
          
          <Typography
            variant="body2"
            color="text.secondary"
            gutterBottom
            sx={{ mb: 3, lineHeight: 1.6 }}
          >
            {component.description}
          </Typography>
          
          <List disablePadding>
            {component.features.map((feature, i) => (
              <ListItem
                key={i}
                disableGutters
                disablePadding
                sx={{ 
                  py: 0.5,
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: 'translateX(8px)',
                    '& .feature-icon': {
                      color: techColor,
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 30 }}>
                  <CheckCircle 
                    className="feature-icon"
                    sx={{ 
                      fontSize: 18, 
                      color: 'primary.main',
                      transition: 'color 0.2s ease',
                    }} 
                  />
                </ListItemIcon>
                <ListItemText
                  primary={feature}
                  primaryTypographyProps={{
                    variant: 'body2',
                    fontWeight: 500,
                  }}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>
    </Zoom>
  );
};

const SystemComponents = ({ systemComponents }) => {
  return (
    <Box
      sx={{
        py: { xs: 8, md: 12 },
        position: 'relative',
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ mb: { xs: 6, md: 8 }, textAlign: 'center' }}>
          <Typography
            variant="h2"
            fontWeight={800}
            gutterBottom
            sx={{
              mb: 2,
              fontSize: { xs: '2rem', md: '2.5rem' },
            }}
          >
            System Architecture
          </Typography>
          
          <Typography
            variant="h6"
            color="text.secondary"
            sx={{
              fontWeight: 500,
              maxWidth: '800px',
              mx: 'auto',
              px: { xs: 2, md: 0 },
              lineHeight: 1.6,
            }}
          >
            SafeStreets is built with a modular, scalable architecture using modern technologies
            designed for reliability and performance.
          </Typography>
        </Box>
        
        <Grid container spacing={4}>
          {systemComponents.map((component, index) => (
            <Grid item xs={12} sm={6} key={index}>
              <ComponentCard component={component} index={index} />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default SystemComponents;
