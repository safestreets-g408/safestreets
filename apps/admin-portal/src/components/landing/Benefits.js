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
  Grow,
  LinearProgress,
} from '@mui/material';
import { motion } from 'framer-motion';
import { TrendingUp } from '@mui/icons-material';

const BenefitCard = ({ benefit, index }) => {
  const theme = useTheme();
  const [isHovered, setIsHovered] = useState(false);
  
  // Extract numeric value for progress bar
  const numericValue = parseInt(benefit.value.replace('%', '').replace('+', ''));
  
  return (
    <Grow in={true} timeout={400 + (index * 200)}>
      <Card
        component={motion.div}
        whileHover={{ scale: 1.05, rotateY: 5 }}
        elevation={0}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        sx={{
          height: '100%',
          borderRadius: '20px',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          background: theme.palette.mode === 'dark'
            ? `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)}, ${alpha(theme.palette.primary.dark, 0.1)})`
            : `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)}, ${alpha(theme.palette.primary.light, 0.05)})`,
          border: theme.palette.mode === 'dark'
            ? `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
            : `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          backdropFilter: 'blur(10px)',
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            boxShadow: theme.palette.mode === 'dark'
              ? `0 20px 50px ${alpha(theme.palette.primary.main, 0.3)}`
              : `0 20px 50px ${alpha(theme.palette.common.black, 0.15)}`,
            transform: 'translateY(-10px)',
            borderColor: alpha(theme.palette.primary.main, 0.4),
            '& .benefit-icon': {
              transform: 'scale(1.2) rotate(10deg)',
            },
            '& .progress-bar': {
              transform: 'scaleX(1)',
            },
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: `linear-gradient(90deg, transparent, ${alpha(theme.palette.primary.main, 0.1)}, transparent)`,
            transition: 'left 0.6s ease',
          },
          '&:hover::before': {
            left: '100%',
          },
        }}
      >
        <CardContent sx={{ p: 4, textAlign: 'center', position: 'relative', zIndex: 1 }}>
          <Box className="benefit-icon" sx={{ mb: 2 }}>
            <TrendingUp 
              sx={{ 
                fontSize: 48, 
                color: 'primary.main',
                transition: 'all 0.3s ease',
              }} 
            />
          </Box>
          
          <Typography
            variant="h2"
            fontWeight={800}
            gutterBottom
            sx={{
              mb: 2,
              fontSize: { xs: '2.5rem', md: '3rem' },
              background: theme.palette.mode === 'dark'
                ? `linear-gradient(135deg, ${theme.palette.primary.light}, ${theme.palette.primary.main})`
                : `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              position: 'relative',
            }}
          >
            {benefit.value}
          </Typography>
          
          {/* Progress bar */}
          <Box sx={{ mb: 2, px: 2 }}>
            <LinearProgress
              className="progress-bar"
              variant="determinate"
              value={isHovered ? numericValue : 0}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                transition: 'all 0.8s ease',
                transform: 'scaleX(0)',
                transformOrigin: 'left',
                '& .MuiLinearProgress-bar': {
                  background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                  borderRadius: 3,
                },
              }}
            />
          </Box>
          
          <Typography
            variant="h6"
            fontWeight={700}
            sx={{ 
              mb: 2,
              background: theme.palette.mode === 'dark'
                ? `linear-gradient(135deg, ${theme.palette.text.primary}, ${theme.palette.primary.light})`
                : 'inherit',
              backgroundClip: theme.palette.mode === 'dark' ? 'text' : 'inherit',
              WebkitBackgroundClip: theme.palette.mode === 'dark' ? 'text' : 'inherit',
              WebkitTextFillColor: theme.palette.mode === 'dark' ? 'transparent' : 'inherit',
            }}
          >
            {benefit.title}
          </Typography>
          
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ lineHeight: 1.6 }}
          >
            {benefit.description}
          </Typography>
        </CardContent>
      </Card>
    </Grow>
  );
};

const Benefits = ({ benefits, benefitsRef }) => {
  const theme = useTheme();
  
  return (
    <Box
      ref={benefitsRef}
      sx={{
        py: { xs: 12, md: 16 },
        position: 'relative',
        background: theme.palette.mode === 'dark'
          ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.8)}, ${alpha(theme.palette.primary.dark, 0.1)})`
          : `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.04)}, ${alpha(theme.palette.background.paper, 0.8)})`,
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: '20%',
          right: '-10%',
          width: '30%',
          height: '60%',
          background: `radial-gradient(ellipse, ${alpha(theme.palette.primary.main, 0.15)}, transparent)`,
          borderRadius: '50%',
          animation: 'pulse 4s ease-in-out infinite',
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          bottom: '20%',
          left: '-10%',
          width: '30%',
          height: '60%',
          background: `radial-gradient(ellipse, ${alpha(theme.palette.secondary.main, 0.1)}, transparent)`,
          borderRadius: '50%',
          animation: 'pulse 4s ease-in-out infinite reverse',
        },
        '@keyframes pulse': {
          '0%, 100%': { opacity: 0.5, transform: 'scale(1)' },
          '50%': { opacity: 1, transform: 'scale(1.1)' },
        },
      }}
    >
      {/* Background decoration */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          width: { xs: '80%', md: '40%' },
          height: '100%',
          background: `radial-gradient(ellipse at right, ${alpha(theme.palette.primary.main, 0.2)}, transparent 70%)`,
          zIndex: 0,
        }}
      />
      
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
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
            Measurable Benefits & ROI
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
            SafeStreets delivers quantifiable improvements to road maintenance operations,
            with clear return on investment across key performance indicators.
          </Typography>
        </Box>
        
        <Grid container spacing={4}>
          {benefits.map((benefit, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <BenefitCard benefit={benefit} index={index} />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Benefits;
