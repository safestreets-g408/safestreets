import React from 'react';
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
} from '@mui/material';

const FeatureCard = ({ feature, index }) => {
  const theme = useTheme();
  
  return (
    <Grow in={true} timeout={500 + (index * 100)}>
      <Card
        elevation={0}
        sx={{
          height: '100%',
          borderRadius: '20px',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          border: theme.palette.mode === 'dark'
            ? `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
            : `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          background: theme.palette.mode === 'dark'
            ? `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.8)}, ${alpha(theme.palette.primary.dark, 0.05)})`
            : 'background.paper',
          backdropFilter: 'blur(10px)',
          '&:hover': {
            boxShadow: theme.palette.mode === 'dark'
              ? `0 15px 50px ${alpha(feature.color || theme.palette.primary.main, 0.3)}`
              : `0 15px 50px ${alpha(theme.palette.common.black, 0.12)}`,
            transform: 'translateY(-8px) scale(1.02)',
            borderColor: alpha(feature.color || theme.palette.primary.main, 0.4),
            '& .feature-icon': {
              transform: 'scale(1.1) rotate(5deg)',
              boxShadow: `0 15px 30px ${alpha(feature.color || theme.palette.primary.main, 0.4)}`,
            },
            '& .feature-title': {
              color: feature.color || theme.palette.primary.main,
            },
          },
          overflow: 'visible',
          position: 'relative'
        }}
      >
        <CardContent sx={{ p: 4 }}>
          <Box
            className="feature-icon"
            sx={{
              width: 70,
              height: 70,
              borderRadius: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 3,
              color: 'white',
              background: `linear-gradient(135deg, ${feature.color || theme.palette.primary.main}, ${alpha(feature.color || theme.palette.primary.main, 0.8)})`,
              boxShadow: `0 12px 24px ${alpha(feature.color || theme.palette.primary.main, 0.3)}`,
              position: 'relative',
              transition: 'all 0.3s ease',
              '&::before': {
                content: '""',
                position: 'absolute',
                inset: 0,
                borderRadius: '16px',
                background: `linear-gradient(135deg, ${alpha(feature.color || theme.palette.primary.main, 0.2)}, transparent)`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::before': {
                opacity: 1,
              },
            }}
          >
            {feature.icon}
          </Box>
          
          <Typography
            className="feature-title"
            variant="h6"
            fontWeight={700}
            gutterBottom
            sx={{ 
              mb: 2,
              transition: 'color 0.3s ease',
            }}
          >
            {feature.title}
          </Typography>
          
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ 
              lineHeight: 1.7,
              fontSize: '0.95rem',
            }}
          >
            {feature.description}
          </Typography>
        </CardContent>
      </Card>
    </Grow>
  );
};

const Features = ({ features, featuresRef }) => {
  return (
    <Box
      ref={featuresRef}
      sx={{
        py: { xs: 10, md: 14 },
        background: (theme) => theme.palette.mode === 'dark'
          ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.8)}, ${alpha(theme.palette.primary.dark, 0.1)})`
          : `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.04)}, ${alpha(theme.palette.secondary.light, 0.02)})`,
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: (theme) => theme.palette.mode === 'dark'
            ? `radial-gradient(circle at 30% 50%, ${alpha(theme.palette.primary.main, 0.1)}, transparent)`
            : 'none',
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ mb: { xs: 8, md: 10 }, textAlign: 'center', position: 'relative', zIndex: 1 }}>
          <Typography
            variant="h2"
            fontWeight={800}
            gutterBottom
            sx={{
              mb: 3,
              fontSize: { xs: '2.25rem', md: '2.75rem' },
              background: (theme) => theme.palette.mode === 'dark'
                ? `linear-gradient(135deg, ${theme.palette.text.primary}, ${theme.palette.primary.light})`
                : `linear-gradient(135deg, ${theme.palette.text.primary}, ${theme.palette.primary.main})`,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Powerful AI-Driven Features
          </Typography>
          
          <Typography
            variant="h6"
            color="text.secondary"
            sx={{
              fontWeight: 500,
              maxWidth: '800px',
              mx: 'auto',
              px: { xs: 2, md: 0 },
              lineHeight: 1.7,
              fontSize: '1.1rem',
            }}
          >
            Our system combines cutting-edge computer vision with intelligent automation
            to revolutionize road maintenance workflows.
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <FeatureCard feature={feature} index={index} />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Features;
