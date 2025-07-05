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
          borderRadius: '16px',
          transition: 'all 0.3s ease',
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          '&:hover': {
            boxShadow: `0 10px 40px ${alpha(theme.palette.common.black, 0.1)}`,
            transform: 'translateY(-6px)',
            borderColor: 'transparent'
          },
          overflow: 'visible',
          position: 'relative'
        }}
      >
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              width: 60,
              height: 60,
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 2,
              color: 'white',
              bgcolor: feature.color || theme.palette.primary.main,
              boxShadow: `0 10px 20px ${alpha(feature.color || theme.palette.primary.main, 0.3)}`,
            }}
          >
            {feature.icon}
          </Box>
          
          <Typography
            variant="h6"
            fontWeight={700}
            gutterBottom
            sx={{ mb: 1.5 }}
          >
            {feature.title}
          </Typography>
          
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ lineHeight: 1.6 }}
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
        py: { xs: 8, md: 12 },
        bgcolor: (theme) => alpha(theme.palette.primary.main, 0.04),
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
              lineHeight: 1.6,
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
