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

const BenefitCard = ({ benefit, index }) => {
  const theme = useTheme();
  
  return (
    <Grow in={true} timeout={300 + (index * 150)}>
      <Card
        elevation={0}
        sx={{
          height: '100%',
          borderRadius: '16px',
          transition: 'all 0.3s ease',
          bgcolor: alpha(theme.palette.background.paper, 0.9),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          '&:hover': {
            boxShadow: `0 10px 30px ${alpha(theme.palette.common.black, 0.1)}`,
            transform: 'translateY(-6px)',
          },
        }}
      >
        <CardContent sx={{ p: 3, textAlign: 'center' }}>
          <Typography
            variant="h2"
            fontWeight={800}
            gutterBottom
            sx={{
              color: 'primary.main',
              mb: 1,
              background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            {benefit.value}
          </Typography>
          
          <Typography
            variant="h6"
            fontWeight={700}
            sx={{ mb: 1 }}
          >
            {benefit.title}
          </Typography>
          
          <Typography
            variant="body2"
            color="text.secondary"
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
        py: { xs: 8, md: 12 },
        position: 'relative',
        bgcolor: alpha(theme.palette.primary.main, 0.04),
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
