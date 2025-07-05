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
  Zoom,
} from '@mui/material';

const InnovationCard = ({ innovation, index }) => {
  const theme = useTheme();
  
  return (
    <Zoom in={true} style={{ transitionDelay: `${index * 100}ms` }}>
      <Card
        elevation={0}
        sx={{
          height: '100%',
          borderRadius: '16px',
          transition: 'all 0.3s ease',
          bgcolor: alpha(theme.palette.primary.main, 0.03),
          border: `1px solid ${alpha(theme.palette.primary.main, 0.12)}`,
          '&:hover': {
            boxShadow: `0 10px 30px ${alpha(theme.palette.common.black, 0.1)}`,
            transform: 'translateY(-6px)',
            borderColor: alpha(theme.palette.primary.main, 0.3),
          },
          overflow: 'visible',
          position: 'relative',
        }}
      >
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              width: 50,
              height: 50,
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 2,
              color: 'white',
              background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
              boxShadow: `0 8px 20px ${alpha(theme.palette.primary.main, 0.3)}`,
            }}
          >
            {innovation.icon}
          </Box>
          
          <Typography
            variant="h6"
            fontWeight={700}
            gutterBottom
            sx={{ mb: 1 }}
          >
            {innovation.title}
          </Typography>
          
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ lineHeight: 1.6, mb: 2.5 }}
          >
            {innovation.description}
          </Typography>
          
          <Box
            sx={{
              p: 1.5,
              borderRadius: '10px',
              bgcolor: alpha(theme.palette.primary.main, 0.08),
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Typography
              variant="h5"
              fontWeight={800}
              color="primary.main"
            >
              {innovation.metric}
            </Typography>
            
            <Typography
              variant="caption"
              fontWeight={600}
              color="text.secondary"
              sx={{ lineHeight: 1.2 }}
            >
              {innovation.metricLabel}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Zoom>
  );
};

const Innovations = ({ innovations }) => {
  return (
    <Box
      sx={{
        py: { xs: 8, md: 12 },
        position: 'relative',
        overflow: 'hidden',
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
            Innovative AI Technology
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
            Our platform leverages cutting-edge artificial intelligence to deliver
            unprecedented accuracy and efficiency in road damage management.
          </Typography>
        </Box>
        
        <Grid container spacing={4}>
          {innovations.map((innovation, index) => (
            <Grid item xs={12} md={4} key={index}>
              <InnovationCard innovation={innovation} index={index} />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Innovations;
