import React from 'react';
import { useTheme } from '@mui/material/styles';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  alpha,
  Fade,
} from '@mui/material';
import { motion } from 'framer-motion';

const StatCard = ({ stat, delay, statsVisible }) => {
  const theme = useTheme();
  
  return (
    <Fade in={statsVisible} timeout={1000}>
      <Paper
        component={motion.div}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 + (delay * 0.1) }}
        elevation={0}
        sx={{
          p: 3,
          height: '100%',
          borderRadius: '16px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          boxShadow: `0 8px 20px ${alpha(theme.palette.common.black, 0.05)}`,
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: `0 12px 30px ${alpha(theme.palette.common.black, 0.08)}`,
            borderColor: alpha(theme.palette.primary.main, 0.3),
          },
        }}
      >
        <Box
          sx={{
            p: 1.5,
            mb: 2,
            borderRadius: '50%',
            bgcolor: alpha(theme.palette.primary.main, 0.1),
            color: 'primary.main',
          }}
        >
          {stat.icon}
        </Box>
        
        <Typography
          variant="h3"
          fontWeight={800}
          sx={{
            mb: 1,
            color: 'primary.main',
            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          {stat.value}
          <Box component="span" sx={{ fontSize: '60%', ml: 0.5 }}>
            {stat.suffix}
          </Box>
        </Typography>
        
        <Typography
          variant="subtitle1"
          fontWeight={600}
          color="text.secondary"
        >
          {stat.label}
        </Typography>
      </Paper>
    </Fade>
  );
};

const Statistics = ({ stats, statsVisible }) => {
  return (
    <Box
      sx={{
        py: { xs: 8, md: 10 },
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={3}>
          {stats.map((stat, index) => (
            <Grid item xs={6} md={3} key={index}>
              <StatCard 
                stat={stat} 
                delay={index} 
                statsVisible={statsVisible}
              />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Statistics;
