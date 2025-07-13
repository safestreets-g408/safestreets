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
          p: 4,
          height: '100%',
          borderRadius: '20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          border: theme.palette.mode === 'dark'
            ? `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
            : `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          background: theme.palette.mode === 'dark'
            ? `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.8)}, ${alpha(theme.palette.primary.dark, 0.05)})`
            : 'background.paper',
          backdropFilter: 'blur(10px)',
          boxShadow: theme.palette.mode === 'dark'
            ? `0 8px 20px ${alpha(theme.palette.common.black, 0.3)}`
            : `0 8px 20px ${alpha(theme.palette.common.black, 0.05)}`,
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-8px) scale(1.02)',
            boxShadow: theme.palette.mode === 'dark'
              ? `0 16px 40px ${alpha(theme.palette.primary.main, 0.2)}`
              : `0 16px 40px ${alpha(theme.palette.common.black, 0.1)}`,
            borderColor: alpha(theme.palette.primary.main, 0.4),
            '& .stat-icon': {
              transform: 'scale(1.15) rotate(5deg)',
              boxShadow: `0 8px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
            },
          },
        }}
      >
        <Box
          className="stat-icon"
          sx={{
            p: 2,
            mb: 3,
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.15)}, ${alpha(theme.palette.primary.dark, 0.1)})`,
            color: 'primary.main',
            border: theme.palette.mode === 'dark'
              ? `2px solid ${alpha(theme.palette.primary.main, 0.3)}`
              : 'none',
            transition: 'all 0.3s ease',
            boxShadow: `0 4px 15px ${alpha(theme.palette.primary.main, 0.2)}`,
          }}
        >
          {stat.icon}
        </Box>
        
        <Typography
          variant="h3"
          fontWeight={800}
          sx={{
            mb: 1.5,
            fontSize: { xs: '2rem', md: '2.5rem' },
            background: theme.palette.mode === 'dark'
              ? `linear-gradient(135deg, ${theme.palette.primary.light}, ${theme.palette.primary.main})`
              : `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          {stat.value}
          <Box 
            component="span" 
            sx={{ 
              fontSize: '60%', 
              ml: 0.5,
              opacity: 0.8,
            }}
          >
            {stat.suffix}
          </Box>
        </Typography>
        
        <Typography
          variant="subtitle1"
          fontWeight={600}
          color="text.secondary"
          sx={{ fontSize: '1rem' }}
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
