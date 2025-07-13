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
  Zoom,
  Chip,
} from '@mui/material';
import { motion } from 'framer-motion';

const InnovationCard = ({ innovation, index }) => {
  const theme = useTheme();
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <Zoom in={true} style={{ transitionDelay: `${index * 200}ms` }}>
      <Card
        component={motion.div}
        whileHover={{ scale: 1.03, rotateY: 5 }}
        elevation={0}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        sx={{
          height: '100%',
          borderRadius: '24px',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          background: theme.palette.mode === 'dark'
            ? `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)}, ${alpha(theme.palette.primary.dark, 0.1)})`
            : `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.03)}, ${alpha(theme.palette.background.paper, 0.8)})`,
          border: theme.palette.mode === 'dark'
            ? `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
            : `1px solid ${alpha(theme.palette.primary.main, 0.12)}`,
          backdropFilter: 'blur(10px)',
          '&:hover': {
            boxShadow: theme.palette.mode === 'dark'
              ? `0 20px 50px ${alpha(theme.palette.primary.main, 0.3)}`
              : `0 20px 50px ${alpha(theme.palette.common.black, 0.15)}`,
            transform: 'translateY(-10px) rotateX(5deg)',
            borderColor: alpha(theme.palette.primary.main, 0.5),
            '& .innovation-icon': {
              transform: 'scale(1.2) rotate(10deg)',
              boxShadow: `0 15px 30px ${alpha(theme.palette.primary.main, 0.5)}`,
            },
            '& .metric-chip': {
              transform: 'scale(1.1)',
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
            },
          },
          overflow: 'visible',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            borderRadius: '24px',
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, transparent)`,
            opacity: isHovered ? 1 : 0,
            transition: 'opacity 0.3s ease',
            pointerEvents: 'none',
          },
        }}
      >
        <CardContent sx={{ p: 4, position: 'relative', zIndex: 1 }}>
          <Box
            className="innovation-icon"
            sx={{
              width: 70,
              height: 70,
              borderRadius: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 3,
              color: 'white',
              background: `linear-gradient(135deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
              boxShadow: `0 12px 25px ${alpha(theme.palette.primary.main, 0.4)}`,
              transition: 'all 0.3s ease',
              position: 'relative',
              '&::before': {
                content: '""',
                position: 'absolute',
                inset: '-2px',
                borderRadius: '18px',
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                zIndex: -1,
                opacity: 0.3,
              },
            }}
          >
            {innovation.icon}
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography
              variant="h6"
              fontWeight={700}
              sx={{ 
                flex: 1,
                background: theme.palette.mode === 'dark'
                  ? `linear-gradient(135deg, ${theme.palette.text.primary}, ${theme.palette.primary.light})`
                  : 'inherit',
                backgroundClip: theme.palette.mode === 'dark' ? 'text' : 'inherit',
                WebkitBackgroundClip: theme.palette.mode === 'dark' ? 'text' : 'inherit',
                WebkitTextFillColor: theme.palette.mode === 'dark' ? 'transparent' : 'inherit',
              }}
            >
              {innovation.title}
            </Typography>
            <Chip
              className="metric-chip"
              label={innovation.metric}
              size="small"
              sx={{
                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.15)}, ${alpha(theme.palette.primary.light, 0.1)})`,
                color: 'primary.main',
                fontWeight: 700,
                fontSize: '0.75rem',
                transition: 'all 0.3s ease',
                border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
              }}
            />
          </Box>
          
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ 
              lineHeight: 1.7,
              mb: 2,
            }}
          >
            {innovation.description}
          </Typography>
          
          <Typography
            variant="caption"
            color="primary"
            sx={{ 
              fontWeight: 600,
              fontSize: '0.8rem',
            }}
          >
            {innovation.metricLabel}
          </Typography>
        </CardContent>
      </Card>
    </Zoom>
  );
};

const Innovations = ({ innovations }) => {
  const theme = useTheme();
  
  return (
    <Box
      sx={{
        py: { xs: 12, md: 16 },
        position: 'relative',
        overflow: 'hidden',
        background: theme.palette.mode === 'dark'
          ? `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.9)}, ${alpha(theme.palette.primary.dark, 0.1)}, ${alpha(theme.palette.background.default, 0.9)})`
          : `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.8)}, ${alpha(theme.palette.primary.light, 0.05)})`,
        '&::before': {
          content: '""',
          position: 'absolute',
          top: '10%',
          left: '-20%',
          width: '40%',
          height: '80%',
          background: `radial-gradient(ellipse, ${alpha(theme.palette.primary.main, 0.15)}, transparent)`,
          borderRadius: '50%',
          animation: 'float 6s ease-in-out infinite',
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          bottom: '10%',
          right: '-20%',
          width: '40%',
          height: '80%',
          background: `radial-gradient(ellipse, ${alpha(theme.palette.secondary.main, 0.1)}, transparent)`,
          borderRadius: '50%',
          animation: 'float 6s ease-in-out infinite reverse',
        },
        '@keyframes float': {
          '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
          '50%': { transform: 'translateY(-20px) rotate(5deg)' },
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ mb: { xs: 8, md: 10 }, textAlign: 'center' }}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <Typography
              variant="h2"
              fontWeight={800}
              gutterBottom
              sx={{
                mb: 3,
                fontSize: { xs: '2.25rem', md: '2.75rem' },
                background: theme.palette.mode === 'dark'
                  ? `linear-gradient(135deg, ${theme.palette.text.primary}, ${theme.palette.primary.light}, ${theme.palette.secondary.light})`
                  : `linear-gradient(135deg, ${theme.palette.text.primary}, ${theme.palette.primary.main})`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                position: 'relative',
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  bottom: '-10px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: '60px',
                  height: '4px',
                  background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                  borderRadius: '2px',
                },
              }}
            >
              Innovative AI Technology
            </Typography>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
          >
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
              Our platform leverages cutting-edge artificial intelligence to deliver
              unprecedented accuracy and efficiency in road damage management.
            </Typography>
          </motion.div>
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
