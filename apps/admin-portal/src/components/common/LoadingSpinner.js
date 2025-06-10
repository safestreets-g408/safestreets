import React from 'react';
import { Box, CircularProgress, Typography, useTheme, alpha } from '@mui/material';

const LoadingSpinner = ({ message = 'Loading...', size = 'default' }) => {
  const theme = useTheme();
  
  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight={size === 'small' ? '120px' : '200px'}
      sx={{
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: size === 'small' ? '80px' : '120px',
          height: size === 'small' ? '80px' : '120px',
          borderRadius: '50%',
          background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.08)}, ${alpha(theme.palette.secondary.main, 0.08)})`,
          animation: 'pulse 2s ease-in-out infinite',
        },
        '@keyframes pulse': {
          '0%': {
            transform: 'translate(-50%, -50%) scale(0.95)',
            opacity: 0.6,
          },
          '50%': {
            transform: 'translate(-50%, -50%) scale(1)',
            opacity: 0.9,
          },
          '100%': {
            transform: 'translate(-50%, -50%) scale(0.95)',
            opacity: 0.6,
          },
        },
      }}
    >
      <CircularProgress 
        size={size === 'small' ? 32 : 40}
        thickness={4}
        sx={{
          color: theme.palette.primary.main,
          '& .MuiCircularProgress-circle': {
            strokeLinecap: 'round',
          },
        }}
      />
      {message && (
        <Typography
          variant="body2"
          sx={{ 
            mt: 2,
            color: alpha(theme.palette.text.primary, 0.7),
            fontWeight: 500,
            letterSpacing: '0.2px',
          }}
        >
          {message}
        </Typography>
      )}
    </Box>
  );
};

export default LoadingSpinner; 