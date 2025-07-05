import React from 'react';
import { Box } from '@mui/material';
import { ArrowForward } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

const ScrollToTop = () => {
  const theme = useTheme();
  
  return (
    <Box
      onClick={() => window.scrollTo({top: 0, behavior: 'smooth'})}
      role="presentation"
      sx={{
        position: 'fixed',
        bottom: 16,
        right: 16,
        zIndex: 1000,
        bgcolor: theme.palette.primary.main,
        color: 'white',
        borderRadius: '50%',
        p: 1,
        display: 'flex',
        boxShadow: 4,
        cursor: 'pointer',
        '&:hover': {
          bgcolor: theme.palette.primary.dark,
        }
      }}
    >
      <ArrowForward sx={{ transform: 'rotate(-90deg)' }} />
    </Box>
  );
};

export default ScrollToTop;
