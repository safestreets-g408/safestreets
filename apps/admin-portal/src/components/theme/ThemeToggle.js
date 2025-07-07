import React from 'react';
import { IconButton, Tooltip, useTheme } from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { useThemeContext } from '../../context/ThemeContext';

const ThemeToggle = ({ color = 'inherit', size = 'medium' }) => {
  const theme = useTheme();
  const { darkMode, toggleDarkMode } = useThemeContext();

  return (
    <Tooltip title={darkMode ? "Switch to light mode" : "Switch to dark mode"}>
      <IconButton
        onClick={toggleDarkMode}
        color={color}
        size={size}
        aria-label="toggle dark/light mode"
        sx={{ 
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'rotate(30deg)',
          }
        }}
      >
        {darkMode ? (
          <Brightness7Icon />
        ) : (
          <Brightness4Icon />
        )}
      </IconButton>
    </Tooltip>
  );
};

export default ThemeToggle;
