import React from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material';
import { useThemeContext } from '../context/ThemeContext';
import { createAppTheme } from '../theme';

// ThemeWrapper applies the current theme mode (light/dark)
const ThemeWrapper = ({ children }) => {
  const { darkMode } = useThemeContext();
  
  // Create theme based on current mode
  const currentTheme = React.useMemo(() => {
    return createAppTheme(darkMode ? 'dark' : 'light');
  }, [darkMode]);

  return (
    <MuiThemeProvider theme={currentTheme}>
      {children}
    </MuiThemeProvider>
  );
};

export default ThemeWrapper;
