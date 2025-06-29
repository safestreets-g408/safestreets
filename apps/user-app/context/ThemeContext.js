import React, { createContext, useState, useContext, useEffect } from 'react';
import { useColorScheme, Platform, Appearance } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { DefaultTheme, DarkTheme, Provider as PaperProvider } from 'react-native-paper';
import { lightTheme, darkTheme } from '../theme';

// Theme types
export const THEME_MODE = {
  LIGHT: 'light',
  DARK: 'dark',
  SYSTEM: 'system'
};

// Create context
const ThemeContext = createContext();

// Theme Provider component
export const ThemeProvider = ({ children }) => {
  // Get the device color scheme, and update whenever it changes
  const deviceColorScheme = useColorScheme();
  const [themeMode, setThemeMode] = useState(THEME_MODE.SYSTEM);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [systemThemeDetected, setSystemThemeDetected] = useState(null);
  
  // Enhanced system theme detection
  useEffect(() => {
    const detectSystemTheme = () => {
      try {
        // Enhanced detection for both platforms
        let detectedTheme = 'light'; // default fallback
        
        if (Platform.OS === 'ios') {
          // iOS specific detection using useColorScheme
          detectedTheme = deviceColorScheme || 'light';
          console.log('iOS system theme detected:', detectedTheme);
        } else if (Platform.OS === 'android') {
          // Android specific detection using useColorScheme
          detectedTheme = deviceColorScheme || 'light';
          console.log('Android system theme detected:', detectedTheme);
        } else {
          // Web or other platforms
          detectedTheme = deviceColorScheme || 'light';
          console.log('Other platform theme detected:', detectedTheme);
        }
        
        setSystemThemeDetected(detectedTheme);
        console.log('Final detected system theme:', detectedTheme);
        
        return detectedTheme;
      } catch (error) {
        console.error('Error detecting system theme:', error);
        const fallback = 'light';
        setSystemThemeDetected(fallback);
        return fallback;
      }
    };
    
    // Initial detection
    detectSystemTheme();
    
    // Listen for actual system theme changes using Appearance API on supported platforms
    let appearanceListener = null;
    
    if (Platform.OS === 'ios' || Platform.OS === 'android') {
      // Use React Native's built-in listener for system theme changes
      try {
        const { Appearance } = require('react-native');
        if (Appearance && Appearance.addChangeListener) {
          appearanceListener = Appearance.addChangeListener(({ colorScheme }) => {
            console.log('System theme changed via Appearance API:', colorScheme);
            setSystemThemeDetected(colorScheme || 'light');
          });
        }
      } catch (error) {
        console.warn('Appearance API not available, falling back to polling:', error);
      }
    }
    
    // Fallback: Poll for changes (as backup mechanism)
    const intervalId = setInterval(() => {
      const currentTheme = detectSystemTheme();
      if (currentTheme !== systemThemeDetected) {
        console.log('System theme changed from', systemThemeDetected, 'to', currentTheme);
      }
    }, 2000); // Check every 2 seconds as fallback
    
    return () => {
      // Clean up listeners
      if (appearanceListener) {
        try {
          const { Appearance } = require('react-native');
          if (Appearance && Appearance.removeChangeListener) {
            Appearance.removeChangeListener(appearanceListener);
          }
        } catch (error) {
          console.warn('Error removing Appearance listener:', error);
        }
      }
      clearInterval(intervalId);
    };
  }, [deviceColorScheme, systemThemeDetected]);
  
  // For debugging - log theme changes
  useEffect(() => {
    try {
      console.log('Device color scheme updated:', deviceColorScheme || 'not available');
      console.log('Platform:', Platform.OS);
      console.log('System theme detected:', systemThemeDetected);
    } catch (error) {
      console.error('Error logging color scheme:', error);
    }
  }, [deviceColorScheme, systemThemeDetected]);
  
  // Basic error detection - React Native doesn't have window.addEventListener
  useEffect(() => {
    // Do a simple check on startup to ensure theme functionality is working
    try {
      // Try to use the device color scheme as a basic check
      if (deviceColorScheme === null && Platform.OS !== 'web') {
        console.warn('deviceColorScheme is null, might be an issue with appearance detection');
      }
    } catch (error) {
      console.error('Theme detection error:', error);
      setHasError(true);
    }
  }, [deviceColorScheme]);

  // Load saved theme preference
  useEffect(() => {
    const loadThemePreference = async () => {
      try {
        const savedThemeMode = await AsyncStorage.getItem('themeMode');
        if (savedThemeMode) {
          setThemeMode(savedThemeMode);
        } else {
          // Explicitly set to system if no preference exists
          setThemeMode(THEME_MODE.SYSTEM);
          // Save this preference
          await AsyncStorage.setItem('themeMode', THEME_MODE.SYSTEM);
        }
      } catch (error) {
        console.error('Error loading theme preference:', error);
        // On error, make sure we default to system
        setThemeMode(THEME_MODE.SYSTEM);
      } finally {
        setIsLoading(false);
      }
    };

    loadThemePreference();
  }, []);

  // Save theme preference when it changes
  useEffect(() => {
    if (!isLoading) {
      const saveThemePreference = async () => {
        try {
          await AsyncStorage.setItem('themeMode', themeMode);
        } catch (error) {
          console.error('Error saving theme preference:', error);
        }
      };
      
      saveThemePreference();
    }
  }, [themeMode, isLoading]);

  // Determine the actual theme to use based on settings and system theme
  const getActiveThemeMode = React.useCallback(() => {
    if (themeMode === THEME_MODE.SYSTEM) {
      // Use our enhanced system theme detection
      const currentSystemTheme = systemThemeDetected || deviceColorScheme || 'light';
      console.log('Using system theme:', currentSystemTheme);
      return currentSystemTheme === 'dark' ? THEME_MODE.DARK : THEME_MODE.LIGHT;
    }
    return themeMode;
  }, [themeMode, deviceColorScheme, systemThemeDetected]);

  // Get the appropriate theme based on active theme mode - memoized for performance
  const activeTheme = React.useMemo(() => {
    try {
      const active = getActiveThemeMode();
      console.log('Active theme mode:', active);
      const selectedTheme = active === THEME_MODE.DARK ? darkTheme : lightTheme;
      console.log('Selected theme background:', selectedTheme?.colors?.background);
      return selectedTheme;
    } catch (error) {
      console.error('Error in activeTheme memo:', error);
      return lightTheme;
    }
  }, [getActiveThemeMode]);

  // Get the paper theme - memoized for performance
  const paperTheme = React.useMemo(() => {
    try {
      const active = getActiveThemeMode();
      
      // Ensure we have valid base themes
      const baseDefaultTheme = DefaultTheme || { colors: {} };
      const baseDarkTheme = DarkTheme || { colors: {} };
      const safeDefaultColors = baseDefaultTheme.colors || {};
      const safeDarkColors = baseDarkTheme.colors || {};
      
      const theme = active === THEME_MODE.DARK 
        ? { ...baseDarkTheme, colors: { ...safeDarkColors, ...darkTheme.colors } }
        : { ...baseDefaultTheme, colors: { ...safeDefaultColors, ...lightTheme.colors } };
      console.log('Paper theme background:', theme?.colors?.background);
      return theme;
    } catch (error) {
      console.error('Error in paperTheme memo:', error);
      // Extra safe fallback
      return { 
        colors: { 
          ...lightTheme.colors 
        } 
      };
    }
  }, [getActiveThemeMode]);

  // Handle theme mode change
  const changeThemeMode = React.useCallback((mode) => {
    try {
      console.log('Changing theme mode from', themeMode, 'to', mode);
      console.log('Device color scheme:', deviceColorScheme);
      setThemeMode(mode);
      // Force a re-render to ensure the theme change takes effect
      console.log('Theme mode changed successfully to:', mode);
    } catch (error) {
      console.error('Error changing theme mode:', error);
    }
  }, [themeMode, deviceColorScheme]);

  // Create context value safely with useMemo to prevent recreation
  const contextValue = React.useMemo(() => {
    try {
      // If there was an error, provide safe fallback values
      if (hasError) {
        return {
          themeMode: THEME_MODE.LIGHT,
          systemTheme: 'light',
          isDarkMode: false,
          isSystemTheme: false,
          theme: lightTheme,
          changeThemeMode: () => {},
          getEffectiveThemeMode: () => THEME_MODE.LIGHT,
        };
      }
      
      // Normal values with safe access
      const effectiveThemeMode = getActiveThemeMode();
      const detectedSystemTheme = systemThemeDetected || deviceColorScheme || 'light';
      
      return {
        themeMode,
        systemTheme: detectedSystemTheme,
        isDarkMode: effectiveThemeMode === THEME_MODE.DARK,
        isSystemTheme: themeMode === THEME_MODE.SYSTEM,
        theme: activeTheme,
        changeThemeMode,
        getEffectiveThemeMode: getActiveThemeMode,
        // Additional info for debugging
        deviceColorScheme,
        platformInfo: {
          os: Platform.OS,
          systemThemeDetected,
        }
      };
    } catch (error) {
      console.error('Error creating context value:', error);
      // Ultra-safe fallback
      return {
        themeMode: THEME_MODE.LIGHT,
        systemTheme: 'light',
        isDarkMode: false,
        isSystemTheme: false,
        theme: lightTheme,
        changeThemeMode: () => {},
        getEffectiveThemeMode: () => THEME_MODE.LIGHT,
      };
    }
  }, [themeMode, deviceColorScheme, hasError, activeTheme, getActiveThemeMode, changeThemeMode, systemThemeDetected]);

  return (
    <ThemeContext.Provider value={contextValue}>
      <PaperProvider theme={paperTheme}>
        {children}
      </PaperProvider>
    </ThemeContext.Provider>
  );
};

// Custom hook for using the theme context
export const useThemeContext = () => {
  try {
    const context = useContext(ThemeContext);
    if (!context) {
      console.warn('useThemeContext used outside ThemeProvider, using fallbacks');
      // Instead of throwing, return a fallback context
      return {
        themeMode: THEME_MODE.SYSTEM,
        systemTheme: 'light',
        isDarkMode: false,
        isSystemTheme: true,
        theme: lightTheme,
        changeThemeMode: () => {},
        getEffectiveThemeMode: () => THEME_MODE.LIGHT,
      };
    }
    return context;
  } catch (error) {
    console.error('Error in useThemeContext:', error);
    // Return fallback on error
    return {
      themeMode: THEME_MODE.SYSTEM,
      systemTheme: 'light',
      isDarkMode: false,
      isSystemTheme: true,
      theme: lightTheme, 
      changeThemeMode: () => {},
      getEffectiveThemeMode: () => THEME_MODE.LIGHT,
    };
  }
};

export default ThemeContext;
