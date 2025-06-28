import React, { createContext, useState, useContext, useEffect } from 'react';
import { useColorScheme, Platform } from 'react-native';
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
  
  // For debugging
  useEffect(() => {
    try {
      console.log('Device color scheme:', deviceColorScheme || 'not available');
    } catch (error) {
      console.error('Error logging color scheme:', error);
    }
  }, [deviceColorScheme]);
  
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
      // If system preference is set, use it ('dark' or 'light')
      // If no system preference (null/undefined), default to light
      return deviceColorScheme === 'dark' ? THEME_MODE.DARK : THEME_MODE.LIGHT;
    }
    return themeMode;
  }, [themeMode, deviceColorScheme]);

  // Get the appropriate theme based on active theme mode - memoized for performance
  const activeTheme = React.useMemo(() => {
    const active = getActiveThemeMode();
    return active === THEME_MODE.DARK ? darkTheme : lightTheme;
  }, [getActiveThemeMode]);

  // Get the paper theme - memoized for performance
  const paperTheme = React.useMemo(() => {
    const active = getActiveThemeMode();
    return active === THEME_MODE.DARK 
      ? { ...DarkTheme, colors: { ...DarkTheme.colors, ...darkTheme.colors } }
      : { ...DefaultTheme, colors: { ...DefaultTheme.colors, ...lightTheme.colors } };
  }, [getActiveThemeMode]);

  // Handle theme mode change
  const changeThemeMode = React.useCallback((mode) => {
    try {
      console.log('Changing theme mode from', themeMode, 'to', mode);
      setThemeMode(mode);
    } catch (error) {
      console.error('Error changing theme mode:', error);
    }
  }, [themeMode]);

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
      return {
        themeMode,
        systemTheme: deviceColorScheme || 'light',
        isDarkMode: effectiveThemeMode === THEME_MODE.DARK,
        isSystemTheme: themeMode === THEME_MODE.SYSTEM,
        theme: activeTheme,
        changeThemeMode,
        getEffectiveThemeMode: getActiveThemeMode,
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
  }, [themeMode, deviceColorScheme, hasError, activeTheme, getActiveThemeMode, changeThemeMode]);

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
