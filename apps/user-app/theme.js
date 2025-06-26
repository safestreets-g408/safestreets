import { DefaultTheme } from 'react-native-paper';

// Define application theme aligned with admin portal color scheme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    // Primary colors - from admin portal
    primary: '#2563eb',
    primaryLight: '#60a5fa',
    primaryDark: '#1d4ed8',
    
    // Secondary colors
    secondary: '#6b7280',
    secondaryLight: '#9ca3af',
    secondaryDark: '#374151',
    
    // Feedback colors
    success: '#10b981',
    successLight: '#34d399',
    successDark: '#059669',
    
    error: '#dc2626',
    errorLight: '#ef4444',
    errorDark: '#b91c1c',
    
    warning: '#f59e0b',
    warningLight: '#fbbf24',
    warningDark: '#d97706',
    
    info: '#3b82f6',
    infoLight: '#60a5fa',
    infoDark: '#2563eb',
    
    // Background colors
    background: '#f9fafb',
    surface: '#ffffff',
    surfaceVariant: '#f3f4f6',
    
    // Text colors
    text: '#111827',
    textSecondary: '#6b7280',
    placeholder: '#9ca3af',
    disabled: '#d1d5db',
    
    // Additional colors
    accent: '#3b82f6',
    card: '#ffffff',
    border: '#e5e7eb',
    
    // Gradient colors
    gradient: {
      primary: ['#2563eb', '#1d4ed8'],
      secondary: ['#6b7280', '#374151'],
      blue: ['#60a5fa', '#3b82f6'],
      purple: ['#667eea', '#764ba2'],
      success: ['#34d399', '#10b981'],
      warning: ['#fbbf24', '#f59e0b'],
      danger: ['#ef4444', '#dc2626'],
      dark: ['#1f2937', '#111827'],
      light: ['#ffffff', '#f3f4f6'],
    },
    
    // Grey palette
    grey: {
      50: '#f9fafb',
      100: '#f3f4f6',
      200: '#e5e7eb',
      300: '#d1d5db',
      400: '#9ca3af',
      500: '#6b7280',
      600: '#4b5563',
      700: '#374151',
      800: '#1f2937',
      900: '#111827',
    },
  },
  
  // Visual properties
  roundness: 12,
  animation: {
    scale: 1.0,
  },
  
  // Typography
  fonts: {
    ...DefaultTheme.fonts,
    regular: { fontFamily: 'System', fontWeight: '400' },
    medium: { fontFamily: 'System', fontWeight: '500' },
    light: { fontFamily: 'System', fontWeight: '300' },
    bold: { fontFamily: 'System', fontWeight: '700' },
  },
  
  // Shadows
  shadows: {
    small: {
      elevation: 2,
      shadowColor: 'rgba(0,0,0,0.25)',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.08,
      shadowRadius: 3,
    },
    medium: {
      elevation: 4,
      shadowColor: 'rgba(0,0,0,0.25)',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.12,
      shadowRadius: 6,
    },
    large: {
      elevation: 8,
      shadowColor: 'rgba(0,0,0,0.25)',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.2,
      shadowRadius: 12,
    }
  },
  
  // Spacing
  spacing: {
    xs: 4,
    s: 8,
    m: 16,
    l: 24,
    xl: 32,
    xxl: 48,
  },
  
  // Card styles
  cards: {
    default: {
      borderRadius: 12,
      padding: 16,
    },
    elevated: {
      borderRadius: 16,
      padding: 20,
      backgroundColor: '#ffffff',
      shadowColor: 'rgba(0,0,0,0.25)',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.2,
      shadowRadius: 12,
      elevation: 8,
    },
    interactive: {
      borderRadius: 16,
      padding: 20,
      backgroundColor: '#ffffff',
      shadowColor: 'rgba(0,0,0,0.25)',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.2,
      shadowRadius: 12,
      elevation: 8,
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    }
  }
};

export default theme;
