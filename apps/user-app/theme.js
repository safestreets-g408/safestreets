import { DefaultTheme } from 'react-native-paper';

// Common theme components
const commonThemeProps = {
  roundness: 12,
  animation: {
    scale: 1.0,
  },
  fonts: {
    ...DefaultTheme.fonts,
    regular: { fontFamily: 'System', fontWeight: '400' },
    medium: { fontFamily: 'System', fontWeight: '500' },
    light: { fontFamily: 'System', fontWeight: '300' },
    bold: { fontFamily: 'System', fontWeight: '700' },
  },
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
  spacing: {
    xs: 4,
    s: 8,
    m: 16,
    l: 24,
    xl: 32,
    xxl: 48,
  },
};

// Light theme aligned with admin portal color scheme
export const lightTheme = {
  ...DefaultTheme,
  ...commonThemeProps,
  colors: {
    ...DefaultTheme.colors,
    // Primary colors - exactly matching admin portal
    primary: '#2563eb',
    primaryLight: '#60a5fa',
    primaryDark: '#1d4ed8',
    onPrimary: '#ffffff',
    
    // Secondary colors
    secondary: '#6b7280',
    secondaryLight: '#9ca3af',
    secondaryDark: '#374151',
    onSecondary: '#ffffff',
    
    // Feedback colors
    success: '#10b981',
    successLight: '#34d399',
    successDark: '#059669',
    onSuccess: '#ffffff',
    
    error: '#dc2626',
    errorLight: '#ef4444',
    errorDark: '#b91c1c',
    onError: '#ffffff',
    
    warning: '#f59e0b',
    warningLight: '#fbbf24',
    warningDark: '#d97706',
    onWarning: '#000000',
    
    info: '#3b82f6',
    infoLight: '#60a5fa',
    infoDark: '#2563eb',
    onInfo: '#ffffff',
    
    // Background colors - matching admin portal
    background: '#f9fafb',
    surface: '#ffffff',
    surfaceVariant: '#f3f4f6',
    surfaceDisabled: '#e5e7eb',
    
    // Text colors - matching admin portal
    text: '#111827',
    textSecondary: '#6b7280',
    placeholder: '#9ca3af',
    disabled: '#d1d5db',
    onSurface: '#111827',
    onSurfaceVariant: '#6b7280',
    
    // Card and border
    card: '#ffffff',
    cardShadow: 'rgba(0,0,0,0.1)',
    border: '#e5e7eb',
    outline: '#d1d5db',
    
    // Status colors
    pending: '#f59e0b',
    assigned: '#3b82f6',
    inProgress: '#f59e0b',
    completed: '#10b981',
    rejected: '#dc2626',
    
    // Gradient colors
    gradient: {
      primary: ['#2563eb', '#1d4ed8'],
      secondary: ['#6b7280', '#374151'],
      success: ['#34d399', '#10b981'],
      warning: ['#fbbf24', '#f59e0b'],
      danger: ['#ef4444', '#dc2626'],
    },
    
    // Elevation
    elevation: {
      level0: 'transparent',
      level1: '#ffffff',
      level2: '#f9fafb',
      level3: '#f3f4f6',
      level4: '#f3f4f6',
      level5: '#f1f5f9',
    },
  }
};

// Dark theme with modern, attractive dark colors
export const darkTheme = {
  ...DefaultTheme,
  ...commonThemeProps,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    // Primary colors - brighter and more vibrant for dark mode
    primary: '#4f46e5',
    primaryLight: '#6366f1',
    primaryDark: '#3730a3',
    onPrimary: '#ffffff',
    
    // Secondary colors - warmer grays
    secondary: '#a1a1aa',
    secondaryLight: '#d4d4d8',
    secondaryDark: '#71717a',
    onSecondary: '#ffffff',
    
    // Feedback colors - more vibrant
    success: '#22c55e',
    successLight: '#4ade80',
    successDark: '#16a34a',
    onSuccess: '#ffffff',
    
    error: '#ef4444',
    errorLight: '#f87171',
    errorDark: '#dc2626',
    onError: '#ffffff',
    
    warning: '#f59e0b',
    warningLight: '#fbbf24',
    warningDark: '#d97706',
    onWarning: '#000000',
    
    info: '#3b82f6',
    infoLight: '#60a5fa',
    infoDark: '#2563eb',
    onInfo: '#ffffff',
    
    // Background colors - modern dark with slight blue tint
    background: '#0f0f23',
    surface: '#1a1a2e',
    surfaceVariant: '#16213e',
    surfaceDisabled: '#2a2a3e',
    
    // Text colors - better contrast
    text: '#f1f5f9',
    textSecondary: '#cbd5e1',
    placeholder: '#94a3b8',
    disabled: '#64748b',
    onSurface: '#f1f5f9',
    onSurfaceVariant: '#cbd5e1',
    
    // Card and border - subtle purple/blue tint
    card: '#1e1e3f',
    cardShadow: 'rgba(0,0,0,0.6)',
    border: '#2d3748',
    outline: '#4a5568',
    
    // Status colors - more vibrant
    pending: '#f59e0b',
    assigned: '#3b82f6',
    inProgress: '#8b5cf6',
    completed: '#22c55e',
    rejected: '#ef4444',
    
    // Gradient colors - more attractive combinations
    gradient: {
      primary: ['#4f46e5', '#7c3aed'],
      secondary: ['#6366f1', '#8b5cf6'],
      success: ['#22c55e', '#16a34a'],
      warning: ['#f59e0b', '#d97706'],
      danger: ['#ef4444', '#dc2626'],
    },
    
    // Elevation - layered dark surfaces
    elevation: {
      level0: '#0f0f23',
      level1: '#1a1a2e',
      level2: '#1e1e3f',
      level3: '#252547',
      level4: '#2a2a52',
      level5: '#2f2f5a',
    },
  }
};

// Export default theme (light) for backward compatibility
const theme = lightTheme;
export default theme;
