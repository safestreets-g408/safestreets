import { theme } from '../theme';

export function getThemeValue(path, fallback) {
  try {
    if (!path) return fallback;
    
    const keys = path.split('.');
    let current = theme;
    
    for (const key of keys) {
      if (current === undefined || current === null) return fallback;
      current = current[key];
    }
    
    return current !== undefined && current !== null ? current : fallback;
  } catch (error) {
    console.warn(`Error accessing theme property '${path}':`, error);
    return fallback;
  }
}


export const safeTheme = {
  // Colors
  colors: {
    primary: getThemeValue('colors.primary', '#1e40af'),
    secondary: getThemeValue('colors.secondary', '#4b5563'),
    background: getThemeValue('colors.background', '#f8fafc'),
    surface: getThemeValue('colors.surface', '#ffffff'),
    error: getThemeValue('colors.error', '#dc2626'),
    text: getThemeValue('colors.text', '#0f172a'),
  },
  
  // Typography
  typography: {
    titleLarge: getThemeValue('typography.titleLarge', { fontSize: 18, fontWeight: '500' }),
    titleMedium: getThemeValue('typography.titleMedium', { fontSize: 16, fontWeight: '500' }),
    titleSmall: getThemeValue('typography.titleSmall', { fontSize: 14, fontWeight: '500' }),
    bodyLarge: getThemeValue('typography.bodyLarge', { fontSize: 16, fontWeight: '400' }),
    bodyMedium: getThemeValue('typography.bodyMedium', { fontSize: 14, fontWeight: '400' }),
    bodySmall: getThemeValue('typography.bodySmall', { fontSize: 12, fontWeight: '400' }),
  },
  
  // Spacing
  spacing: {
    xs: getThemeValue('spacing.xs', 4),
    sm: getThemeValue('spacing.sm', 8),
    md: getThemeValue('spacing.md', 16),
    lg: getThemeValue('spacing.lg', 24),
    xl: getThemeValue('spacing.xl', 32),
  },
  
  // Shadows
  shadows: {
    small: getThemeValue('shadows.small', {
      shadowColor: 'rgba(0,0,0,0.1)',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 1,
      shadowRadius: 2,
      elevation: 2,
    }),
    medium: getThemeValue('shadows.medium', {
      shadowColor: 'rgba(0,0,0,0.1)',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 1,
      shadowRadius: 4,
      elevation: 4,
    }),
    large: getThemeValue('shadows.large', {
      shadowColor: 'rgba(0,0,0,0.1)',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 1,
      shadowRadius: 8,
      elevation: 8,
    }),
  },
  
  // Border Radius
  borderRadius: {
    small: getThemeValue('borderRadius.small', 8),
    medium: getThemeValue('borderRadius.medium', 12),
    large: getThemeValue('borderRadius.large', 16),
    full: getThemeValue('borderRadius.full', 999),
  },
};

export default safeTheme;
