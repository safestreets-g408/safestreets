import { createTheme } from '@mui/material/styles';

// Create theme based on mode (light/dark)
const createAppTheme = (mode) => createTheme({
  palette: {
    mode,
    primary: {
      main: '#2563eb',
      light: '#60a5fa',
      dark: '#1d4ed8',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#6b7280',
      light: '#9ca3af',
      dark: '#374151',
      contrastText: '#ffffff',
    },
    error: {
      main: '#dc2626',
      light: '#ef4444',
      dark: '#b91c1c',
    },
    warning: {
      main: '#f59e0b',
      light: '#fbbf24',
      dark: '#d97706',
    },
    info: {
      main: '#3b82f6',
      light: '#60a5fa',
      dark: '#2563eb',
    },
    success: {
      main: '#10b981',
      light: '#34d399',
      dark: '#059669',
    },
    background: {
      default: mode === 'dark' ? '#121212' : '#f9fafb',
      paper: mode === 'dark' ? '#1e1e1e' : '#ffffff',
      appBar: mode === 'dark' ? '#1a1a1a' : '#ffffff',
      card: mode === 'dark' ? '#1e1e1e' : '#ffffff',
      dialog: mode === 'dark' ? '#1e1e1e' : '#ffffff',
      menu: mode === 'dark' ? '#1e1e1e' : '#ffffff',
    },
    text: {
      primary: mode === 'dark' ? '#f3f4f6' : '#111827',
      secondary: mode === 'dark' ? '#d1d5db' : '#6b7280',
      disabled: mode === 'dark' ? '#6b7280' : '#9ca3af',
    },
    action: {
      active: mode === 'dark' ? '#f3f4f6' : '#111827',
      hover: mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
      selected: mode === 'dark' ? 'rgba(255, 255, 255, 0.16)' : 'rgba(0, 0, 0, 0.08)',
      disabled: mode === 'dark' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.26)',
      disabledBackground: mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
    },
    divider: mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
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
  spacing: 8,
  shape: {
    borderRadius: 4,
  },
  typography: {
    fontFamily: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
    h1: {
      fontWeight: 600,
      fontSize: '2.25rem',
      lineHeight: 1.2,
      letterSpacing: '-0.025em',
    },
    h2: {
      fontWeight: 600,
      fontSize: '1.875rem',
      lineHeight: 1.25,
      letterSpacing: '-0.025em',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.33,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.125rem',
      lineHeight: 1.4,
    },
    h6: {
      fontWeight: 600,
      fontSize: '1rem',
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.5,
      // Use theme-aware color that adjusts for dark mode
      color: mode === 'dark' ? '#f3f4f6' : '#1e293b',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      // Use theme-aware color that adjusts for dark mode
      color: mode === 'dark' ? '#d1d5db' : '#475569',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      fontSize: '0.875rem',
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
      color: mode === 'dark' ? '#a1a1aa' : '#64748b',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: (theme) => ({
        body: {
          backgroundColor: theme.palette.background.default,
          color: theme.palette.text.primary,
          transition: 'background-color 0.3s ease, color 0.3s ease',
        }
      }),
    },
    MuiAppBar: {
      styleOverrides: {
        root: ({ theme }) => ({
          backgroundColor: theme.palette.mode === 'dark' 
            ? theme.palette.background.appBar 
            : '#ffffff',
          color: theme.palette.text.primary,
          borderBottom: theme.palette.mode === 'dark'
            ? '1px solid rgba(255, 255, 255, 0.1)'
            : '1px solid rgba(0, 0, 0, 0.1)',
        }),
      },
    },

    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '10px 20px',
          boxShadow: 'none',
          fontWeight: 600,
          fontSize: '0.875rem',
          textTransform: 'none',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            boxShadow: 'none',
            transform: 'translateY(-1px)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
            boxShadow: '0 8px 25px rgba(102, 126, 234, 0.4)',
          },
        },
        outlined: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
            backgroundColor: 'rgba(102, 126, 234, 0.04)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: 16,
          boxShadow: theme.palette.mode === 'dark' 
            ? '0 4px 20px rgba(0, 0, 0, 0.2)' 
            : '0 4px 20px rgba(0, 0, 0, 0.08)',
          border: theme.palette.mode === 'dark'
            ? '1px solid rgba(255, 255, 255, 0.1)'
            : '1px solid rgba(0, 0, 0, 0.06)',
          background: theme.palette.background.card,
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            boxShadow: theme.palette.mode === 'dark'
              ? '0 12px 40px rgba(0, 0, 0, 0.4)'
              : '0 12px 40px rgba(0, 0, 0, 0.12)',
            transform: 'translateY(-4px)',
          },
        }),
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: ({ theme }) => ({
          backgroundColor: theme.palette.mode === 'dark' 
            ? theme.palette.background.paper 
            : '#ffffff',
          color: theme.palette.text.primary,
          transition: 'background-color 0.3s ease',
        }),
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: ({ theme }) => ({
          backgroundColor: theme.palette.mode === 'dark'
            ? theme.palette.background.menu
            : '#ffffff',
          color: theme.palette.text.primary,
        }),
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: ({ theme }) => ({
          backgroundColor: theme.palette.mode === 'dark'
            ? theme.palette.background.dialog
            : '#ffffff',
          color: theme.palette.text.primary,
        }),
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: ({ theme }) => ({
          backgroundColor: theme.palette.mode === 'dark' 
            ? theme.palette.background.paper
            : '#ffffff',
          transition: 'background-color 0.3s ease',
        }),
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: ({ theme }) => ({
          color: theme.palette.mode === 'dark' 
            ? theme.palette.text.primary
            : theme.palette.text.secondary,
          '&:hover': {
            backgroundColor: theme.palette.mode === 'dark'
              ? 'rgba(255, 255, 255, 0.1)'
              : 'rgba(0, 0, 0, 0.04)',
          },
        }),
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderColor: theme.palette.mode === 'dark'
            ? 'rgba(255, 255, 255, 0.12)'
            : 'rgba(0, 0, 0, 0.12)',
        }),
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          margin: '4px 0',
          '&.Mui-selected': {
            background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)',
            },
          },
          '&:hover': {
            backgroundColor: 'rgba(102, 126, 234, 0.06)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          fontSize: '0.75rem',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
          },
        },
      },
    },
  },
  shadows: [
    'none',
    '0 2px 8px rgba(0, 0, 0, 0.08)',
    '0 4px 12px rgba(0, 0, 0, 0.1)',
    '0 8px 24px rgba(0, 0, 0, 0.12)',
    '0 12px 32px rgba(0, 0, 0, 0.14)',
    '0 16px 40px rgba(0, 0, 0, 0.16)',
    '0 20px 48px rgba(0, 0, 0, 0.18)',
    '0 24px 56px rgba(0, 0, 0, 0.2)',
    '0 28px 64px rgba(0, 0, 0, 0.22)',
    '0 32px 72px rgba(0, 0, 0, 0.24)',
    '0 36px 80px rgba(0, 0, 0, 0.26)',
    '0 40px 88px rgba(0, 0, 0, 0.28)',
    '0 44px 96px rgba(0, 0, 0, 0.3)',
    '0 48px 104px rgba(0, 0, 0, 0.32)',
    '0 52px 112px rgba(0, 0, 0, 0.34)',
    '0 56px 120px rgba(0, 0, 0, 0.36)',
    '0 60px 128px rgba(0, 0, 0, 0.38)',
    '0 64px 136px rgba(0, 0, 0, 0.4)',
    '0 68px 144px rgba(0, 0, 0, 0.42)',
    '0 72px 152px rgba(0, 0, 0, 0.44)',
    '0 76px 160px rgba(0, 0, 0, 0.46)',
    '0 80px 168px rgba(0, 0, 0, 0.48)',
    '0 84px 176px rgba(0, 0, 0, 0.5)',
    '0 88px 184px rgba(0, 0, 0, 0.52)',
    '0 92px 192px rgba(0, 0, 0, 0.54)'
  ],
});

// Export the theme creator function
export { createAppTheme };

// Default theme for backward compatibility
const theme = createAppTheme('light');
export default theme;