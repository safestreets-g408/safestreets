import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1a73e8',
      light: '#4791db',
      dark: '#115293',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#41b883',
      light: '#69c49a',
      dark: '#2d805c',
      contrastText: '#ffffff',
    },
    error: {
      main: '#ef476f',
      light: '#f27491',
      dark: '#a73250',
    },
    warning: {
      main: '#ffd166',
      light: '#ffdc85',
      dark: '#b29247',
    },
    info: {
      main: '#118ab2',
      light: '#41a1c2',
      dark: '#0b617d',
    },
    success: {
      main: '#06d6a0',
      light: '#39dfb8',
      dark: '#049570',
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
    text: {
      primary: '#2c3e50',
      secondary: '#546e7a',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 600,
      fontSize: '2.5rem',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.5,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 16px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
          },
        },
        contained: {
          '&:hover': {
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
          '&:hover': {
            boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
  shape: {
    borderRadius: 8,
  },
  shadows: [
    'none',
    '0 2px 4px rgba(0,0,0,0.05)',
    '0 4px 8px rgba(0,0,0,0.08)',
    '0 8px 16px rgba(0,0,0,0.1)',
    '0 12px 20px rgba(0,0,0,0.12)',
    '0 16px 24px rgba(0,0,0,0.14)',
    '0 20px 28px rgba(0,0,0,0.16)',
    '0 24px 32px rgba(0,0,0,0.18)',
    '0 28px 36px rgba(0,0,0,0.2)',
    '0 32px 40px rgba(0,0,0,0.22)',
    '0 36px 44px rgba(0,0,0,0.24)',
    '0 40px 48px rgba(0,0,0,0.26)',
    '0 44px 52px rgba(0,0,0,0.28)',
    '0 48px 56px rgba(0,0,0,0.3)',
    '0 52px 60px rgba(0,0,0,0.32)',
    '0 56px 64px rgba(0,0,0,0.34)',
    '0 60px 68px rgba(0,0,0,0.36)',
    '0 64px 72px rgba(0,0,0,0.38)',
    '0 68px 76px rgba(0,0,0,0.4)',
    '0 72px 80px rgba(0,0,0,0.42)',
    '0 76px 84px rgba(0,0,0,0.44)',
    '0 80px 88px rgba(0,0,0,0.46)',
    '0 84px 92px rgba(0,0,0,0.48)',
    '0 88px 96px rgba(0,0,0,0.5)',
    '0 92px 100px rgba(0,0,0,0.52)'
  ],
});

export default theme;