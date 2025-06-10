import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  Box,
  useTheme,
  useMediaQuery,
  Stack,
  IconButton,
  Alert,
  Snackbar
} from '@mui/material';
import InventoryIcon from '@mui/icons-material/Inventory';
import HomeIcon from '@mui/icons-material/Home';
import { api } from '../utils/api';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showError, setShowError] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email || !password) {
      setError('Please fill in all fields');
      setShowError(true);
      return;
    }

    try {
      setLoading(true);
      const response = await api.post('/auth/login', { email, password });
      localStorage.setItem('auth_token', response.token);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Login failed. Please check your credentials.');
      setShowError(true);
    } finally {
      setLoading(false);
    }
  };

  const handleCloseError = () => {
    setShowError(false);
  };

  return (
    <Container maxWidth="lg" sx={{ 
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      py: 4
    }}>
      <IconButton 
        onClick={() => navigate('/')}
        sx={{ 
          position: 'absolute',
          top: 20,
          left: 20,
          color: theme.palette.primary.main
        }}
      >
        <HomeIcon sx={{ fontSize: 32 }} />
      </IconButton>
      <Paper
        elevation={3}
        sx={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          borderRadius: '24px',
          overflow: 'hidden',
          minHeight: '600px',
          width: '100%',
          background: theme.palette.background.paper,
          boxShadow: '0 8px 40px rgba(0, 0, 0, 0.12)'
        }}
      >
        {/* Left Side - Product Info */}
        <Box
          sx={{
            flex: '1 1 50%',
            p: 6,
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
            color: 'white',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
            <InventoryIcon sx={{ fontSize: 48, mr: 2 }} />
            <Typography variant="h3" sx={{ fontWeight: 700 }}>
              DCS
            </Typography>
          </Box>
          
          <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
            Welcome to the Damage Control System
          </Typography>
          
          <Typography variant="h6" sx={{ mb: 4, opacity: 0.9 }}>
            Monitor and manage damage control operations in real-time
          </Typography>

          <Box sx={{ mb: 4 }}>
            <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
              ✓ Real-time damage assessment tracking
            </Typography>
            <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
              ✓ Comprehensive repair status monitoring
            </Typography>
          </Box>
        </Box>

        <Box
          sx={{
            flex: '1 1 50%',
            p: 6,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
        >
          <Typography variant="h4" sx={{ mb: 4, fontWeight: 600 }}>
            Sign In
          </Typography>

          <Box component="form" onSubmit={handleSubmit}>

            <TextField
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              variant="outlined"
              sx={{ mb: 3 }}
            />

            <TextField
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              variant="outlined"
              sx={{ mb: 4 }}
            />

            <Stack spacing={2}>
              <Button
                type="submit"
                fullWidth
                variant="contained"
                disabled={!email || !password || loading}
                sx={{
                  py: 1.8,
                  borderRadius: '12px',
                  fontSize: '1.1rem',
                  textTransform: 'none',
                  fontWeight: 600,
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                  '&:hover': {
                    boxShadow: '0 6px 16px rgba(0, 0, 0, 0.25)',
                  }
                }}
              >
                {loading ? 'Signing in...' : 'Sign In to Dashboard'}
              </Button>
            </Stack>

          </Box>
        </Box>
      </Paper>

      <Snackbar 
        open={showError} 
        autoHideDuration={6000} 
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

    </Container>
  );
};

export default Login;
