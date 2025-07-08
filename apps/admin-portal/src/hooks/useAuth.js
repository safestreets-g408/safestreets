import { useState, useEffect, createContext, useContext } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { TOKEN_KEY, USER_KEY, API_ENDPOINTS } from '../config/constants';
import api, { setAuthRedirect, validateToken } from '../services/apiService';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  // Set up redirect function for session expiration
  useEffect(() => {
    setAuthRedirect(() => {
      // Navigate to login with return URL
      navigate('/login', { 
        replace: true,
        state: { from: location.pathname }
      });
    });
  }, [navigate, location]);

  // Check authentication status on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem(TOKEN_KEY);
      const userData = localStorage.getItem(USER_KEY);
      
      if (token && userData) {
        try {
          // Validate token with backend
          const isValid = await validateToken();
          
          if (isValid) {
            setUser(JSON.parse(userData));
          } else {
            // Token is invalid, clear storage and don't set user
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_KEY);
          }
        } catch (error) {
          console.error('Auth validation error:', error);
          // Clear auth data on error
          localStorage.removeItem(TOKEN_KEY);
          localStorage.removeItem(USER_KEY);
        }
      }
      
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  const login = async (email, password) => {
    try {
      const response = await api.post(`${API_ENDPOINTS.AUTH}/login`, {
        email, 
        password
      });

      const data = response.data;
      localStorage.setItem(TOKEN_KEY, data.token);
      localStorage.setItem(USER_KEY, JSON.stringify(data.admin));
      setUser(data.admin);
      
      // If we have a stored redirect location, use that, otherwise go to dashboard
      const redirectTo = location.state?.from || '/dashboard';
      navigate(redirectTo);
      
      return true;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const logout = () => {
    // Clear all auth-related data
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    
    // Reset user state
    setUser(null);
    
    // Navigate to landing page with replace to prevent back navigation to dashboard
    navigate('/', { replace: true });
  };

  const updateUser = (userData) => {
    const updatedUser = { ...user, ...userData };
    localStorage.setItem(USER_KEY, JSON.stringify(updatedUser));
    setUser(updatedUser);
  };

  const value = {
    user,
    loading,
    login,
    logout,
    updateUser,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 