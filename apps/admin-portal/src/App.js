import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider as MuiThemeProvider, CssBaseline } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { AuthProvider } from './hooks/useAuth';
import { TenantProvider } from './context/TenantContext';
import { SearchProvider } from './context/SearchContext';
import { SocketProvider } from './context/SocketContext';
import { ThemeProvider, useThemeContext } from './context/ThemeContext';

// Theme
import { createAppTheme } from './theme';

// Layout
import MainLayout from './components/layout/MainLayout';
import ProtectedRoute from './components/auth/ProtectedRoute';

// Pages
import Dashboard from './pages/Dashboard';
import Reports from './pages/Reports';
import MapView from './pages/MapView';
import Analytics from './pages/Analytics';
import Repair from './pages/Repair';
import Historical from './pages/Historical';
import Login from './pages/Login';
import Profile from './pages/Profile';
import AiAnalysis from './pages/AiAnalysis';
import ManageTenants from './pages/ManageTenants';
import TenantDetails from './pages/TenantDetails';
import SearchResults from './pages/SearchResults';
import Chat from './pages/Chat';
import Landing from './pages/Landing';
import AiChatPage from './pages/AiChatPage';
import RequestAccess from './pages/RequestAccess';
import ManageAccessRequests from './pages/ManageAccessRequests';

// ThemeApp component handles the MUI theme based on theme context
const ThemeApp = ({ children }) => {
  const { darkMode } = useThemeContext();
  
  // Create theme based on current mode
  const currentTheme = React.useMemo(() => {
    return createAppTheme(darkMode ? 'dark' : 'light');
  }, [darkMode]);

  return (
    <MuiThemeProvider theme={currentTheme}>
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
};

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <ThemeApp>
          <LocalizationProvider dateAdapter={AdapterDateFns}>
            <AuthProvider>
              <TenantProvider>
                <SearchProvider>
                  <SocketProvider>
                    <Routes>
                      <Route path="/" element={<Landing />} />
                      <Route path="/landing" element={<Landing />} />
                      <Route path="/login" element={<Login />} />
                      <Route path="/request-access" element={<RequestAccess />} />
                    
                      <Route element={<ProtectedRoute><MainLayout /></ProtectedRoute>}>
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/reports" element={<Reports />} />
                        <Route path="/map" element={<MapView />} />
                        <Route path="/analytics" element={<Analytics />} />
                        <Route path="/repairs" element={<Repair />} />
                        <Route path="/historical" element={<Historical />} />
                        <Route path="/profile" element={<Profile />} />
                        <Route path="/ai-analysis" element={<AiAnalysis />} />
                        <Route path="/ai-chat" element={<AiChatPage />} />
                        <Route path="/search-results" element={<SearchResults />} />
                        <Route path="/chat" element={<Chat />} />
                        {/* Tenant Management */}
                        <Route path="/tenants" element={<ManageTenants />} />
                        <Route path="/tenants/:tenantId" element={<TenantDetails />} />
                        <Route path="/access-requests" element={<ManageAccessRequests />} />
                      </Route>
        
                      {/* Catch all route */}
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </SocketProvider>
                </SearchProvider>
              </TenantProvider>
            </AuthProvider>
          </LocalizationProvider>
        </ThemeApp>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;