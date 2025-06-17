import React, { createContext, useContext, useState, useEffect } from 'react';
import { getAuthToken, getFieldWorkerData, logout } from '../utils/auth';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [fieldWorker, setFieldWorker] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = await getAuthToken();
      const fieldWorkerData = await getFieldWorkerData();
      
      if (token && fieldWorkerData) {
        setIsAuthenticated(true);
        setFieldWorker(fieldWorkerData);
      } else {
        setIsAuthenticated(false);
        setFieldWorker(null);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
      setIsAuthenticated(false);
      setFieldWorker(null);
    } finally {
      setIsLoading(false);
    }
  };

  const login = (fieldWorkerData) => {
    setIsAuthenticated(true);
    setFieldWorker(fieldWorkerData);
  };

  const logoutUser = async () => {
    try {
      await logout();
      setIsAuthenticated(false);
      setFieldWorker(null);
    } catch (error) {
      console.error('Error during logout:', error);
    }
  };

  const updateFieldWorker = (updatedData) => {
    setFieldWorker(updatedData);
  };

  const value = {
    isAuthenticated,
    fieldWorker,
    isLoading,
    login,
    logout: logoutUser,
    updateFieldWorker,
    checkAuthStatus,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
