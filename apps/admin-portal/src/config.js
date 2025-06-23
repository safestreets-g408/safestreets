
const config = {
  // Base URLs for various services
  backend: {
    baseURL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:5030',
  },
  aiServer: {
    baseURL: process.env.REACT_APP_AI_SERVER_URL || 'http://localhost:5000',
  },
  // Add other configuration settings here as needed
};

export default config;
