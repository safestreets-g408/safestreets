import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import AiChatInterface from '../components/aiChat/AiChatInterface';

const AiChatPage = () => {
  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          AI Assistant
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Ask our Gemini-powered AI assistant questions about road maintenance, damage assessment, 
          or get help with administrative tasks.
        </Typography>
      </Paper>
      
      <AiChatInterface />
    </Box>
  );
};

export default AiChatPage;
