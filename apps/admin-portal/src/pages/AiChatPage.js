import React from 'react';
import { Box, Typography, Container } from '@mui/material';
import AiChatInterface from '../components/aiChat/AiChatInterface';
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined';

const AiChatPage = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 1.5 }}>
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
        <SmartToyOutlinedIcon 
          sx={{ 
            mr: 1, 
            fontSize: '1rem',
            color: 'text.secondary'
          }} 
        />
        <Typography 
          variant="subtitle1" 
          sx={{ 
            fontWeight: 500,
            color: 'text.primary'
          }}
        >
          AI Assistant
        </Typography>
      </Box>
      
      <AiChatInterface />
    </Container>
  );
};

export default AiChatPage;
