import React, { useState, useCallback, useMemo } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  useTheme,
  useMediaQuery,
  Box,
  IconButton,
  Drawer
} from '@mui/material';

import {
  Chat as ChatIcon,
  AutoAwesome as AutoAwesomeIcon,
  Menu as MenuIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import ChatRoomsList from '../components/chat/ChatRoomsList';
import ChatWindow from '../components/chat/ChatWindow';

const Chat = () => {
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleRoomSelect = useCallback((room) => {
    setSelectedRoom(room);
    if (isMobile) {
      setMobileDrawerOpen(false);
    }
  }, [isMobile]);

  const handleCloseChat = useCallback(() => {
    setSelectedRoom(null);
  }, []);

  const handleDrawerToggle = useCallback(() => {
    setMobileDrawerOpen(!mobileDrawerOpen);
  }, [mobileDrawerOpen]);

  const emptyStateContent = useMemo(() => (
    <Box 
      sx={{ 
        width: '100%', 
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Paper 
        elevation={0} 
        sx={{ 
          maxWidth: 400,
          p: 3,
          textAlign: 'center',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
        }}
      >
        <Box sx={{
          width: 60,
          height: 60,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 1,
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          mx: 'auto',
          mb: 2,
        }}>
          <ChatIcon sx={{ fontSize: 28 }} />
        </Box>
        
        <Typography 
          variant="h6" 
          gutterBottom 
          sx={{ 
            fontWeight: 500,
            mb: 1,
          }}
        >
          Select a conversation
        </Typography>
        
        <Typography 
          variant="body2" 
          sx={{ 
            color: 'text.secondary',
            mb: 3,
          }}
        >
          Choose a conversation from the list to start messaging or get support
        </Typography>
        
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
          py: 1,
          px: 2,
          borderRadius: 1,
          bgcolor: 'action.hover',
          border: '1px solid',
          borderColor: 'divider',
          mx: 'auto',
          maxWidth: 'fit-content',
        }}>
          <AutoAwesomeIcon sx={{ 
            fontSize: '0.875rem', 
            color: 'primary.main',
          }} />
          <Typography 
            variant="caption" 
            sx={{ 
              color: 'text.primary',
              fontWeight: 500,
            }}
          >
            AI-powered support available
          </Typography>
        </Box>
      </Paper>
    </Box>
  ), []);

  return (
    <Container 
      maxWidth="xl" 
      sx={{ 
        py: 2, 
        height: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        bgcolor: 'background.default',
      }}
    >
      {/* Header with minimal styling */}
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1.5, 
          mb: 2,
          py: 1.5,
          px: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        {isMobile && (
          <IconButton
            onClick={handleDrawerToggle}
            size="small"
            sx={{ 
              color: 'text.secondary',
              p: 0.75,
            }}
            aria-label="Open chat menu"
          >
            <MenuIcon fontSize="small" />
          </IconButton>
        )}
        <Typography 
          variant="subtitle1" 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1,
            fontWeight: 500,
            flex: 1,
          }}
        >
          <ChatIcon sx={{ fontSize: '1.125rem', color: 'primary.main' }} />
          Chat Support
        </Typography>
      </Box>
      
      {/* Main Content with minimal styling */}
      <Grid container spacing={2} sx={{ flex: 1, height: 0 }}>
        {/* Mobile Drawer for Chat Rooms */}
        {isMobile && (
          <Drawer
            variant="temporary"
            open={mobileDrawerOpen}
            onClose={handleDrawerToggle}
            ModalProps={{ 
              keepMounted: true,
              disableScrollLock: true
            }}
            sx={{
              '& .MuiDrawer-paper': {
                width: 280,
                boxSizing: 'border-box',
                bgcolor: 'background.paper',
                borderRight: '1px solid',
                borderColor: 'divider',
              }
            }}
          >
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              py: 1.5,
              px: 2,
              borderBottom: '1px solid',
              borderColor: 'divider',
              height: '60px',
            }}>
              <Typography variant="subtitle2" sx={{ 
                fontWeight: 500,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}>
                <ChatIcon sx={{ fontSize: '1.125rem', color: 'primary.main' }} />
                Conversations
              </Typography>
              <IconButton 
                onClick={handleDrawerToggle}
                size="small"
                sx={{ color: 'text.secondary', p: 0.5 }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
            <ChatRoomsList
              onSelectRoom={handleRoomSelect}
              selectedRoomId={selectedRoom?.tenantId}
            />
          </Drawer>
        )}

        {/* Desktop Chat Rooms List */}
        {!isMobile && (
          <Grid item xs={12} md={4} lg={3}>
            <Paper 
              elevation={0}
              sx={{ 
                height: '100%', 
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
                overflow: 'hidden',
              }}
            >
              <ChatRoomsList
                onSelectRoom={handleRoomSelect}
                selectedRoomId={selectedRoom?.tenantId}
              />
            </Paper>
          </Grid>
        )}

        {/* Chat Window */}
        <Grid item xs={12} md={isMobile ? 12 : 8} lg={isMobile ? 12 : 9}>
          <Paper
            elevation={0}
            sx={{ 
              height: '100%',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
              overflow: 'hidden',
            }}
          >
            {selectedRoom ? (
              <ChatWindow
                tenantId={selectedRoom.tenantId}
                tenantName={selectedRoom.tenantName}
                contactName={selectedRoom.contactName}
                roomType={selectedRoom.roomType}
                roomId={selectedRoom.roomId}
                onClose={handleCloseChat}
              />
            ) : (
              emptyStateContent
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Chat;
