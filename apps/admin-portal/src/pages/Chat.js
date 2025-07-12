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
  Drawer,
  Fade
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Chat as ChatIcon,
  AutoAwesome as AutoAwesomeIcon,
  Menu as MenuIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      style={{ width: '100%', height: '100%' }}
    >
      <Paper 
        elevation={0} 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center', 
          color: 'text.secondary', 
          p: { xs: 3, md: 5 },
          borderRadius: 3,
          background: theme.palette.mode === 'dark' 
            ? alpha(theme.palette.background.paper, 0.8)
            : 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(10px)',
          border: `1px solid ${theme.palette.mode === 'dark' 
            ? alpha(theme.palette.background.paper, 0.9) 
            : 'rgba(255, 255, 255, 0.8)'}`,
          position: 'relative',
          overflow: 'hidden',
          boxShadow: theme.palette.mode === 'dark'
            ? `0 8px 32px ${alpha(theme.palette.common.black, 0.2)}`
            : '0 8px 32px rgba(0, 0, 0, 0.08)',
          '&::after': {
            content: '""',
            position: 'absolute',
            top: '-50%',
            left: '-50%',
            width: '200%',
            height: '200%',
            background: theme.palette.mode === 'dark'
              ? `radial-gradient(circle, ${alpha(theme.palette.primary.main, 0.05)} 0%, transparent 70%)`
              : 'radial-gradient(circle, rgba(59, 130, 246, 0.03) 0%, transparent 70%)',
            zIndex: 0,
          },
        }}
      >
        <motion.div
          animate={{ 
            rotate: [0, 5, -5, 0],
            scale: [1, 1.05, 1]
          }}
          transition={{ 
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          style={{
            position: 'relative',
            zIndex: 1
          }}
        >
          <Box sx={{
            width: 100,
            height: 100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '24px',
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            boxShadow: '0 15px 30px rgba(139, 92, 246, 0.25)',
            mb: 4,
          }}>
            <ChatIcon sx={{ 
              fontSize: 48, 
              color: 'white',
            }} />
          </Box>
        </motion.div>
        <Typography 
          variant="h4" 
          gutterBottom 
          sx={{ 
            fontWeight: 700, 
            color: 'text.primary',
            textAlign: 'center',
            mb: 2,
            position: 'relative',
            zIndex: 1,
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            letterSpacing: '-0.02em',
          }}
        >
          Welcome to Chat Support
        </Typography>
        <Typography 
          variant="body1" 
          textAlign="center" 
          sx={{ 
            maxWidth: 400, 
            lineHeight: 1.8,
            color: 'text.secondary',
            opacity: 0.9,
            position: 'relative',
            zIndex: 1,
            fontSize: '1.05rem',
            mb: 4,
          }}
        >
          Select a conversation from the list to start messaging. 
          Get instant support and stay connected with your team.
        </Typography>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          style={{ marginTop: '8px', position: 'relative', zIndex: 1 }}
        >
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            py: 1.5,
            px: 3,
            borderRadius: 3,
            background: theme.palette.mode === 'dark'
              ? `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.15)} 0%, ${alpha(theme.palette.secondary.main || theme.palette.primary.light, 0.15)} 100%)`
              : 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)',
            border: theme.palette.mode === 'dark'
              ? `1px solid ${alpha(theme.palette.primary.main, 0.3)}`
              : '1px solid rgba(139, 92, 246, 0.2)',
          }}>
            <AutoAwesomeIcon sx={{ 
              fontSize: 20, 
              color: theme.palette.mode === 'dark' ? theme.palette.primary.light : '#8b5cf6' 
            }} />
            <Typography 
              variant="body2" 
              sx={{ 
                color: theme.palette.mode === 'dark' ? theme.palette.primary.light : '#6d28d9',
                fontWeight: 600
              }}
            >
              AI-powered support available
            </Typography>
          </Box>
        </motion.div>
      </Paper>
    </motion.div>
  ), [theme.palette.mode, theme.palette.background.paper, theme.palette.common.black, theme.palette.primary.main, theme.palette.primary.light, theme.palette.secondary.main]);

  return (
    <Container 
      maxWidth="xl" 
      sx={{ 
        py: { xs: 2, md: 4 }, 
        height: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        background: theme.palette.mode === 'dark' 
          ? `linear-gradient(180deg, ${alpha(theme.palette.background.default, 0.8)} 0%, ${alpha(theme.palette.background.paper, 0.8)} 100%)`
          : 'linear-gradient(180deg, rgba(249, 250, 251, 0.8) 0%, rgba(242, 244, 247, 0.8) 100%)',
      }}
    >
      {/* Header with enhanced styling */}
      <Fade in timeout={800}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 2, 
            mb: { xs: 3, md: 4 },
            position: 'relative',
            backdropFilter: 'blur(8px)',
            borderRadius: 3,
            py: 2,
            px: { xs: 2, md: 3 },
            background: theme.palette.mode === 'dark'
              ? `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.9)} 0%, ${alpha(theme.palette.background.paper, 0.8)} 100%)`
              : 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.8) 100%)',
            boxShadow: theme.palette.mode === 'dark'
              ? `0 4px 20px ${alpha(theme.palette.common.black, 0.2)}`
              : '0 4px 20px rgba(0, 0, 0, 0.08)',
            border: `1px solid ${theme.palette.mode === 'dark'
              ? alpha(theme.palette.background.paper, 0.9)
              : 'rgba(255, 255, 255, 0.8)'}`,
          }}
        >
          {isMobile && (
            <IconButton
              onClick={handleDrawerToggle}
              sx={{ 
                bgcolor: 'background.paper',
                boxShadow: '0 2px 10px rgba(0, 0, 0, 0.08)',
                '&:hover': { 
                  bgcolor: 'grey.100',
                  transform: 'scale(1.05)'
                },
                transition: 'all 0.2s ease-in-out'
              }}
              aria-label="Open chat menu"
            >
              <MenuIcon />
            </IconButton>
          )}
          <Typography 
            variant="h4" 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 2,
              fontWeight: 700,
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              flex: 1,
              letterSpacing: '-0.02em',
            }}
          >
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 50,
              height: 50,
              borderRadius: '14px',
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
            }}>
              <ChatIcon sx={{ color: 'white', fontSize: 28 }} />
            </Box>
            Chat Support
          </Typography>
        </Box>
      </Fade>
      
      {/* Main Content with modern styling */}
      <Grid container spacing={3} sx={{ flex: 1, height: 0 }}>
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
                width: 320,
                boxSizing: 'border-box',
                bgcolor: 'background.default',
                backgroundImage: 'linear-gradient(180deg, rgba(249, 250, 251, 0.95) 0%, rgba(242, 244, 247, 0.95) 100%)',
                backdropFilter: 'blur(10px)',
                borderRight: '1px solid rgba(255, 255, 255, 0.8)',
                boxShadow: '5px 0 30px rgba(0, 0, 0, 0.1)',
              }
            }}
          >
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              p: 2.5,
              borderBottom: '1px solid',
              borderColor: 'divider',
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
            }}>
              <Typography variant="h6" sx={{ 
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: 1.5,
                background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                <ChatIcon sx={{ color: '#6d28d9' }} />
                Conversations
              </Typography>
              <IconButton 
                onClick={handleDrawerToggle}
                size="small"
                sx={{ 
                  bgcolor: 'rgba(255, 255, 255, 0.8)',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
                  '&:hover': { 
                    bgcolor: 'rgba(255, 255, 255, 0.95)',
                    transform: 'scale(1.05)'
                  },
                  transition: 'all 0.2s ease-in-out'
                }}
              >
                <CloseIcon />
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
            <Fade in timeout={600}>
              <div style={{ height: '100%', position: 'relative' }}>
                <Box
                  sx={{
                    position: 'absolute',
                    top: 15,
                    left: 15,
                    right: 15,
                    bottom: 0,
                    borderRadius: 3,
                    background: 'rgba(139, 92, 246, 0.03)',
                    zIndex: 0,
                  }}
                />
                <ChatRoomsList
                  onSelectRoom={handleRoomSelect}
                  selectedRoomId={selectedRoom?.tenantId}
                />
              </div>
            </Fade>
          </Grid>
        )}

        {/* Chat Window */}
        <Grid item xs={12} md={isMobile ? 12 : 8} lg={isMobile ? 12 : 9}>
          <Fade in timeout={800}>
            <div style={{ height: '100%', position: 'relative' }}>
              <Box
                sx={{
                  position: 'absolute',
                  top: 15,
                  left: 15,
                  right: 15,
                  bottom: 0,
                  borderRadius: 3,
                  background: 'rgba(59, 130, 246, 0.03)',
                  zIndex: 0,
                  display: selectedRoom ? 'block' : 'none',
                }}
              />
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
            </div>
          </Fade>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Chat;
