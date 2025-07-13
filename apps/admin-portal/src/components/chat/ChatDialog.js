import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Grid,
  IconButton,
  Box,
  Typography,
  Slide,
  useTheme,
  Fade,
  useMediaQuery,
  alpha
} from '@mui/material';
import {
  Close as CloseIcon,
  Chat as ChatIcon
} from '@mui/icons-material';
import ChatRoomsList from './ChatRoomsList';
import ChatWindow from './ChatWindow';
import { useSocket } from '../../context/SocketContext';

// Slide transition for the dialog
const Transition = React.forwardRef(function Transition(props, ref) {
  return <Slide direction="up" ref={ref} {...props} />;
});

const ChatDialog = ({ open, onClose }) => {
  const [selectedRoom, setSelectedRoom] = useState(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { markAsRead } = useSocket();

  // Clear the room's unread count when selecting it
  const handleRoomSelect = (room) => {
    if (room?.tenantId) {
      markAsRead(room.tenantId);
    }
    setSelectedRoom(room);
  };

  const handleCloseChat = () => {
    setSelectedRoom(null);
  };

  const handleDialogClose = () => {
    setSelectedRoom(null);
    onClose();
  };

  useEffect(() => {
    if (!open) {
      // Reset selected room when dialog closes
      setSelectedRoom(null);
    }
  }, [open]);

  return (
    <Dialog
      open={open}
      onClose={handleDialogClose}
      maxWidth="lg"
      fullWidth
      TransitionComponent={Transition}
      PaperProps={{
        elevation: 8,
        sx: { 
          height: isMobile ? '100vh' : '85vh', 
          maxHeight: isMobile ? '100vh' : 850,
          borderRadius: isMobile ? 0 : 1,
          overflow: 'hidden',
          background: theme.palette.background.paper,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid',
          borderColor: theme.palette.divider,
          transform: open ? 'scale(1)' : 'scale(0.98)',
        }
      }}
      sx={{
        '& .MuiBackdrop-root': {
          backdropFilter: 'blur(4px)',
          backgroundColor: 'rgba(0, 0, 0, 0.4)',
        }
      }}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        py: 1.5,
        px: 2,
        borderBottom: `1px solid ${theme.palette.divider}`,
        bgcolor: theme.palette.background.paper,
        position: 'relative',
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ChatIcon fontSize="small" sx={{ color: theme.palette.text.secondary }} />
          <Box>
            <Typography 
              variant="subtitle2"
              sx={{
                fontWeight: 500,
                color: theme.palette.text.primary,
              }}
            >
              Chat Support
            </Typography>
            <Typography 
              variant="caption"
              sx={{
                fontWeight: 500,
                color: alpha(theme.palette.text.primary, 0.6),
                letterSpacing: '0.01em',
              }}
            >
              Send and receive messages in real-time
            </Typography>
          </Box>
        </Box>
        <IconButton 
          onClick={handleDialogClose}
          sx={{ 
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            boxShadow: '0 3px 12px rgba(0, 0, 0, 0.08)',
            '&:hover': { 
              bgcolor: 'rgba(255, 255, 255, 1)',
              transform: 'scale(1.08)',
              boxShadow: '0 6px 16px rgba(0, 0, 0, 0.12)',
            },
            transition: 'all 0.25s ease-in-out'
          }}
        >
          <CloseIcon sx={{ color: theme.palette.text.secondary }} />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 0, height: 'calc(100% - 82px)' }}>
        <Grid container sx={{ height: '100%' }}>
          {/* Chat Rooms List */}
          <Grid item xs={12} md={4} lg={3} sx={{ 
            borderRight: { md: `1px solid ${theme.palette.divider}` },
            height: '100%',
            position: 'relative',
          }}>
            <ChatRoomsList
              onSelectRoom={handleRoomSelect}
              selectedRoomId={selectedRoom?.tenantId}
            />
          </Grid>

          {/* Chat Window */}
          <Grid item xs={12} md={8} lg={9} sx={{ 
            height: '100%',
            position: 'relative',
            bgcolor: theme.palette.background.default,
          }}>
            {selectedRoom ? (
              <ChatWindow
                tenantId={selectedRoom.tenantId}
                tenantName={selectedRoom.tenantName}
                contactName={selectedRoom.contactName}
                onClose={handleCloseChat}
              />
            ) : (
              <Fade in={!selectedRoom} timeout={300}>
                <Box
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'text.secondary',
                    p: 5,
                    position: 'relative',
                    zIndex: 1,
                  }}
                >
                  <Box sx={{
                    width: 120,
                    height: 120,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: '28px',
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%)',
                    boxShadow: '0 8px 20px rgba(139, 92, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.6)',
                    mb: 4,
                    border: '1px solid rgba(139, 92, 246, 0.25)',
                    position: 'relative',
                    overflow: 'hidden',
                    '&:before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      height: '50%',
                      background: 'rgba(255, 255, 255, 0.3)',
                    }
                  }}>
                    <ChatIcon sx={{ fontSize: 50, color: theme.palette.primary.main, opacity: 0.9 }} />
                  </Box>
                  <Typography 
                    variant="h5" 
                    gutterBottom
                    sx={{
                      fontWeight: 700,
                      background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      mb: 2,
                      letterSpacing: '-0.01em',
                    }}
                  >
                    Select a conversation
                  </Typography>
                  <Typography variant="body1" textAlign="center" sx={{ 
                    maxWidth: 420,
                    lineHeight: 1.8,
                    color: alpha(theme.palette.text.secondary, 0.85),
                    mb: 4,
                    fontWeight: 400,
                  }}>
                    Choose a tenant from the list to view messages and provide support. 
                    All conversations are secured and encrypted end-to-end.
                  </Typography>
                </Box>
              </Fade>
            )}
          </Grid>
        </Grid>
      </DialogContent>
    </Dialog>
  );
};

export default ChatDialog;
