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
  useMediaQuery
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
        elevation: 2,
        sx: { 
          height: isMobile ? '100vh' : '85vh', 
          maxHeight: isMobile ? '100vh' : 850,
          borderRadius: isMobile ? 0 : 1,
          overflow: 'hidden',
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
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ChatIcon fontSize="small" color="primary" />
          <Typography variant="subtitle1" fontWeight={500}>
            Chat Support
          </Typography>
        </Box>
        <IconButton onClick={handleDialogClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 0, height: 'calc(100% - 64px)' }}>
        <Grid container sx={{ height: '100%' }}>
          {/* Chat Rooms List */}
          <Grid item xs={12} md={4} lg={3} sx={{ 
            borderRight: { md: `1px solid ${theme.palette.divider}` },
            height: '100%',
          }}>
            <ChatRoomsList
              onSelectRoom={handleRoomSelect}
              selectedRoomId={selectedRoom?.tenantId}
            />
          </Grid>

          {/* Chat Window */}
          <Grid item xs={12} md={8} lg={9} sx={{ 
            height: '100%',
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
                    p: 3,
                  }}
                >
                  <Box sx={{
                    width: 80,
                    height: 80,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: '50%',
                    bgcolor: theme.palette.action.hover,
                    mb: 3,
                  }}>
                    <ChatIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />
                  </Box>
                  <Typography 
                    variant="h6" 
                    gutterBottom
                    sx={{
                      fontWeight: 600,
                      mb: 1,
                      color: theme.palette.text.primary,
                    }}
                  >
                    Select a conversation
                  </Typography>
                  <Typography 
                    variant="body2" 
                    textAlign="center" 
                    color="text.secondary"
                    sx={{ 
                      maxWidth: 400,
                      mb: 2,
                    }}
                  >
                    Choose a tenant from the list to view messages and provide support.
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
