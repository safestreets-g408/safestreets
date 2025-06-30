import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Grid,
  IconButton,
  Box,
  Typography
} from '@mui/material';
import {
  Close as CloseIcon,
  Chat as ChatIcon
} from '@mui/icons-material';
import ChatRoomsList from './ChatRoomsList';
import ChatWindow from './ChatWindow';

const ChatDialog = ({ open, onClose }) => {
  const [selectedRoom, setSelectedRoom] = useState(null);

  const handleRoomSelect = (room) => {
    setSelectedRoom(room);
  };

  const handleCloseChat = () => {
    setSelectedRoom(null);
  };

  const handleDialogClose = () => {
    setSelectedRoom(null);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleDialogClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { 
          height: '80vh', 
          maxHeight: 800,
          borderRadius: 3,
          overflow: 'hidden',
          backgroundImage: 'linear-gradient(180deg, rgba(249, 250, 251, 0.95) 0%, rgba(242, 244, 247, 0.95) 100%)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 10px 40px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.8) inset',
          border: '1px solid rgba(255, 255, 255, 0.8)',
        }
      }}
      sx={{
        '& .MuiBackdrop-root': {
          backdropFilter: 'blur(4px)',
          backgroundColor: 'rgba(15, 23, 42, 0.4)',
        }
      }}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        pb: 1.5,
        pt: 2,
        px: 3,
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
        borderBottom: '1px solid rgba(139, 92, 246, 0.15)',
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box sx={{
            width: 40,
            height: 40,
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            boxShadow: '0 4px 12px rgba(139, 92, 246, 0.25)',
          }}>
            <ChatIcon sx={{ color: 'white', fontSize: 22 }} />
          </Box>
          <Typography 
            variant="h6"
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
            }}
          >
            Chat Support
          </Typography>
        </Box>
        <IconButton 
          onClick={handleDialogClose}
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
          <CloseIcon sx={{ color: '#6b7280' }} />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 0, height: 'calc(100% - 70px)' }}>
        <Grid container sx={{ height: '100%' }}>
          {/* Chat Rooms List */}
          <Grid item xs={12} md={4} lg={3} sx={{ 
            borderRight: { md: '1px solid rgba(226, 232, 240, 0.8)' },
            height: '100%',
            position: 'relative',
          }}>
            <Box
              sx={{
                position: 'absolute',
                top: 12,
                left: 12,
                right: 12,
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
          </Grid>

          {/* Chat Window */}
          <Grid item xs={12} md={8} lg={9} sx={{ 
            height: '100%',
            position: 'relative',
          }}>
            <Box
              sx={{
                position: 'absolute',
                top: 12,
                left: 12,
                right: 12,
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
                onClose={handleCloseChat}
              />
            ) : (
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
                  width: 100,
                  height: 100,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '24px',
                  background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
                  boxShadow: '0 5px 15px rgba(139, 92, 246, 0.1)',
                  mb: 3,
                  border: '1px solid rgba(139, 92, 246, 0.2)',
                }}>
                  <ChatIcon sx={{ fontSize: 40, color: '#8b5cf6', opacity: 0.8 }} />
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
                  }}
                >
                  Select a conversation
                </Typography>
                <Typography variant="body1" textAlign="center" sx={{ 
                  maxWidth: 360,
                  lineHeight: 1.8,
                  color: '#6b7280',
                  mb: 4,
                }}>
                  Choose a tenant from the list to view messages and provide support
                </Typography>
              </Box>
            )}
          </Grid>
        </Grid>
      </DialogContent>
    </Dialog>
  );
};

export default ChatDialog;
