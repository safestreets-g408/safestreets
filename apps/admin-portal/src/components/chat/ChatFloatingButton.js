import React, { useState } from 'react';
import {
  Fab,
  Badge,
  Tooltip
} from '@mui/material';
import {
  Chat as ChatIcon
} from '@mui/icons-material';
import { useSocket } from '../../context/SocketContext';
import ChatDialog from './ChatDialog';

const ChatFloatingButton = () => {
  const [chatOpen, setChatOpen] = useState(false);
  const { unreadCounts } = useSocket();

  // Calculate total unread messages
  const totalUnreadCount = Object.values(unreadCounts).reduce((sum, count) => sum + count, 0);

  const handleChatToggle = () => {
    setChatOpen(prev => !prev);
  };

  return (
    <>
      <Tooltip title="Open Chat" placement="left">
        <Fab
          onClick={handleChatToggle}
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            zIndex: 1000,
            width: 64,
            height: 64,
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            color: 'white',
            boxShadow: '0 8px 20px rgba(139, 92, 246, 0.4)',
            transition: 'all 0.2s ease-in-out',
            border: '2px solid rgba(255, 255, 255, 0.8)',
            '&:hover': {
              background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
              transform: 'translateY(-4px) scale(1.05)',
              boxShadow: '0 12px 24px rgba(139, 92, 246, 0.5)',
            }
          }}
        >
          <Badge 
            badgeContent={totalUnreadCount} 
            max={99}
            sx={{
              '& .MuiBadge-badge': {
                background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)',
                color: 'white',
                fontWeight: 700,
                fontSize: '0.85rem',
                minWidth: 24,
                height: 24,
                padding: '0 6px',
                border: '2px solid white',
                boxShadow: '0 2px 6px rgba(239, 68, 68, 0.4)',
                animation: totalUnreadCount > 0 ? 'pulse 2s infinite' : 'none',
                '@keyframes pulse': {
                  '0%': { transform: 'scale(1)' },
                  '50%': { transform: 'scale(1.1)' },
                  '100%': { transform: 'scale(1)' }
                }
              }
            }}
          >
            <ChatIcon sx={{ fontSize: 28 }} />
          </Badge>
        </Fab>
      </Tooltip>

      <ChatDialog
        open={chatOpen}
        onClose={() => setChatOpen(false)}
      />
    </>
  );
};

export default ChatFloatingButton;
