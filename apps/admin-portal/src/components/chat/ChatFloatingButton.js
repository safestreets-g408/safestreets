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
          color="primary"
          aria-label="chat"
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            zIndex: 1000,
          }}
        >
          <Badge 
            badgeContent={totalUnreadCount} 
            max={99}
            color="error"
          >
            <ChatIcon />
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
