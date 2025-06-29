const jwt = require('jsonwebtoken');
const Admin = require('../models/Admin');
const Tenant = require('../models/Tenant');
const ChatMessage = require('../models/ChatMessage');
const ChatRoom = require('../models/ChatRoom');

class SocketManager {
  constructor(io) {
    this.io = io;
    this.userSockets = new Map(); // Map userId to socket.id
    this.socketUsers = new Map(); // Map socket.id to user info
    
    this.setupSocketHandlers();
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log('User connected:', socket.id);

      // Handle authentication
      socket.on('authenticate', async (token) => {
        try {
          const decoded = jwt.verify(token, process.env.JWT_SECRET);
          let user;

          if (decoded.adminId) {
            // This is an admin token
            user = await Admin.findById(decoded.adminId).populate('tenant');
            if (user) {
              const userInfo = {
                id: user._id.toString(),
                name: user.name,
                role: user.role,
                tenantId: user.tenant?._id?.toString() || null
              };

              // Store user-socket mapping
              this.userSockets.set(userInfo.id, socket.id);
              this.socketUsers.set(socket.id, userInfo);

              socket.userId = userInfo.id;
              socket.userInfo = userInfo;

              // Join appropriate rooms
              if (userInfo.role === 'super-admin') {
                socket.join('super_admin');
                // Join all tenant chat rooms for super admin
                const chatRooms = await ChatRoom.find({ isActive: true });
                chatRooms.forEach(room => {
                  socket.join(`chat_${room.tenantId}`);
                });
              } else if (userInfo.tenantId) {
                socket.join(`chat_${userInfo.tenantId}`);
              }

              socket.emit('authenticated', userInfo);
              console.log(`Admin authenticated: ${userInfo.name} (${userInfo.role})`);
            }
          } else {
            // This might be a tenant token
            user = await Tenant.findById(decoded.id);
            if (user) {
              const userInfo = {
                id: user._id.toString(),
                name: user.name,
                role: 'tenant_admin',
                tenantId: user._id.toString()
              };

              // Store user-socket mapping
              this.userSockets.set(userInfo.id, socket.id);
              this.socketUsers.set(socket.id, userInfo);

              socket.userId = userInfo.id;
              socket.userInfo = userInfo;

              socket.join(`chat_${userInfo.tenantId}`);

              socket.emit('authenticated', userInfo);
              console.log(`Tenant authenticated: ${userInfo.name} (${userInfo.role})`);
            }
          }

          if (!user) {
            socket.emit('auth_error', 'User not found');
            return;
          }

        } catch (error) {
          console.error('Authentication error:', error);
          socket.emit('auth_error', 'Invalid token');
        }
      });

      // Handle joining chat room
      socket.on('join_chat', async (tenantId) => {
        if (!socket.userInfo) {
          socket.emit('error', 'Not authenticated');
          return;
        }

        // Check if admin has access to this chat
        if (socket.userInfo.role !== 'super-admin' && socket.userInfo.tenantId !== tenantId) {
          socket.emit('error', 'Access denied');
          return;
        }

        socket.join(`chat_${tenantId}`);
        console.log(`User ${socket.userInfo.name} joined chat for tenant ${tenantId}`);
      });

      // Handle sending messages
      socket.on('send_message', async (data) => {
        try {
          if (!socket.userInfo) {
            socket.emit('error', 'Not authenticated');
            return;
          }

          const { tenantId, message, messageType = 'text', attachmentUrl } = data;

          // Check if admin has access to this chat
          if (socket.userInfo.role !== 'super-admin' && socket.userInfo.tenantId !== tenantId) {
            socket.emit('error', 'Access denied');
            return;
          }

          const chatId = `tenant_${tenantId}`;

          // Create message
          const chatMessage = new ChatMessage({
            chatId,
            senderId: socket.userInfo.id,
            senderModel: 'Admin',
            senderName: socket.userInfo.name,
            senderRole: socket.userInfo.role === 'super-admin' ? 'super_admin' : 'tenant_admin',
            message,
            messageType,
            attachmentUrl
          });

          await chatMessage.save();

          // Update chat room with last message
          await ChatRoom.findOneAndUpdate(
            { tenantId },
            {
              lastMessage: {
                message,
                senderId: socket.userInfo.id,
                senderName: socket.userInfo.name,
                timestamp: new Date()
              },
              $inc: {
                [`unreadCount.${socket.userInfo.role === 'super-admin' ? 'tenantAdmin' : 'superAdmin'}`]: 1
              }
            }
          );

          // Emit message to all users in the chat room
          this.io.to(`chat_${tenantId}`).emit('new_message', {
            ...chatMessage.toObject(),
            timestamp: chatMessage.createdAt
          });

          // Emit notification to super admin if message is from tenant admin
          if (socket.userInfo.role !== 'super-admin') {
            this.io.to('super_admin').emit('chat_notification', {
              tenantId,
              tenantName: (await ChatRoom.findOne({ tenantId }))?.tenantName,
              message,
              senderName: socket.userInfo.name,
              timestamp: new Date()
            });
          }

        } catch (error) {
          console.error('Error sending message:', error);
          socket.emit('error', 'Failed to send message');
        }
      });

      // Handle typing indicators
      socket.on('typing', (data) => {
        if (!socket.userInfo) return;
        
        const { tenantId, isTyping } = data;
        socket.to(`chat_${tenantId}`).emit('user_typing', {
          userId: socket.userInfo.id,
          userName: socket.userInfo.name,
          isTyping
        });
      });

      // Handle marking messages as read
      socket.on('mark_read', async (tenantId) => {
        try {
          if (!socket.userInfo) return;

          const chatId = `tenant_${tenantId}`;

          // Mark messages as read
          await ChatMessage.updateMany(
            { 
              chatId,
              senderId: { $ne: socket.userInfo.id },
              isRead: false
            },
            { 
              isRead: true,
              $push: {
                readBy: {
                  userId: socket.userInfo.id,
                  readAt: new Date()
                }
              }
            }
          );

          // Reset unread count for this user
          const updateField = socket.userInfo.role === 'super-admin' ? 'unreadCount.superAdmin' : 'unreadCount.tenantAdmin';
          await ChatRoom.findOneAndUpdate(
            { tenantId },
            { [updateField]: 0 }
          );

          // Notify other users that messages were read
          socket.to(`chat_${tenantId}`).emit('messages_read', {
            userId: socket.userInfo.id,
            tenantId
          });

        } catch (error) {
          console.error('Error marking messages as read:', error);
        }
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        if (socket.userInfo) {
          this.userSockets.delete(socket.userInfo.id);
          this.socketUsers.delete(socket.id);
          console.log(`User disconnected: ${socket.userInfo.name}`);
        } else {
          console.log('Unknown user disconnected:', socket.id);
        }
      });
    });
  }

  // Method to send notification to specific user
  sendNotificationToUser(userId, notification) {
    const socketId = this.userSockets.get(userId);
    if (socketId) {
      this.io.to(socketId).emit('notification', notification);
    }
  }

  // Method to send notification to all super admins
  sendNotificationToSuperAdmins(notification) {
    this.io.to('super_admin').emit('notification', notification);
  }
}

module.exports = SocketManager;
