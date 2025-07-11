const jwt = require('jsonwebtoken');
const Admin = require('../models/Admin');
const Tenant = require('../models/Tenant');
const FieldWorker = require('../models/FieldWorker');
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
      
      // Add a connection timeout to ensure authentication happens
      const authTimeout = setTimeout(() => {
        if (!socket.userId) {
          console.log('Socket connection timed out without authentication:', socket.id);
          socket.disconnect(true);
        }
      }, 10000); // 10 seconds timeout for authentication
      
      // Clear timeout on disconnect
      socket.on('disconnect', () => {
        clearTimeout(authTimeout);
      });

      // Handle authentication
      socket.on('authenticate', async (token) => {
        console.log('Authentication attempt for socket:', socket.id);
        clearTimeout(authTimeout); // Clear timeout on authentication attempt
        try {
          const decoded = jwt.verify(token, process.env.JWT_SECRET);
          let user;
          
          if (socket.handshake.auth && socket.handshake.auth.userType === 'fieldworker') {
            // This is a field worker token
            if (decoded.fieldWorkerId) {
              user = await FieldWorker.findById(decoded.fieldWorkerId).populate('tenant');
              if (user) {
                const userInfo = {
                  id: user._id.toString(),
                  name: user.name,
                  role: 'field_worker',
                  tenantId: user.tenant?.toString() || null,
                  userType: 'fieldworker'
                };

                // Store user-socket mapping
                this.userSockets.set(userInfo.id, socket.id);
                this.socketUsers.set(socket.id, userInfo);

                socket.userId = userInfo.id;
                socket.userInfo = userInfo;

                // Join a field worker room for notifications
                socket.join(`fieldworker_${userInfo.id}`);
                
                // Join tenant rooms for better message routing
                if (userInfo.tenantId) {
                  socket.join(`tenant_fieldworkers_${userInfo.tenantId}`);
                  socket.join(`tenant_${userInfo.tenantId}`);
                  socket.join(`chat_${userInfo.tenantId}`);
                }
                
                // Join admin chat rooms for more reliable message delivery
                const ChatRoom = require('../models/ChatRoom');
                try {
                  // Find all chat rooms this field worker is part of
                  const chatRooms = await ChatRoom.find({
                    'participants': {
                      $elemMatch: { userId: user._id, userModel: 'FieldWorker' }
                    }
                  });
                  
                  // Join each chat room
                  for (const room of chatRooms) {
                    socket.join(`chat_${room._id}`);
                    
                    // Also join direct admin chat rooms
                    const adminParticipant = room.participants.find(p => p.userModel === 'Admin');
                    if (adminParticipant && adminParticipant.userId) {
                      socket.join(`admin_${adminParticipant.userId.toString()}`);
                    }
                  }
                } catch (err) {
                  console.error('Error joining chat rooms for field worker:', err);
                }

                socket.emit('authenticated', userInfo);
                console.log(`Field Worker authenticated: ${userInfo.name}`);
                return;
              }
            }
          } else if (decoded.adminId) {
            // This is an admin token
            user = await Admin.findById(decoded.adminId).populate('tenant');
            if (user) {
              const userInfo = {
                id: user._id.toString(),
                name: user.name,
                role: user.role,
                tenantId: user.tenant?._id?.toString() || null,
                userType: 'admin'
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
                tenantId: user._id.toString(),
                userType: 'tenant'
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

      // Handle joining chat room for field worker
      socket.on('join_room', async (data) => {
        if (!socket.userInfo) {
          socket.emit('error', 'Not authenticated');
          return;
        }
        
        // Join the requested room
        socket.join(data.room);
        console.log(`User ${socket.userInfo.name} joined room: ${data.room}`);
      });
      
      // Handle field worker typing notifications
      socket.on('typing', async (data) => {
        if (!socket.userInfo) {
          socket.emit('error', 'Not authenticated');
          return;
        }
        
        if (socket.userInfo.userType === 'fieldworker' && data.adminId) {
          // Emit typing notification to admin
          this.io.to(`admin_${data.adminId}`).emit('field_worker_typing', {
            fieldWorkerId: socket.userInfo.id,
            userName: data.userName || socket.userInfo.name,
            isTyping: data.isTyping
          });
        } else if (socket.userInfo.userType === 'admin' && data.fieldWorkerId) {
          // Emit typing notification to field worker
          this.io.to(`fieldworker_${data.fieldWorkerId}`).emit('admin_typing', {
            adminId: socket.userInfo.id,
            userName: data.userName || socket.userInfo.name,
            isTyping: data.isTyping
          });
        }
      });
      
      // Handle marking messages as read for field workers
      socket.on('mark_as_read', async (data) => {
        try {
          if (!socket.userInfo) {
            socket.emit('error', 'Not authenticated');
            return;
          }
          
          // Handle field worker marking admin messages as read
          if (socket.userInfo.userType === 'fieldworker' && data.adminId) {
            const fieldWorkerId = socket.userInfo.id;
            
            // Find the chat room between this field worker and the admin
            const chatRoom = await ChatRoom.findOne({
              participants: {
                $all: [
                  { $elemMatch: { userId: fieldWorkerId, userModel: 'FieldWorker' } },
                  { $elemMatch: { userId: data.adminId, userModel: 'Admin' } }
                ]
              }
            });
            
            if (chatRoom) {
              await ChatMessage.updateMany(
                { 
                  chatId: chatRoom._id,
                  senderId: data.adminId,
                  senderModel: 'Admin',
                  'readBy.userId': { $ne: fieldWorkerId }
                },
                { 
                  $push: { 
                    readBy: { 
                      userId: fieldWorkerId,
                      userModel: 'FieldWorker',
                      readAt: new Date()
                    } 
                  } 
                }
              );
              
              // Broadcast to the admin that messages were read
              this.io.to(`admin_${data.adminId}`).emit('messages_read_by_field_worker', {
                fieldWorkerId,
                adminId: data.adminId,
                chatRoomId: chatRoom._id
              });
              
              console.log(`Field worker ${fieldWorkerId} marked messages from admin ${data.adminId} as read`);
            }
          }
        } catch (error) {
          console.error('Error marking messages as read (field worker):', error);
        }
      });
      
      // Add ping-pong for connection testing
      socket.on('ping_server', () => {
        console.log(`Ping received from socket ${socket.id}`);
        socket.emit('pong_server', { timestamp: new Date().toISOString() });
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

      // Handle direct_message event from client (backup for message delivery)
      socket.on('direct_message', async (messageData) => {
        try {
          if (!socket.userInfo) {
            socket.emit('error', 'Not authenticated');
            return;
          }
          
          console.log('Received direct_message:', messageData);
          
          // Forward the message to relevant rooms
          if (messageData.adminId) {
            this.io.to(`admin_${messageData.adminId}`).emit('new_message', messageData.message);
            console.log(`Forwarded direct message to admin_${messageData.adminId}`);
            
            // Also send a notification
            if (messageData.forceNotify) {
              this.io.to(`admin_${messageData.adminId}`).emit('chat_notification', {
                type: 'new_message',
                senderId: messageData.message.senderId,
                senderName: messageData.message.senderName || 'Field Worker',
                message: messageData.message.message.length > 30 ? 
                  `${messageData.message.message.substring(0, 30)}...` : 
                  messageData.message.message,
                timestamp: new Date()
              });
            }
          }
          
          // If we have info about tenant ID, notify all admins for this tenant
          if (socket.userInfo.tenantId) {
            this.io.to(`chat_${socket.userInfo.tenantId}`).emit('new_message', messageData.message);
            console.log(`Forwarded direct message to chat_${socket.userInfo.tenantId}`);
          }
          
          // Also notify super admins
          this.io.to('super_admin').emit('new_message', messageData.message);
        } catch (error) {
          console.error('Error handling direct_message:', error);
        }
      });
      
      // Handle manual message forwarding (backup for message delivery)
      socket.on('manual_message', async (messageData) => {
        try {
          if (!socket.userInfo) {
            socket.emit('error', 'Not authenticated');
            return;
          }
          
          console.log('Received manual_message:', messageData);
          
          // Forward the message to relevant rooms
          if (messageData.adminId) {
            this.io.to(`admin_${messageData.adminId}`).emit('new_message', messageData);
          }
          
          if (messageData.chatRoomId) {
            this.io.to(`chat_${messageData.chatRoomId}`).emit('new_message', messageData);
          }
          
          // Also forward to tenant room if available
          if (socket.userInfo.tenantId) {
            this.io.to(`chat_${socket.userInfo.tenantId}`).emit('new_message', messageData);
          }
        } catch (error) {
          console.error('Error handling manual message:', error);
        }
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
