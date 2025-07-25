const Admin = require('../models/Admin');
const FieldWorker = require('../models/FieldWorker');
const ChatRoom = require('../models/ChatRoom');
const ChatMessage = require('../models/ChatMessage');
const DamageReport = require('../models/DamageReport');

// Get list of admins the field worker can chat with
const getAdminChatList = async (req, res) => {
  try {
    const { fieldWorker } = req;
    
    // Get the tenant ID that the field worker belongs to
    const tenantId = fieldWorker.tenant;
    
    // Find all admins for this tenant plus super-admins
    const admins = await Admin.find({ 
      $or: [
        { tenant: tenantId },
        { role: 'super-admin' }
      ]
    }).select('_id name email role profileImage tenant createdAt');
    
    // Get all chat rooms between this field worker and admins
    const chatRooms = await ChatRoom.find({
      'participants': {
        $elemMatch: { userId: fieldWorker._id, userModel: 'FieldWorker' }
      }
    });
    
    // Create a map of admin IDs to chat rooms
    const adminChatRooms = {};
    chatRooms.forEach(room => {
      // Find the admin participant in this room
      const adminParticipant = room.participants.find(p => 
        p.userModel === 'Admin'
      );
      
      if (adminParticipant) {
        adminChatRooms[adminParticipant.userId.toString()] = room._id;
      }
    });
    
    // Get unread message counts for each admin
    const unreadCounts = {};
    await Promise.all(
      Object.entries(adminChatRooms).map(async ([adminId, roomId]) => {
        const count = await ChatMessage.countDocuments({
          chatId: roomId,
          senderId: adminId,
          senderModel: 'Admin',
          readBy: { $not: { $elemMatch: { userId: fieldWorker._id } } }
        });
        unreadCounts[adminId] = count;
      })
    );
    
    // Get last message for each admin chat
    const lastMessages = {};
    await Promise.all(
      Object.entries(adminChatRooms).map(async ([adminId, roomId]) => {
        const message = await ChatMessage.findOne({
          chatId: roomId
        }).sort({ createdAt: -1 });
        
        if (message) {
          lastMessages[adminId] = {
            message: message.message,
            timestamp: message.createdAt
          };
        }
      })
    );
    
    // Transform admin data to include online status, unread counts, etc.
    const adminsList = admins.map(admin => {
      const adminId = admin._id.toString();
      return {
        _id: adminId,
        name: admin.name,
        role: admin.role,
        profileImage: admin.profileImage,
        isOnline: false, // This would be set by socket.io in real app
        unreadCount: unreadCounts[adminId] || 0,
        lastMessage: lastMessages[adminId] || null,
        hasChat: !!adminChatRooms[adminId]
      };
    });
    
    res.json(adminsList);
  } catch (error) {
    console.error('Error getting admin chat list:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Get or create chat room with admin
const getChatRoom = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { recipientId, recipientType } = req.body;
    
    // Validate fieldWorker
    if (!fieldWorker || !fieldWorker._id) {
      console.error('Invalid fieldWorker in getChatRoom:', fieldWorker);
      return res.status(400).json({ message: 'FieldWorker not found in request' });
    }
    
    if (recipientType !== 'admin') {
      return res.status(400).json({ message: 'Invalid recipient type' });
    }
    
    // Validate recipientId
    if (!recipientId) {
      return res.status(400).json({ message: 'recipientId is required' });
    }
    
    // Find the admin
    const admin = await Admin.findById(recipientId);
    if (!admin) {
      return res.status(404).json({ message: 'Admin not found' });
    }
    
    console.log('Creating chat room with fieldWorker:', fieldWorker._id, 'and admin:', admin._id);
    
    // Check if fieldWorker has access to this admin (tenant check)
    if (admin.role !== 'super-admin' && admin.tenant.toString() !== fieldWorker.tenant.toString()) {
      return res.status(403).json({ message: 'You do not have permission to chat with this admin' });
    }
    
    // Look for existing chat room
    let chatRoom = await ChatRoom.findOne({
      participants: {
        $all: [
          { $elemMatch: { userId: fieldWorker._id, userModel: 'FieldWorker' } },
          { $elemMatch: { userId: admin._id, userModel: 'Admin' } }
        ]
      }
    });
    
    // If no chat room exists, create one
    if (!chatRoom) {
      // Validate participant data before creating chat room
      if (!fieldWorker._id || !fieldWorker.name) {
        console.error('Invalid fieldWorker data for chat room creation:', { id: fieldWorker._id, name: fieldWorker.name });
        return res.status(400).json({ message: 'Invalid fieldWorker data' });
      }
      
      if (!admin._id || !admin.name) {
        console.error('Invalid admin data for chat room creation:', { id: admin._id, name: admin.name });
        return res.status(400).json({ message: 'Invalid admin data' });
      }
      
      const participantData = [
        {
          userId: fieldWorker._id,
          userModel: 'FieldWorker',
          name: fieldWorker.name,
          role: 'field_worker'
        },
        {
          userId: admin._id,
          userModel: 'Admin',
          name: admin.name,
          role: admin.role === 'super-admin' ? 'super_admin' : 'tenant_admin'
        }
      ];
      
      console.log('Creating chat room with participants:', participantData);
      
      chatRoom = new ChatRoom({
        roomType: 'admin_fieldworker',
        participants: participantData
      });
      
      await chatRoom.save();
      console.log('Chat room created successfully:', chatRoom._id);
    }
    
    res.json({ 
      roomId: chatRoom._id,
      participants: chatRoom.participants 
    });
  } catch (error) {
    console.error('Error getting chat room:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Get messages for a chat with an admin
const getChatMessages = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { adminId } = req.params;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 50;
    const skip = (page - 1) * limit;
    
    // Find the chat room
    const chatRoom = await ChatRoom.findOne({
      participants: {
        $all: [
          { $elemMatch: { userId: fieldWorker._id, userModel: 'FieldWorker' } },
          { $elemMatch: { userId: adminId, userModel: 'Admin' } }
        ]
      }
    });
    
    if (!chatRoom) {
      return res.status(404).json({ message: 'Chat not found' });
    }
    
    // Get messages
    const messages = await ChatMessage.find({ chatId: chatRoom._id })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit + 1); // Get one extra to check if there are more
    
    const hasMore = messages.length > limit;
    const chatMessages = hasMore ? messages.slice(0, limit) : messages;
    
    // Mark messages as read
    await ChatMessage.updateMany(
      { 
        chatId: chatRoom._id,
        senderId: adminId,
        senderModel: 'Admin',
        'readBy.userId': { $ne: fieldWorker._id }
      },
      { 
        $push: { 
          readBy: { 
            userId: fieldWorker._id,
            userModel: 'FieldWorker',
            readAt: new Date()
          } 
        } 
      }
    );
    
    res.json({
      messages: chatMessages.reverse(), // Return in chronological order
      hasMore,
      page,
      limit
    });
  } catch (error) {
    console.error('Error getting chat messages:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Send message to an admin
const sendMessage = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { adminId } = req.params;
    const { message, messageType = 'text' } = req.body;
    
    if (!message) {
      return res.status(400).json({ message: 'Message content is required' });
    }
    
    // Find the chat room
    const chatRoom = await ChatRoom.findOne({
      participants: {
        $all: [
          { $elemMatch: { userId: fieldWorker._id, userModel: 'FieldWorker' } },
          { $elemMatch: { userId: adminId, userModel: 'Admin' } }
        ]
      }
    });
    
    if (!chatRoom) {
      return res.status(404).json({ message: 'Chat not found' });
    }
    
    // Create message
    const chatMessage = new ChatMessage({
      chatId: chatRoom._id,
      message,
      messageType,
      senderId: fieldWorker._id,
      senderModel: 'FieldWorker',
      senderName: fieldWorker.name,
      senderRole: 'field_worker',
      readBy: [
        {
          userId: fieldWorker._id,
          userModel: 'FieldWorker',
          readAt: new Date()
        }
      ]
    });
    
    await chatMessage.save();
    
    // Update chat room with last message
    chatRoom.lastMessage = chatMessage._id;
    chatRoom.lastMessageAt = chatMessage.createdAt;
    await chatRoom.save();
    
    // Handle socket.io notifications
    if (req.io) {
      try {
        console.log(`Emitting new_message events for message: ${chatMessage._id}`);
        
        // Get admin details for more targeting
        const Admin = require('../models/Admin');
        const adminUser = await Admin.findById(adminId).select('tenant role');
        
        // Create serializable message object with all required fields
        const messageObj = {
          ...chatMessage.toObject(),
          _id: chatMessage._id.toString(),
          senderId: chatMessage.senderId.toString(),
          chatId: chatMessage.chatId.toString(),
          // Add explicit fields that may be needed by the admin portal
          senderName: fieldWorker.name,
          tenantId: fieldWorker.tenant ? fieldWorker.tenant.toString() : null,
          senderRole: 'field_worker',
          adminId: adminId,
          fromFieldWorker: true
        };
        
        // Use a single consistent event name
        const eventName = 'new_message';
        
        // Emit to admin-specific room - primary target
        req.io.to(`admin_${adminId}`).emit(eventName, messageObj);
        console.log(`Emitted to admin_${adminId}`);
        
        // Emit to chat room
        req.io.to(`chat_${chatRoom._id}`).emit(eventName, messageObj);
        console.log(`Emitted to chat_${chatRoom._id}`);
        
        // ALWAYS emit to super_admin room for visibility - this is crucial
        req.io.to('super_admin').emit(eventName, messageObj);
        console.log('Emitted to super_admin room');
        
        // If admin has a tenant, emit to that tenant's room
        if (adminUser && adminUser.tenant) {
          const tenantId = adminUser.tenant.toString ? adminUser.tenant.toString() : adminUser.tenant;
          req.io.to(`chat_${tenantId}`).emit(eventName, messageObj);
          req.io.to(`tenant_${tenantId}`).emit(eventName, messageObj);
          console.log(`Emitted to tenant rooms for tenant ${tenantId}`);
        }
        
        // Also broadcast to the field worker's tenant room
        if (fieldWorker.tenant) {
          const fwTenantId = fieldWorker.tenant.toString ? fieldWorker.tenant.toString() : fieldWorker.tenant;
          req.io.to(`tenant_fieldworkers_${fwTenantId}`).emit(eventName, messageObj);
          req.io.to(`tenant_${fwTenantId}`).emit(eventName, messageObj);
          req.io.to(`chat_${fwTenantId}`).emit(eventName, messageObj); // Additional broadcast
          console.log(`Emitted to field worker tenant rooms for ${fwTenantId}`);
        }
        
        // Send a notification event for admin portal to update UI
        const notificationData = {
          type: 'new_message',
          senderId: fieldWorker._id.toString(),
          senderName: fieldWorker.name,
          tenantId: fieldWorker.tenant ? fieldWorker.tenant.toString() : null,
          message: message.length > 30 ? `${message.substring(0, 30)}...` : message,
          chatId: chatRoom._id.toString(),
          timestamp: new Date(),
          adminId: adminId,
          directMessage: true,
          fromFieldWorker: true
        };
        
        // Emit notification to all relevant targets
        req.io.to(`admin_${adminId}`).emit('chat_notification', notificationData);
        req.io.to('super_admin').emit('chat_notification', notificationData);
        
        // Also send to all admins in the tenant if applicable
        if (fieldWorker.tenant) {
          req.io.to(`tenant_${fieldWorker.tenant}`).emit('chat_notification', notificationData);
        }
        
        // Also broadcast to all connected clients (will be filtered on client side)
        req.io.emit('global_message', {
          type: 'admin_message',
          targetAdmin: adminId,
          message: messageObj,
          notification: notificationData
        });
        
        console.log('Socket notifications complete');
      } catch (socketErr) {
        console.error('Error sending socket notifications:', socketErr);
      }
    } else {
      console.warn('Socket.io not available - message sent but notification not emitted');
    }
    
    res.json(chatMessage);
  } catch (error) {
    console.error('Error sending message:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Mark messages as read
const markMessagesAsRead = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { adminId } = req.params;
    
    // Find the chat room
    const chatRoom = await ChatRoom.findOne({
      participants: {
        $all: [
          { $elemMatch: { userId: fieldWorker._id, userModel: 'FieldWorker' } },
          { $elemMatch: { userId: adminId, userModel: 'Admin' } }
        ]
      }
    });
    
    if (!chatRoom) {
      return res.status(404).json({ message: 'Chat not found' });
    }
    
    // Mark messages as read
    const result = await ChatMessage.updateMany(
      { 
        chatId: chatRoom._id,
        senderId: adminId,
        senderModel: 'Admin',
        'readBy.userId': { $ne: fieldWorker._id }
      },
      { 
        $push: { 
          readBy: { 
            userId: fieldWorker._id,
            userModel: 'FieldWorker',
            readAt: new Date()
          } 
        } 
      }
    );
    
    res.json({ 
      success: true,
      messagesMarkedAsRead: result.nModified || 0
    });
  } catch (error) {
    console.error('Error marking messages as read:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Share a damage report with an admin
const shareReport = async (req, res) => {
  try {
    const { fieldWorker } = req;
    const { adminId } = req.params;
    const { reportId } = req.body;
    
    if (!reportId) {
      return res.status(400).json({ message: 'Report ID is required' });
    }
    
    // Find the chat room
    const chatRoom = await ChatRoom.findOne({
      participants: {
        $all: [
          { $elemMatch: { userId: fieldWorker._id, userModel: 'FieldWorker' } },
          { $elemMatch: { userId: adminId, userModel: 'Admin' } }
        ]
      }
    });
    
    if (!chatRoom) {
      return res.status(404).json({ message: 'Chat not found' });
    }
    
    // Find the damage report
    const report = await DamageReport.findById(reportId);
    if (!report) {
      return res.status(404).json({ message: 'Damage report not found' });
    }
    
    // Check if field worker has access to this report
    if (report.fieldWorkerId.toString() !== fieldWorker._id.toString()) {
      return res.status(403).json({ message: 'You do not have permission to share this report' });
    }
    
    // Create report summary
    const reportSummary = {
      reportId: report._id,
      damageType: report.damageType,
      severity: report.severity,
      status: report.status,
      location: report.location?.address || 'Unknown location',
      createdAt: report.createdAt
    };
    
    // Create message with report data
    const message = `__REPORT_JSON__:${JSON.stringify(reportSummary)}`;
    
    const chatMessage = new ChatMessage({
      chatId: chatRoom._id,
      message,
      messageType: 'text', // Use text type with special prefix
      senderId: fieldWorker._id,
      senderModel: 'FieldWorker',
      senderName: fieldWorker.name,
      senderRole: 'field_worker',
      metadata: {
        reportId: report._id
      },
      readBy: [
        {
          userId: fieldWorker._id,
          userModel: 'FieldWorker',
          readAt: new Date()
        }
      ]
    });
    
    await chatMessage.save();
    
    // Update chat room with last message
    chatRoom.lastMessage = chatMessage._id;
    chatRoom.lastMessageAt = chatMessage.createdAt;
    await chatRoom.save();
    
    res.json(chatMessage);
  } catch (error) {
    console.error('Error sharing report:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

module.exports = {
  getAdminChatList,
  getChatRoom,
  getChatMessages,
  sendMessage,
  markMessagesAsRead,
  shareReport
};
