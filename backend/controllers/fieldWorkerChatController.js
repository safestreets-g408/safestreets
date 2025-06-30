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
    
    if (recipientType !== 'admin') {
      return res.status(400).json({ message: 'Invalid recipient type' });
    }
    
    // Find the admin
    const admin = await Admin.findById(recipientId);
    if (!admin) {
      return res.status(404).json({ message: 'Admin not found' });
    }
    
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
      chatRoom = new ChatRoom({
        roomType: 'admin_fieldworker',
        participants: [
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
        ]
      });
      
      await chatRoom.save();
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
    
    // Emit socket event (handled by socket.io setup)
    
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
