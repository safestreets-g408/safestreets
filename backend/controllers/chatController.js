const ChatMessage = require('../models/ChatMessage');
const ChatRoom = require('../models/ChatRoom');
const Admin = require('../models/Admin');
const Tenant = require('../models/Tenant');

// Get chat room for a tenant or create if doesn't exist
const getChatRoom = async (req, res) => {
  try {
    const { tenantId } = req.params;
    const { admin } = req;

    console.log('getChatRoom - Admin:', admin.name, 'Role:', admin.role, 'Tenant ID requested:', tenantId);
    console.log('getChatRoom - Admin tenant:', admin.tenant);

    // Check if admin has access to this chat
    if (admin.role !== 'super-admin') {
      // For non-super admin, check if they belong to this tenant
      // Handle both populated and unpopulated tenant field
      const adminTenantId = admin.tenant?._id?.toString() || admin.tenant?.toString();
      if (!adminTenantId || adminTenantId !== tenantId) {
        return res.status(403).json({ message: 'Access denied - tenant mismatch' });
      }
    }

    let chatRoom = await ChatRoom.findOne({ tenantId });

    if (!chatRoom) {
      // Get tenant details
      const tenant = await Tenant.findById(tenantId);
      if (!tenant) {
        return res.status(404).json({ message: 'Tenant not found' });
      }

      console.log('Creating new chat room for tenant:', tenant.name);

      // Create new chat room
      chatRoom = new ChatRoom({
        tenantId,
        tenantName: tenant.name,
        participants: []
      });

      // Add super admin to participants if exists
      const superAdmin = await Admin.findOne({ role: 'super-admin' });
      if (superAdmin) {
        chatRoom.participants.push({
          userId: superAdmin._id,
          userModel: 'Admin',
          name: superAdmin.name,
          role: 'super_admin'
        });
      }

      // Add tenant admin to participants (find tenant admin for this tenant)
      const tenantAdmin = await Admin.findOne({ tenant: tenantId, role: { $in: ['admin', 'tenant-owner'] } });
      if (tenantAdmin) {
        chatRoom.participants.push({
          userId: tenantAdmin._id,
          userModel: 'Admin',
          name: tenantAdmin.name,
          role: 'tenant_admin'
        });
      }

      await chatRoom.save();
      console.log('Chat room created successfully');
    }

    res.json(chatRoom);
  } catch (error) {
    console.error('Error getting chat room:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get all chat rooms (for super admin)
const getAllChatRooms = async (req, res) => {
  try {
    const { admin } = req;

    console.log('getAllChatRooms - Admin:', admin.name, 'Role:', admin.role);

    if (admin.role !== 'super-admin') {
      return res.status(403).json({ message: 'Access denied - super admin required' });
    }

    // Get traditional tenant chat rooms
    const chatRooms = await ChatRoom.find({ isActive: true })
      .sort({ 'lastMessage.timestamp': -1, updatedAt: -1 });

    // Get field worker chat rooms (admin_fieldworker type)
    const fieldWorkerChats = await ChatRoom.find({ 
      roomType: 'admin_fieldworker',
      isActive: { $ne: false }
    }).populate({
      path: 'participants.userId',
      select: 'name email role tenant profileImage'
    }).sort({ lastMessageAt: -1, updatedAt: -1 });

    // Transform field worker chats to match expected format
    const transformedFieldWorkerChats = await Promise.all(
      fieldWorkerChats.map(async (room) => {
        const fieldWorkerParticipant = room.participants.find(p => p.userModel === 'FieldWorker');
        const adminParticipant = room.participants.find(p => p.userModel === 'Admin');
        
        if (!fieldWorkerParticipant) return null;
        
        // Get the actual field worker document
        const FieldWorker = require('../models/FieldWorker');
        const fieldWorker = await FieldWorker.findById(fieldWorkerParticipant.userId).populate('tenant', 'name');
        
        if (!fieldWorker) return null;
        
        // Get last message
        const lastMessage = await ChatMessage.findOne({ chatId: room._id })
          .sort({ createdAt: -1 })
          .select('message createdAt senderId senderName senderModel');
        
        // Get unread count for admin
        const unreadCount = await ChatMessage.countDocuments({
          chatId: room._id,
          senderModel: 'FieldWorker',
          readBy: { $not: { $elemMatch: { userId: admin._id } } }
        });
        
        return {
          _id: room._id,
          tenantId: fieldWorker.tenant?._id || fieldWorker.tenant,
          tenantName: fieldWorker.tenant?.name || 'Unknown Tenant',
          roomType: 'field_worker_chat',
          fieldWorker: {
            _id: fieldWorker._id,
            name: fieldWorker.name,
            email: fieldWorker.email,
            profileImage: fieldWorker.profileImage
          },
          admin: adminParticipant ? {
            _id: adminParticipant.userId,
            name: adminParticipant.name,
            role: adminParticipant.role
          } : null,
          lastMessage: lastMessage ? {
            message: lastMessage.message,
            timestamp: lastMessage.createdAt,
            senderName: lastMessage.senderName,
            senderModel: lastMessage.senderModel
          } : null,
          unreadCount,
          isActive: true,
          updatedAt: room.lastMessageAt || room.updatedAt
        };
      })
    );
    
    // Filter out null results and combine with regular chat rooms
    const validFieldWorkerChats = transformedFieldWorkerChats.filter(chat => chat !== null);
    const allChats = [...chatRooms, ...validFieldWorkerChats];
    
    // Sort by last activity
    allChats.sort((a, b) => {
      const aTime = a.lastMessage?.timestamp || a.updatedAt || new Date(0);
      const bTime = b.lastMessage?.timestamp || b.updatedAt || new Date(0);
      return new Date(bTime) - new Date(aTime);
    });

    console.log('Found chat rooms:', chatRooms.length, 'field worker chats:', validFieldWorkerChats.length);
    res.json(allChats);
  } catch (error) {
    console.error('Error getting chat rooms:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get chat messages for a room
const getChatMessages = async (req, res) => {
  try {
    const { tenantId } = req.params;
    const { page = 1, limit = 50 } = req.query;
    const { admin } = req;

    console.log('getChatMessages - Admin:', admin.name, 'Role:', admin.role, 'Tenant ID:', tenantId);
    console.log('getChatMessages - Admin tenant:', admin.tenant?._id || admin.tenant);

    // Check if admin has access to this chat
    if (admin.role !== 'super-admin') {
      // Handle both populated and unpopulated tenant field
      const adminTenantId = admin.tenant?._id?.toString() || admin.tenant?.toString();
      if (!adminTenantId || adminTenantId !== tenantId) {
        return res.status(403).json({ message: 'Access denied - tenant mismatch' });
      }
    }

    const chatId = `tenant_${tenantId}`;
    const skip = (page - 1) * limit;

    const messages = await ChatMessage.find({ chatId })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit));

    const totalMessages = await ChatMessage.countDocuments({ chatId });

    console.log('Found messages:', messages.length);

    res.json({
      messages: messages.reverse(), // Reverse to show oldest first
      totalMessages,
      currentPage: parseInt(page),
      totalPages: Math.ceil(totalMessages / limit),
      hasMore: skip + messages.length < totalMessages
    });
  } catch (error) {
    console.error('Error getting chat messages:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get chat messages for field worker room
const getFieldWorkerChatMessages = async (req, res) => {
  try {
    const { roomId } = req.params;
    const { page = 1, limit = 50 } = req.query;
    const { admin } = req;

    console.log('getFieldWorkerChatMessages - Admin:', admin.name, 'Room ID:', roomId);

    // Check if this is a valid field worker chat room
    const chatRoom = await ChatRoom.findById(roomId);
    if (!chatRoom || chatRoom.roomType !== 'admin_fieldworker') {
      return res.status(404).json({ message: 'Field worker chat room not found' });
    }

    // Check if admin has access (super admin or participant in the room)
    if (admin.role !== 'super-admin') {
      const isParticipant = chatRoom.participants.some(p => 
        p.userModel === 'Admin' && p.userId.toString() === admin._id.toString()
      );
      if (!isParticipant) {
        return res.status(403).json({ message: 'Access denied' });
      }
    }

    const skip = (page - 1) * limit;

    const messages = await ChatMessage.find({ chatId: roomId })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit));

    const totalMessages = await ChatMessage.countDocuments({ chatId: roomId });

    console.log('Found field worker chat messages:', messages.length);

    res.json({
      messages: messages.reverse(), // Reverse to show oldest first
      totalMessages,
      currentPage: parseInt(page),
      totalPages: Math.ceil(totalMessages / limit),
      hasMore: skip + messages.length < totalMessages
    });
  } catch (error) {
    console.error('Error getting field worker chat messages:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Send a message
const sendMessage = async (req, res) => {
  try {
    const { tenantId } = req.params;
    const { message, messageType = 'text', attachmentUrl } = req.body;
    const { admin } = req;

    console.log('sendMessage - Admin:', admin.name, 'Role:', admin.role, 'Tenant ID requested:', tenantId);
    console.log('sendMessage - Admin tenant:', admin.tenant?._id || admin.tenant);
    
    // Check if admin has access to this chat
    if (admin.role !== 'super-admin') {
      // Handle both populated and unpopulated tenant field
      const adminTenantId = admin.tenant?._id?.toString() || admin.tenant?.toString();
      if (!adminTenantId || adminTenantId !== tenantId) {
        return res.status(403).json({ message: 'Access denied - tenant mismatch' });
      }
    }

    const chatId = `tenant_${tenantId}`;

    // Create message
    const chatMessage = new ChatMessage({
      chatId,
      senderId: admin._id,
      senderModel: 'Admin',
      senderName: admin.name,
      senderRole: admin.role === 'super-admin' ? 'super_admin' : 'tenant_admin',
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
          senderId: admin._id,
          senderName: admin.name,
          timestamp: new Date()
        },
        $inc: {
          [`unreadCount.${admin.role === 'super-admin' ? 'tenantAdmin' : 'superAdmin'}`]: 1
        }
      }
    );

    // Broadcast message via socket if available
    if (req.io) {
      req.io.to(`chat_${tenantId}`).emit('new_message', {
        ...chatMessage.toObject(),
        timestamp: chatMessage.createdAt
      });
    }

    res.status(201).json(chatMessage);
  } catch (error) {
    console.error('Error sending message:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Mark messages as read
const markMessagesAsRead = async (req, res) => {
  try {
    const { tenantId } = req.params;
    const { admin } = req;

    console.log('markMessagesAsRead - Admin:', admin.name, 'Role:', admin.role, 'Tenant ID:', tenantId);
    console.log('markMessagesAsRead - Admin tenant:', admin.tenant?._id || admin.tenant);

    // Check if admin has access to this chat
    if (admin.role !== 'super-admin') {
      // Handle both populated and unpopulated tenant field
      const adminTenantId = admin.tenant?._id?.toString() || admin.tenant?.toString();
      if (!adminTenantId || adminTenantId !== tenantId) {
        return res.status(403).json({ message: 'Access denied - tenant mismatch' });
      }
    }

    const chatId = `tenant_${tenantId}`;

    // Mark messages as read
    await ChatMessage.updateMany(
      { 
        chatId,
        senderId: { $ne: admin._id },
        isRead: false
      },
      { 
        isRead: true,
        $push: {
          readBy: {
            userId: admin._id,
            readAt: new Date()
          }
        }
      }
    );

    // Reset unread count for this admin
    const updateField = admin.role === 'super-admin' ? 'unreadCount.superAdmin' : 'unreadCount.tenantAdmin';
    await ChatRoom.findOneAndUpdate(
      { tenantId },
      { [updateField]: 0 }
    );

    res.json({ message: 'Messages marked as read' });
  } catch (error) {
    console.error('Error marking messages as read:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Get tenant admin's chat rooms (shows super admin contact)
const getTenantChatRooms = async (req, res) => {
  try {
    const { admin } = req;

    console.log('getTenantChatRooms - Admin:', admin.name, 'Role:', admin.role);

    if (admin.role === 'super-admin') {
      return res.status(403).json({ message: 'This endpoint is for tenant admins only' });
    }

    const tenantId = admin.tenant?._id;
    if (!tenantId) {
      return res.status(400).json({ message: 'No tenant associated with this admin' });
    }

    // Get or create chat room for this tenant
    let chatRoom = await ChatRoom.findOne({ tenantId });

    if (!chatRoom) {
      // Get tenant details
      const tenant = await Tenant.findById(tenantId);
      if (!tenant) {
        return res.status(404).json({ message: 'Tenant not found' });
      }

      // Create new chat room
      chatRoom = new ChatRoom({
        tenantId,
        tenantName: tenant.name,
        participants: []
      });

      // Add super admin to participants
      const superAdmin = await Admin.findOne({ role: 'super-admin' });
      if (superAdmin) {
        chatRoom.participants.push({
          userId: superAdmin._id,
          userModel: 'Admin',
          name: superAdmin.name,
          role: 'super_admin'
        });
      }

      // Add current tenant admin to participants
      chatRoom.participants.push({
        userId: admin._id,
        userModel: 'Admin',
        name: admin.name,
        role: 'tenant_admin'
      });

      await chatRoom.save();
    }

    // Return as a single-item array with special formatting for tenant admin view
    const chatRoomForTenant = {
      ...chatRoom.toObject(),
      contactName: 'Super Administrator',
      contactRole: 'super_admin',
      isSupport: true
    };

    res.json([chatRoomForTenant]);
  } catch (error) {
    console.error('Error getting tenant chat rooms:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get chat rooms based on user role (enhanced version)
const getChatRoomsByRole = async (req, res) => {
  try {
    const { admin } = req;

    console.log('getChatRoomsByRole - Admin:', admin.name, 'Role:', admin.role);

    let chatRooms = [];

    if (admin.role === 'super-admin') {
      // Super admin sees all tenant chat rooms
      chatRooms = await ChatRoom.find({ isActive: true })
        .sort({ 'lastMessage.timestamp': -1, updatedAt: -1 });
      
      // Format for super admin view
      chatRooms = chatRooms.map(room => ({
        _id: room._id,
        tenantId: room.tenantId,
        tenantName: room.tenantName,
        contactName: room.tenantName + ' Admin', // Who the super admin is chatting with
        lastMessage: room.lastMessage,
        unreadCount: room.unreadCount?.superAdmin || 0,
        isActive: room.isActive,
        createdAt: room.createdAt,
        updatedAt: room.updatedAt
      }));
    } else {
      // Tenant admin sees their chat with super admin
      const tenantId = admin.tenant?._id;
      if (!tenantId) {
        return res.status(400).json({ message: 'No tenant associated with this admin' });
      }

      let chatRoom = await ChatRoom.findOne({ tenantId });
      
      if (!chatRoom) {
        // Create chat room if it doesn't exist
        const tenant = await Tenant.findById(tenantId);
        if (!tenant) {
          return res.status(404).json({ message: 'Tenant not found' });
        }

        chatRoom = new ChatRoom({
          tenantId,
          tenantName: tenant.name,
          participants: []
        });

        // Add super admin to participants
        const superAdmin = await Admin.findOne({ role: 'super-admin' });
        if (superAdmin) {
          chatRoom.participants.push({
            userId: superAdmin._id,
            userModel: 'Admin',
            name: superAdmin.name,
            role: 'super_admin'
          });
        }

        // Add tenant admin to participants
        chatRoom.participants.push({
          userId: admin._id,
          userModel: 'Admin',
          name: admin.name,
          role: 'tenant_admin'
        });

        await chatRoom.save();
      }

      // Format for tenant admin view
      chatRooms = [{
        _id: chatRoom._id,
        tenantId: chatRoom.tenantId,
        tenantName: 'Support', // Display name for tenant admin
        contactName: 'Super Admin', // Who the tenant admin is chatting with
        lastMessage: chatRoom.lastMessage,
        unreadCount: chatRoom.unreadCount?.tenantAdmin || 0,
        isActive: chatRoom.isActive,
        createdAt: chatRoom.createdAt,
        updatedAt: chatRoom.updatedAt
      }];
    }

    console.log('Returning chat rooms:', chatRooms.length);
    res.json(chatRooms);
  } catch (error) {
    console.error('Error getting chat rooms by role:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

module.exports = {
  getChatRoom,
  getAllChatRooms,
  getTenantChatRooms,
  getChatMessages,
  getFieldWorkerChatMessages,
  sendMessage,
  markMessagesAsRead,
  getChatRoomsByRole
};
