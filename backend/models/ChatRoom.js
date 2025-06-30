const mongoose = require('mongoose');

const chatRoomSchema = new mongoose.Schema({
  roomType: {
    type: String,
    enum: ['admin_tenant', 'admin_fieldworker'],
    required: true
  },
  tenantId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Tenant',
    required: function() {
      return this.roomType === 'admin_tenant';
    }
  },
  tenantName: {
    type: String,
    required: function() {
      return this.roomType === 'admin_tenant';
    }
  },
  participants: [{
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      refPath: 'participants.userModel'
    },
    userModel: {
      type: String,
      required: true,
      enum: ['Admin', 'Tenant', 'FieldWorker']
    },
    name: String,
    role: {
      type: String,
      enum: ['super_admin', 'tenant_admin', 'admin', 'field_worker']
    },
    lastSeen: {
      type: Date,
      default: Date.now
    }
  }],
  lastMessage: {
    message: String,
    senderId: mongoose.Schema.Types.ObjectId,
    senderName: String,
    timestamp: Date
  },
  unreadCount: {
    superAdmin: {
      type: Number,
      default: 0
    },
    tenantAdmin: {
      type: Number,
      default: 0
    }
  },
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

chatRoomSchema.index({ tenantId: 1 });

const ChatRoom = mongoose.model('ChatRoom', chatRoomSchema);

module.exports = ChatRoom;
