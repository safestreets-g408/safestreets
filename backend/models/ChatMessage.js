const mongoose = require('mongoose');

const chatMessageSchema = new mongoose.Schema({
  chatId: {
    type: String,
    required: true,
    index: true
  },
  senderId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    refPath: 'senderModel'
  },
  senderModel: {
    type: String,
    required: true,
    enum: ['Admin', 'Tenant']
  },
  senderName: {
    type: String,
    required: true
  },
  senderRole: {
    type: String,
    required: true,
    enum: ['super_admin', 'tenant_admin', 'admin']
  },
  message: {
    type: String,
    required: true,
    trim: true
  },
  messageType: {
    type: String,
    default: 'text',
    enum: ['text', 'image', 'file']
  },
  attachmentUrl: {
    type: String
  },
  isRead: {
    type: Boolean,
    default: false
  },
  readBy: [{
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true
    },
    readAt: {
      type: Date,
      default: Date.now
    }
  }]
}, {
  timestamps: true
});

// Create compound index for efficient querying
chatMessageSchema.index({ chatId: 1, createdAt: -1 });

const ChatMessage = mongoose.model('ChatMessage', chatMessageSchema);

module.exports = ChatMessage;
