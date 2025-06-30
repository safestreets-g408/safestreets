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
    enum: ['Admin', 'Tenant', 'FieldWorker']
  },
  senderName: {
    type: String,
    required: false
  },
  senderRole: {
    type: String,
    required: false,
    enum: ['super_admin', 'tenant_admin', 'admin', 'field_worker']
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
    userModel: {
      type: String,
      required: true,
      enum: ['Admin', 'Tenant', 'FieldWorker']
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
