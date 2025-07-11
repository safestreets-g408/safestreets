/**
 * AI Chat model for Gemini interactions
 */

const mongoose = require('mongoose');

const aiChatSchema = new mongoose.Schema({
  // User who initiated the chat
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    refPath: 'userModel'
  },
  
  // Type of user (Admin or FieldWorker)
  userModel: {
    type: String,
    required: true,
    enum: ['Admin', 'FieldWorker']
  },
  
  // If user belongs to a tenant
  tenantId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Tenant',
    required: false
  },
  
  // Conversation title (auto-generated from first message)
  title: {
    type: String,
    required: false
  },
  
  // Whether the conversation is starred/pinned
  isStarred: {
    type: Boolean,
    default: false
  },
  
  // Messages in this conversation
  messages: [{
    // Content of the message
    content: {
      type: String,
      required: true
    },
    
    // Who sent this message (user or ai)
    role: {
      type: String,
      enum: ['user', 'ai', 'system'],
      required: true
    },
    
    // Timestamp for the message
    timestamp: {
      type: Date,
      default: Date.now
    },
    
    // For images/files shared in chat
    attachments: [{
      type: {
        type: String,
        enum: ['image', 'file'],
        required: true
      },
      url: {
        type: String,
        required: true
      },
      mimeType: String,
      fileName: String
    }],
    
    // Additional metadata for tracking tokens, model used, etc.
    metadata: {
      modelName: String,
      promptTokens: Number,
      completionTokens: Number,
      totalTokens: Number,
      processingTime: Number  // in milliseconds
    }
  }],
  
  // Conversation metadata
  metadata: {
    // AI model used for this conversation
    modelName: {
      type: String,
      default: 'gemini-1.5-flash'
    },
    
    // User's session data for context
    contextData: {
      type: mongoose.Schema.Types.Mixed
    }
  }
}, {
  timestamps: true
});

// Create indexes for efficient querying
aiChatSchema.index({ userId: 1, createdAt: -1 });
aiChatSchema.index({ tenantId: 1, createdAt: -1 });

const AiChat = mongoose.model('AiChat', aiChatSchema);

module.exports = AiChat;
