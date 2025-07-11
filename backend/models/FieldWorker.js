const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const FieldWorkerSchema = new mongoose.Schema({
    name: { 
        type: String, 
        required: true 
    },
    workerId: { 
        type: String, 
        required: true, 
        unique: true 
    },
    email: {
        type: String,
        unique: true,
        lowercase: true
    },
    tenant: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Tenant',
        required: true
    },
    password: {
        type: String,
        minlength: 6
    },
    specialization: { 
        type: String, 
        required: true,
    },
    region: { 
        type: String, 
        required: true 
    },
    activeAssignments: { 
        type: Number, 
        default: 0,
        min: 0,
        max: 10 
    },
    profile: {
        phone: String,
        personalEmail: String,
        receiveDailyUpdates: { type: Boolean, default: false },
        lastActive: Date,
        totalReportsHandled: { type: Number, default: 0 },
        isActive: { type: Boolean, default: true }
    },
    createdAt: { 
        type: Date, 
        default: Date.now 
    },
    deviceTokens: [
        {
            token: {
                type: String,
                required: true
            },
            type: {
                type: String,
                enum: ['ios', 'android', 'web', 'unknown'],
                default: 'unknown'
            },
            lastRegistered: {
                type: Date,
                default: Date.now
            }
        }
    ]
});

// Generate email and password from name and worker ID
FieldWorkerSchema.pre('save', async function(next) {
    if (this.isNew) {
        if (!this.email && this.name && this.workerId) {
            // Generate email from name (lowercase, replace spaces with dots)
            const emailPrefix = this.name.toLowerCase().replace(/\s+/g, '.');
            this.email = `${emailPrefix}@safestreets.worker`;
        }
        
        if (!this.password && this.name && this.workerId) {
            // Generate password from name + workerId (first 3 chars of name + workerId)
            const namePrefix = this.name.toLowerCase().replace(/\s+/g, '').substring(0, 3);
            this.password = `${namePrefix}${this.workerId}`;
        }
    }
    
    next();
});

// Hash password before saving
FieldWorkerSchema.pre('save', async function(next) {
    if (!this.isModified('password')) return next();
    
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
});

// Compare password method
FieldWorkerSchema.methods.matchPassword = async function(enteredPassword) {
    return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('FieldWorker', FieldWorkerSchema);