const mongoose = require('mongoose');

const aiReportSchema = new mongoose.Schema({
    imageId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Image',
        required: true
    },
    tenant: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Tenant',
        required: true
    },
    predictionClass: {
        type: String,
        required: true
    },
    damageType: {
        type: String,
        required: true
    },
    severity: {
        type: String,
        enum: ['LOW', 'MEDIUM', 'HIGH'],
        required: true
    },
    priority: {
        type: Number,
        min: 1,
        max: 10,
        required: true
    },
    location: {
        coordinates: {
            type: [Number], // [longitude, latitude]
            index: '2dsphere'
        },
        address: String
    },
    annotatedImageBase64: {
        type: String,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('AiReport', aiReportSchema);
