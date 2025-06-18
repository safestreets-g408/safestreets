const mongoose = require('mongoose');

const ImageSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true },
    tenant: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Tenant',
        required: true
    },
    image: {
        data: Buffer,
        contentType: String
    },
    result: { type: String, default: 'Pending' },
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Image', ImageSchema);