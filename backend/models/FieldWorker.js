const mongoose = require('mongoose');

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
    createdAt: { 
        type: Date, 
        default: Date.now 
    }
});

module.exports = mongoose.model('FieldWorker', FieldWorkerSchema);