const FieldWorker = require('../models/FieldWorker');

const addFieldWorker = async (req, res) => {
    try {
        const { name, workerId, specialization, region, email, password } = req.body;

        const existingWorker = await FieldWorker.findOne({ 
            $or: [{ workerId }, { email }] 
        });
        if (existingWorker) {
            return res.status(400).json({ 
                message: 'Worker ID or email already exists' 
            });
        }

        const fieldWorker = new FieldWorker({
            name,
            workerId,
            specialization,
            region,
            email,
            password
        });

        await fieldWorker.save();

        // Return field worker data (excluding password)
        const fieldWorkerData = {
            _id: fieldWorker._id,
            name: fieldWorker.name,
            workerId: fieldWorker.workerId,
            email: fieldWorker.email,
            specialization: fieldWorker.specialization,
            region: fieldWorker.region,
            activeAssignments: fieldWorker.activeAssignments,
            profile: fieldWorker.profile || {},
            createdAt: fieldWorker.createdAt
        };

        res.status(201).json(fieldWorkerData);
    } catch (error) {
        res.status(500).json({ message: 'Error adding field worker', error: error.message });
    }
};

const getFieldWorkers = async (req, res) => {
    try {
        const fieldWorkers = await FieldWorker.find().select('-password');
        res.status(200).json(fieldWorkers);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching field workers', error: error.message });
    }
};
const getFieldWorkerById = async (req, res) => {
    try {
        const { workerId } = req.params;
        const fieldWorker = await FieldWorker.findOne({ workerId }).select('-password');

        if (!fieldWorker) {
            return res.status(404).json({ message: 'Field worker not found' });
        }

        res.status(200).json(fieldWorker);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching field worker', error: error.message });
    }
};
// Update field worker details
const updateFieldWorker = async (req, res) => {
    try {
        const { workerId } = req.params;
        const updates = req.body;

        const fieldWorker = await FieldWorker.findOneAndUpdate(
            { workerId },
            updates,
            { new: true, runValidators: true }
        );

        if (!fieldWorker) {
            return res.status(404).json({ message: 'Field worker not found' });
        }

        res.status(200).json(fieldWorker);
    } catch (error) {
        res.status(500).json({ message: 'Error updating field worker', error: error.message });
    }
};

module.exports = {
    addFieldWorker,
    getFieldWorkers,
    updateFieldWorker,
    getFieldWorkerById
};