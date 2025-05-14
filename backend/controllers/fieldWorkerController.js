const FieldWorker = require('../models/FieldWorker');

const addFieldWorker = async (req, res) => {
    try {
        const { name, workerId, specialization, region } = req.body;

        const existingWorker = await FieldWorker.findOne({ workerId });
        if (existingWorker) {
            return res.status(400).json({ message: 'Worker ID already exists' });
        }

        const fieldWorker = new FieldWorker({
            name,
            workerId,
            specialization,
            region
        });

        await fieldWorker.save();

        res.status(201).json(fieldWorker);
    } catch (error) {
        res.status(500).json({ message: 'Error adding field worker', error: error.message });
    }
};

const getFieldWorkers = async (req, res) => {
    try {
        const fieldWorkers = await FieldWorker.find();
        res.status(200).json(fieldWorkers);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching field workers', error: error.message });
    }
};
const getFieldWorkerById = async (req, res) => {
    try {
        const { workerId } = req.params;
        const fieldWorker = await FieldWorker.findOne({ workerId });

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