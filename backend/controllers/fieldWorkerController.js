const FieldWorker = require('../models/FieldWorker');

const addFieldWorker = async (req, res) => {
    try {
        const { name, workerId, specialization, region, email, password } = req.body;

        // Check for duplicate email/workerId within the same tenant
        const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        const existingWorker = await FieldWorker.findOne({ 
            $or: [{ workerId }, { email }],
            ...tenantFilter // Only check within the same tenant
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
            password,
            // Add tenant reference from middleware
            tenant: req.tenantId
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
            tenant: fieldWorker.tenant,
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
        // Apply tenant filter - only get field workers for the current tenant
        // req.tenantId is set by the ensureTenantIsolation middleware
        let tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            // Super admin can see all field workers if no specific tenant is requested
            tenantFilter = {};
        }
        
        const fieldWorkers = await FieldWorker.find(tenantFilter).select('-password');
        res.status(200).json(fieldWorkers);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching field workers', error: error.message });
    }
};
const getFieldWorkerById = async (req, res) => {
    try {
        const { workerId } = req.params;
        
        // Apply tenant filter - only get field workers for the current tenant
        // req.tenantId is set by the ensureTenantIsolation middleware
        const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            // Super admin can see any field worker
            tenantFilter = {};
        }
        
        const fieldWorker = await FieldWorker.findOne({ 
            workerId,
            ...tenantFilter
        }).select('-password');

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
        
        // Don't allow changing the tenant
        if (updates.tenant) {
            delete updates.tenant;
        }
        
        // Apply tenant filter - only update field workers in the current tenant
        // req.tenantId is set by the ensureTenantIsolation middleware
        let tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            // Super admin can update any field worker
            tenantFilter = {};
        }

        const fieldWorker = await FieldWorker.findOneAndUpdate(
            { 
                workerId,
                ...tenantFilter 
            },
            updates,
            { new: true, runValidators: true }
        ).select('-password');

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