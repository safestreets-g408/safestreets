const mongoose = require('mongoose');
const Tenant = require('../models/Tenant');
const Admin = require('../models/Admin');
require('dotenv').config();

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => {
    console.log('Connected to MongoDB');
}).catch(err => {
    console.error('Error connecting to MongoDB:', err);
    process.exit(1);
});

async function setupDemoTenant() {
    const session = await mongoose.startSession();
    session.startTransaction();

    try {
        // Check if demo tenant already exists
        const existingTenant = await Tenant.findOne({ code: 'democity' }).session(session);
        if (existingTenant) {
            console.log('Demo tenant already exists');
            await session.abortTransaction();
            return;
        }

        // Create demo tenant
        const tenant = await Tenant.create([{
            name: 'Demo City',
            code: 'democity',
            description: 'Demo tenant for testing purposes',
            settings: {
                maxAdmins: 2,
                primaryColor: '#1976d2',
                secondaryColor: '#f50057'
            },
            active: true
        }], { session });

        // Create tenant owner admin
        const admin = await Admin.create([{
            name: 'Demo Admin',
            email: 'admin@democity.com',
            password: 'demo123',
            role: 'tenant-owner',
            tenant: tenant[0]._id
        }], { session });

        await session.commitTransaction();
        console.log('Demo tenant setup completed successfully');
        console.log('Tenant Owner credentials:');
        console.log('Email:', admin[0].email);
        console.log('Password: demo123');

    } catch (error) {
        await session.abortTransaction();
        console.error('Error setting up demo tenant:', error);
        throw error;
    } finally {
        session.endSession();
    }
}

// Run setup and close connection
setupDemoTenant()
    .then(() => {
        console.log('Demo tenant setup process completed');
        mongoose.connection.close();
        process.exit(0);
    })
    .catch(error => {
        console.error('Failed to set up demo tenant:', error);
        mongoose.connection.close();
        process.exit(1);
    });
