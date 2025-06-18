const mongoose = require('mongoose');
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

async function createSuperAdmin(name, email, password) {
    try {
        // Check if a super admin already exists with this email
        const existingAdmin = await Admin.findOne({ email });
        
        if (existingAdmin) {
            // Update existing user to super-admin if found
            existingAdmin.role = 'super-admin';
            await existingAdmin.save();
            console.log(`Updated existing user ${email} to super-admin role`);
            return existingAdmin;
        }

        // Create new super admin
        const superAdmin = new Admin({
            name: name || 'System Administrator',
            email: email || 'superadmin@safestreets.com',
            password: password || 'superadmin123',
            role: 'super-admin'
        });

        await superAdmin.save();
        console.log(`Created new super admin: ${superAdmin.email}`);
        return superAdmin;
    } catch (error) {
        console.error('Error creating super admin:', error);
        throw error;
    }
}

// Get command line arguments
const args = process.argv.slice(2);
const [name, email, password] = args;

// Create super admin and close connection
createSuperAdmin(name, email, password)
    .then(() => {
        console.log('Super admin setup completed');
        mongoose.connection.close();
        process.exit(0);
    })
    .catch(error => {
        console.error('Failed to create super admin:', error);
        mongoose.connection.close();
        process.exit(1);
    });
