const FieldWorker = require('./models/FieldWorker');
const mongoose = require('mongoose');
require('dotenv').config();

async function createTestFieldWorker() {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/safe-streets-admin');
    console.log('Connected to MongoDB');

    // Create a test field worker
    const testWorker = new FieldWorker({
      name: 'Ravi Kumar',
      workerId: 'FW001',
      specialization: 'Road Maintenance',
      region: 'North District'
    });

    await testWorker.save();
    console.log('Test field worker created:', {
      name: testWorker.name,
      workerId: testWorker.workerId,
      email: testWorker.email,
      passwordLength: testWorker.password.length
    });

    mongoose.disconnect();
  } catch (error) {
    console.error('Error creating test field worker:', error);
    mongoose.disconnect();
  }
}

createTestFieldWorker();
