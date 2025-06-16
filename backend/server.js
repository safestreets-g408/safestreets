const express = require('express');
const connectDB = require('./utils/db');
const dotenv = require('dotenv');
const cors = require('cors');
const adminRoutes = require('./routes/adminRoutes');
const adminProfileRoutes = require('./routes/adminProfileRoutes');
const imageRoutes = require('./routes/ImageRoutes');
const fieldWorkerRoutes = require('./routes/fieldRoutes');
const damageRoutes = require('./routes/damageRoutes')
const path = require('path');

dotenv.config();
connectDB();

const app = express();
app.use(cors());
// Increase payload size limits for image uploads and large data
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/api/field', fieldWorkerRoutes);

app.use('/api/admin/auth', adminRoutes);
app.use('/api/admin/profile', adminProfileRoutes);
app.use('/api/images', imageRoutes);
app.use('/api/damage', damageRoutes)

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
