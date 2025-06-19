const express = require('express');
const connectDB = require('./utils/db');
const dotenv = require('dotenv');
const cors = require('cors');
const adminRoutes = require('./routes/adminRoutes');
const adminProfileRoutes = require('./routes/adminProfileRoutes');
const imageRoutes = require('./routes/ImageRoutes');
const fieldWorkerRoutes = require('./routes/fieldRoutes');
const fieldWorkerAuthRoutes = require('./routes/fieldWorkerAuthRoutes');
const fieldWorkerDamageRoutes = require('./routes/fieldWorkerDamageRoutes');
const fieldWorkerNotificationRoutes = require('./routes/fieldWorkerNotificationRoutes');
const weatherRoutes = require('./routes/weatherRoutes');
const damageRoutes = require('./routes/damageRoutes');
const tenantRoutes = require('./routes/tenantRoutes');
const aiRoutes = require('./routes/aiRoutes');
const path = require('path');

dotenv.config();
connectDB()

const app = express();
app.use(cors());

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/api/field', fieldWorkerRoutes);
app.use('/api/fieldworker/auth', fieldWorkerAuthRoutes);
app.use('/api/fieldworker/damage', fieldWorkerDamageRoutes);
app.use('/api/fieldworker', fieldWorkerNotificationRoutes);
app.use('/api/fieldworker', weatherRoutes);

app.use('/api/admin/auth', adminRoutes);
app.use('/api/admin/profile', adminProfileRoutes);
app.use('/api/admin/tenants', tenantRoutes);
app.use('/api/admin', require('./routes/tenantAdminRoutes'));
app.use('/api/images', imageRoutes);
app.use('/api/damage', damageRoutes);
app.use('/api/ai', aiRoutes); 

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
