const express = require('express');
const connectDB = require('./utils/db');
const dotenv = require('dotenv');
const cors = require('cors');
const authRoutes = require('./routes/authRoutes');
// const damageRoutes = require('./routes/damageRoutes');
const path = require('path');

dotenv.config();
connectDB();

const app = express();
app.use(cors());
app.use(express.json());
// app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

app.use('/api/auth', authRoutes);
// app.use('/api/damage', damageRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
