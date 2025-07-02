const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const connectDB = require('./utils/db');
const dotenv = require('dotenv');
const cors = require('cors');
const { getRedisClient } = require('./utils/redisClient');
const SocketManager = require('./utils/socketManager');
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
const chatRoutes = require('./routes/chatRoutes');
const fieldWorkerChatRoutes = require('./routes/fieldWorkerChatRoutes');
const path = require('path');

dotenv.config();
connectDB();

// Initialize Redis connection
getRedisClient().catch(err => {
  console.error('Failed to connect to Redis:', err);
  console.log('Server continues without Redis caching');
});

const app = express();
const server = http.createServer(app);

// Setup Socket.IO with CORS
const io = socketIo(server, {
  cors: {
    origin: ["http://localhost:3000", "http://localhost:3001"],
    methods: ["GET", "POST"],
    credentials: true
  }
});

// Initialize Socket Manager
const socketManager = new SocketManager(io);

// Simple health check endpoint for API connectivity testing
app.get('/api/health', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Middleware to attach io to requests
app.use((req, res, next) => {
  req.io = io;
  next();
});

app.use(cors());

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/api/field', fieldWorkerRoutes);
app.use('/api/fieldworker/auth', fieldWorkerAuthRoutes);
app.use('/api/fieldworker/damage', fieldWorkerDamageRoutes);
app.use('/api/fieldworker', fieldWorkerNotificationRoutes);
app.use('/api/fieldworker', weatherRoutes);
app.use('/api/fieldworker/chat', fieldWorkerChatRoutes);

app.use('/api/admin/auth', adminRoutes);
app.use('/api/admin/profile', adminProfileRoutes);
app.use('/api/admin/tenants', tenantRoutes);
app.use('/api/admin', require('./routes/tenantAdminRoutes'));
app.use('/api/images', imageRoutes);
app.use('/api/damage', damageRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/chat', chatRoutes);

const PORT = process.env.PORT || 5000;
server.listen(PORT, '0.0.0.0', () => console.log(`🚀 Server running on port ${PORT}`));
