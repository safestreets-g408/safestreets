# SafeStreets - Setup Guide

This document provides detailed instructions for setting up and configuring the SafeStreets system, including all its components.

## Prerequisites

Ensure you have the following installed on your system:

- **Node.js** (v14.x or higher) - [Download here](https://nodejs.org/)
- **MongoDB** (v4.x or higher) - [Installation guide](https://docs.mongodb.com/manual/installation/)
- **Python** (v3.8 or higher) - [Download here](https://python.org/)
- **Redis** (for caching) - [Download here](https://redis.io/download)
- **Expo CLI** - Install with `npm install -g @expo/cli`
- **Git** - [Installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Installation Process

### 1. Clone the Repository
```bash
git clone https://github.com/safestreets-g408/safestreets.git
cd safestreets
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Configure your environment variables in .env:
# MONGODB_URI=mongodb://localhost:27017/safestreets
# JWT_SECRET=your-super-secret-jwt-key
# PORT=5030
# NODE_ENV=development
# REDIS_URL=redis://localhost:6379

# Start MongoDB service (if not running)
# On macOS with Homebrew: brew services start mongodb-community
# On Linux: sudo systemctl start mongod
# On Windows: net start MongoDB

# Start Redis service (if not running)
# On macOS with Homebrew: brew services start redis
# On Linux: sudo systemctl start redis
# On Windows: net start Redis

# Start the backend server
npm start
```

### 3. Admin Portal Setup
```bash
# Open new terminal and navigate to admin portal
cd apps/admin-portal

# Install dependencies
npm install

# Start the development server
npm start

# Access the admin portal at http://localhost:3000
```

### 4. Mobile App Setup
```bash
# Open new terminal and navigate to user app
cd apps/user-app

# Install dependencies
npm install

# Start the Expo development server
npx expo start

# Use Expo Go app to scan QR code or run on simulator
```

### 5. AI Model Server Setup
```bash
# Open new terminal and navigate to AI model server
cd ai_models_server

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download required model files
./scripts/download_models.sh

# Start the model server
python app.py

# Model server will be available at http://localhost:5000
```

### 6. Automated Setup (Alternative)

Alternatively, you can use the provided start script to launch all services:

```bash
# Make the script executable
chmod +x start-services.sh

# Run the script
./start-services.sh
```

This script will:
- Start the backend API server on port 5030
- Start the AI model server on port 5000
- Start the admin portal on port 3000

You'll still need to start the mobile app separately:
```bash
cd apps/user-app && npx expo start
```

## Configuration Details

### Backend Environment Variables
Create a `.env` file in the `backend` directory:

```env
# Database
MONGODB_URI=mongodb://localhost:27017/safestreets

# Authentication
JWT_SECRET=your-super-secret-jwt-key-make-it-long-and-random
JWT_EXPIRES_IN=7d

# Server
PORT=5000
NODE_ENV=development

# AI Model Server
AI_MODEL_URL=http://localhost:5001
```

### Admin Portal Configuration
Update `apps/admin-portal/src/config/constants.js`:

```javascript
export const API_BASE_URL = 'http://localhost:5030/api';
export const AI_MODEL_URL = 'http://localhost:5000';
```

### Mobile App Configuration
Update `apps/user-app/config.js`:

```javascript
export const API_BASE_URL = 'http://your-backend-ip:5030/api';
export const AI_MODEL_URL = 'http://your-ai-server-ip:5000';
```

## Running the System

### Development Mode
```bash
# Terminal 1: Backend server
cd backend && npm start

# Terminal 2: Admin portal
cd apps/admin-portal && npm start

# Terminal 3: Mobile app
cd apps/user-app && npx expo start

# Terminal 4: AI model server
cd vit_model_server && python app.py
```

### Using VS Code Tasks
If you're using VS Code, you can use the pre-configured task:

```bash
# Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)
# Type "Tasks: Run Task"
# Select "Run Backend"
```

## Testing the Installation

1. **Backend API**: Visit `http://localhost:5030/health` - should return health status
2. **Admin Portal**: Visit `http://localhost:3000` - should load the login page
3. **AI Model Server**: Visit `http://localhost:5000/health` - should return AI server status
4. **Mobile App**: Scan QR code with Expo Go app

## Mobile App Installation Options

### For iOS Simulator
```bash
cd apps/user-app
npx expo start --ios
```

### For Android Emulator
```bash
cd apps/user-app
npx expo start --android
```

### For Physical Device
1. Install Expo Go from App Store (iOS) or Google Play Store (Android)
2. Run `npx expo start` in the `apps/user-app` directory
3. Scan the QR code with your device camera (iOS) or Expo Go app (Android)

## Troubleshooting

### Common Issues and Solutions

#### MongoDB Connection Problems
- Ensure MongoDB service is running
- Check that your MongoDB connection string is correct in `.env`
- Verify network connectivity if using a remote MongoDB instance

#### Node Dependencies Issues
- Try deleting `node_modules` and reinstalling with `npm install`
- Ensure you have the correct Node.js version installed

#### Python Environment Issues
- Verify your Python version matches requirements
- Make sure all dependencies are installed with `pip install -r requirements.txt`
- Check for any conflicting packages in your Python environment

#### Expo/Mobile App Issues
- Update Expo CLI to the latest version
- Clear Expo cache with `expo start -c`
- Ensure your device is on the same network as your development machine

For additional issues, please consult the project repository's issue tracker or contact the development team.
