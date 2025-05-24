# Road Damage Detection and Management System


A comprehensive system for detecting, reporting, and managing road damages using AI-powered image recognition. This solution streamlines maintenance operations through intelligent damage classification, automated reporting, and efficient task management.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Machine Learning Model](#machine-learning-model)
- [Security](#security)

## Overview

The Road Damage Detection and Management System is designed to optimize the workflow of road maintenance operations. Field workers can capture images of road damages via a mobile application, which are then automatically analyzed by a Vision Transformer (ViT) model to classify damage types and assess severity. The system generates comprehensive reports accessible through a web dashboard for administrators, enabling efficient monitoring and assignment of repair tasks.

## System Architecture

The system consists of four main components:

1. **Mobile Application**: For maintenance teams to capture and submit damage reports
2. **AI-Powered Damage Detection**: Using Vision Transformer (ViT) for image analysis
3. **Backend Server**: Node.js + Express + MongoDB for data processing and management
4. **Web Dashboard**: React + MUI for administrative monitoring and task management

![System Architecture](https://drive.google.com/file/d/1iJfp8VTpb5mj2EbvW-Z1lgvsEsuYiLUR/view?usp=sharing)

## Features

### Mobile Application

- **Image Capture**: Take photos of road damages with mobile devices
- **GPS Tagging**: Automatic location tagging for each image
- **Damage Submission**: Secure transmission to backend server
- **Status Tracking**: View history and status of submitted reports
- **Offline Support**: Store images when offline, sync when connected

### AI-Powered Damage Detection

- **Image Classification**: Categorizes damages as potholes, surface cracks, alligator cracking, or erosion/wear
- **Severity Assessment**: Evaluates damage based on crack width, area size, and density
- **Text Summary Generation**: Creates concise damage reports with type, severity, priority, and recommended action
- **Performance Optimization**: Image preprocessing and confidence thresholds for improved accuracy

### Backend Server

- **Image API**: Handles reception, storage, and processing of uploaded images
- **ViT API Integration**: Communicates with ML model and processes responses
- **Database Management**: Stores and manages user data, images, classification results, and repair history
- **Email Automation**: Generates and sends detailed reports to administrators
- **Task Assignment API**: Enables admins to assign repair tasks to field teams

### Web Dashboard

- **Admin Login Panel**: Secure administrative access
- **Visual Reports**: Filtered view of reported damages
- **Map View**: Location-based heatmap of damage reports
- **Analytics & Insights**: Trends, severity distribution, and identification of problem areas
- **Repair Management**: Task assignment and status updates
- **Historical Analysis**: Review of past reports and repair patterns

### Notifications & Alerts

- **Email Alerts**: Automatic notifications for high-priority damages
- **Push Notifications**: Real-time alerts for field teams
- **Repair Reminders**: Deadline notifications for pending tasks

## Technologies Used

### Frontend
- React.js
- Material UI (MUI)
- Mapbox/Leaflet for map visualization
- JWT for authentication

### Backend
- Node.js
- Express.js
- MongoDB
- JSON Web Tokens (JWT)

### Mobile Application
- React Native
- Expo
- AsyncStorage for offline data management

### AI/Machine Learning
- Vision Transformer (ViT) model
- TensorFlow/PyTorch
- Image preprocessing libraries

## Installation

### Prerequisites
- Node.js (v14.x or higher)
- MongoDB (v4.x or higher)
- Python (v3.8 or higher) for ML components
- npm or yarn package manager

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/safestreets-g408/safestreets.git
cd safestreets/backend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env
# Edit .env file with your configuration

# Start the server
npm run dev
```

### Frontend Setup
```bash
# Navigate to the frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```

### Mobile App Setup
```bash
# Navigate to the mobile app directory
cd ../mobile-app

# Install dependencies
npm install

# Start the Expo development server
npx expo start
```

### ML Model Setup

> **Note:** The model server is currently under development. Setup instructions will be provided once it is ready.

## Usage

### Mobile Application
1. **Login**: Field workers log in with their credentials
2. **Capture**: Take photos of road damages
3. **Submit**: Submit the image with optional comments
4. **Track**: Monitor the status of submitted reports

### Web Dashboard
1. **Login**: Administrators access the dashboard via secure login
2. **Monitor**: View damage reports with filtering options
3. **Analyze**: Use the map view and analytics to identify patterns
4. **Assign**: Allocate repair tasks to maintenance teams
5. **Follow-up**: Track repair progress and update statuses

## API Documentation

### Authentication
```
POST /api/auth/login
POST /api/auth/register
GET /api/auth/profile
```

### Damage Reports
```
POST /api/damages/report
GET /api/damages
GET /api/damages/:id
PUT /api/damages/:id
DELETE /api/damages/:id
```

### Task Management
```
POST /api/tasks
GET /api/tasks
GET /api/tasks/:id
PUT /api/tasks/:id
DELETE /api/tasks/:id
```

### Analytics
```
GET /api/analytics/overview
GET /api/analytics/trends
GET /api/analytics/hotspots
```


## Machine Learning Model

The Vision Transformer (ViT) model is trained on a diverse dataset of road damage images. It identifies and classifies damages with high accuracy.

### Damage Categories
- Potholes
- Surface cracks
- Alligator cracking
- Erosion/wear

### Severity Levels
- Low: Minimal impact on road usability
- Medium: Noticeable impact, but not urgent
- High: Significant impact, requires prompt attention
- Critical: Immediate safety hazard, urgent repair needed

## Security

- **Authentication**: JWT-based secure authentication
- **Authorization**: Role-based access control
- **Data Protection**: Input validation and sanitization
- **API Security**: Rate limiting and CORS policies
- **Image Storage**: Secure cloud storage with controlled access

## Contact

Project Maintainer - [safestreetsg408@gmail.com](mailto:safestreetsg408@gmail.com)

Project Repository: [https://github.com/safestreets-g408/safestreets.git](https://github.com/safestreets-g408/safestreets.git)
