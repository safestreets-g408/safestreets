# SafeStreets - Architecture Overview

This document provides a comprehensive overview of the SafeStreets system architecture, component interactions, and design patterns.

## System Architecture

The SafeStreets platform is built on a modern, microservice-oriented architecture with the following core components:

![System Architecture Diagram](./architecture.png)

### Core Components:

#### 1. User-Facing Applications

- **Mobile Application (React Native + Expo)**
  - Used by field workers to capture and report road damage
  - Provides real-time status updates and notifications
  - Supports offline operation with data synchronization
  - Implements GPS auto-tagging and image capture

- **Admin Portal (React + Material-UI)**
  - Provides comprehensive management dashboard for administrators
  - Visualizes damage reports on interactive maps
  - Offers analytics and reporting capabilities
  - Facilitates task management and assignment

#### 2. Backend Services

- **API Server (Node.js + Express)**
  - Implements RESTful API endpoints for all system operations
  - Handles authentication and authorization
  - Manages data persistence and retrieval
  - Implements multi-tenant architecture
  - Provides caching via Redis for performance optimization

- **AI Models Server (Python + Flask)**
  - Hosts machine learning models for image analysis
  - Processes uploaded images for damage detection
  - Classifies road damage types and severity
  - Generates damage report summaries using Google Gemini

#### 3. Data Storage

- **MongoDB**
  - Document-based NoSQL database
  - Stores user data, damage reports, and system configuration
  - Supports multi-tenant data isolation

- **Cloud Storage**
  - Stores image files and other binary assets
  - Configured for high availability and redundancy

## Data Flow

1. **Image Capture & Submission**
   - Field worker captures road damage image via mobile app
   - Image is tagged with GPS coordinates and metadata
   - Image is uploaded to backend API server
   - API server stores image and metadata in database

2. **AI Processing**
   - Backend forwards image to AI Models Server
   - Road classifier validates image contains road surface
   - Vision Transformer (ViT) analyzes image for damage detection
   - YOLO model identifies specific damage areas
   - Google Gemini generates natural language description
   - Results are returned to backend and stored in database

3. **Notification & Assignment**
   - Admin is notified of new damage reports
   - Reports are prioritized based on AI-determined severity
   - Admin assigns repair tasks to field workers
   - Field workers receive notifications and task details

4. **Reporting & Analytics**
   - System generates aggregated reports on damage patterns
   - Analytics engine processes historical data for insights
   - Administrators view dashboard with visualizations and metrics

## Multi-Tenant Architecture

SafeStreets implements a comprehensive multi-tenant architecture that ensures data isolation between different organizations (tenants):

- **Tenant Hierarchy**
  - Super Admins: Global system administrators
  - Tenant Owners: Organization administrators
  - Tenant Admins: Organization managers
  - Field Workers: Organization employees

- **Data Isolation**
  - All data models include tenant references
  - Middleware enforces tenant-specific data access
  - Authentication tokens contain tenant information
  - APIs validate tenant permissions for all operations

- **Tenant Customization**
  - Each tenant has customizable settings
  - Branding options (colors, logos)
  - Custom field configurations
  - Organization-specific dashboards

## Security Implementation

- **Authentication**
  - JWT-based authentication system
  - Token expiration and refresh mechanism
  - Password hashing with bcrypt
  - Rate limiting to prevent brute force attacks

- **Authorization**
  - Role-based access control (RBAC)
  - Tenant-based permission system
  - Middleware validation of permissions
  - API endpoint authorization checks

- **Data Security**
  - HTTPS/TLS for all communications
  - Input validation and sanitization
  - Protection against common web vulnerabilities
  - Secure file handling and storage

## AI Model Architecture

- **YOLO Object Detection**
  - YOLOv8 with YOLOv5 fallback implementation
  - Identifies and localizes damage areas within images
  - Provides bounding boxes with damage classes
  - Robust fallback mechanisms for reliability

- **Vision Transformer (ViT)**
  - HuggingFace/PyTorch implementation
  - Patch size: 16x16 pixels
  - Input resolution: 224x224 pixels
  - Damage classification into multiple categories

- **CNN Road Classifier**
  - Simple but effective CNN implementation
  - Validates images contain actual road surfaces
  - Prevents processing of irrelevant images
  - Improves system accuracy and efficiency

- **Google Gemini Integration**
  - Leverages Gemini 1.5 Flash API
  - Generates natural language descriptions of damage
  - Provides standardized report formatting
  - Incorporates damage type, severity, and location information

## Scalability Considerations

- **Horizontal Scaling**
  - Stateless API design allows for multiple backend instances
  - Load balancing between instances
  - Distributed processing of AI workloads

- **Performance Optimization**
  - Redis caching for frequently accessed data
  - Optimized database queries and indexing
  - Efficient image processing pipeline

- **Resource Management**
  - On-demand AI processing
  - Efficient storage utilization
  - Optimized mobile app resource usage

This architecture provides a robust foundation for the SafeStreets system while allowing for future expansion and enhancement of capabilities.
