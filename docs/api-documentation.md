# API Documentation

This document provides comprehensive information about the SafeStreets API, including endpoints, authentication, request/response formats, and examples.

## Base URL

The base URL for all API endpoints is:

```
http://your-server-address:5030/api
```

## Authentication

SafeStreets uses JWT (JSON Web Token) for authentication. Most endpoints require valid authentication.

### Token Format

```
Authorization: Bearer <jwt_token>
```

### Token Expiration

Tokens expire after 7 days by default. This can be configured in the backend environment variables.

## API Endpoints

### Authentication Endpoints

#### Admin Authentication

**Login**
```http
POST /admin/auth/login
Content-Type: application/json

{
  "email": "admin@safestreets.com",
  "password": "securepassword"
}

Response:
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "admin_id",
    "email": "admin@safestreets.com",
    "name": "Admin Name"
  }
}
```

**Register (Admin Only)**
```http
POST /admin/auth/register
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "email": "newadmin@safestreets.com",
  "password": "securepassword",
  "name": "New Admin"
}

Response:
{
  "success": true,
  "message": "Admin user created successfully"
}
```

#### Admin Profile Management

**Get Profile**
```http
GET /admin/profile
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "id": "admin_id",
    "name": "Admin Name",
    "email": "admin@safestreets.com"
  }
}
```

**Update Profile**
```http
PUT /admin/profile
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "name": "Updated Admin Name"
}

Response:
{
  "success": true,
  "message": "Profile updated successfully"
}
```

#### Field Worker Authentication

**Register**
```http
POST /fieldworker/auth/register
Content-Type: application/json

{
  "email": "worker@safestreets.worker",
  "password": "workerpassword",
  "name": "Worker Name"
}

Response:
{
  "success": true,
  "message": "Field worker registered successfully"
}
```

**Login**
```http
POST /fieldworker/auth/login
Content-Type: application/json

{
  "email": "worker@safestreets.worker",
  "password": "workerpassword"
}

Response:
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "worker_id",
    "email": "worker@safestreets.worker",
    "name": "Worker Name"
  }
}
```

**Get Profile**
```http
GET /fieldworker/auth/profile
Authorization: Bearer <worker_token>

Response:
{
  "success": true,
  "data": {
    "id": "worker_id",
    "name": "Worker Name",
    "email": "worker@safestreets.worker",
    "workerId": "FW001",
    "assignedTasks": [...],
    "completedTasks": [...],
    "joinDate": "2024-01-01T00:00:00Z"
  }
}
```

**Update Profile**
```http
PUT /fieldworker/auth/profile
Content-Type: application/json
Authorization: Bearer <worker_token>

{
  "name": "Updated Name",
  "phone": "123-456-7890"
}

Response:
{
  "success": true,
  "message": "Profile updated successfully"
}
```

### Damage Report Endpoints

#### Admin Damage Management

**Submit New Damage Report**
```http
POST /damage/upload
Content-Type: multipart/form-data
Authorization: Bearer <admin_token>

FormData:
- image: [file]
- location: {"latitude": 40.7128, "longitude": -74.0060}
- description: "Pothole on main street"
- severity: "high"

Response:
{
  "success": true,
  "data": {
    "id": "report_id",
    "status": "pending",
    "createdAt": "2024-06-17T10:30:00Z"
  }
}
```

**Get Damage History**
```http
GET /damage/history
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "history": [
      {
        "id": "history_id",
        "reportId": "report_id",
        "action": "created",
        "timestamp": "2024-06-17T10:30:00Z",
        "user": "Admin Name"
      },
      // ...more history items
    ]
  }
}
```

**Get All Damage Reports**
```http
GET /damage/reports?page=1&limit=10&status=analyzed
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "reports": [
      {
        "id": "report_id_1",
        "location": {"latitude": 40.7128, "longitude": -74.0060},
        "status": "analyzed",
        "createdAt": "2024-06-17T10:30:00Z",
        "aiAnalysis": {...}
      },
      // ...more reports
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 50,
      "pages": 5
    }
  }
}
```

**Get Specific Damage Report**
```http
GET /damage/report/:reportId
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "id": "report_id",
    "aiAnalysis": {
      "classification": "D30",
      "confidence": 0.95,
      "severity": "high"
    },
    "location": {"latitude": 40.7128, "longitude": -74.0060},
    "description": "Pothole on main street",
    "status": "analyzed",
    "createdAt": "2024-06-17T10:30:00Z",
    "images": [
      {
        "id": "image_id",
        "url": "http://your-server/uploads/image.jpg",
        "contentType": "image/jpeg"
      }
    ]
  }
}
```

**Get Report Image**
```http
GET /damage/report/:reportId/image/:type
Authorization: Bearer <jwt_token> (optional, can be in query param)

Response: Image file (binary)
```

**Create Report from AI Analysis**
```http
POST /damage/reports/create-from-ai
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "aiReportId": "ai_report_id",
  "description": "Based on AI detection",
  "priority": "high"
}

Response:
{
  "success": true,
  "data": {
    "reportId": "report_id",
    "status": "created"
  }
}
```

**Create and Assign Report from AI Analysis**
```http
POST /damage/reports/create-and-assign
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "aiReportId": "ai_report_id",
  "description": "Based on AI detection",
  "priority": "high",
  "workerId": "worker_id"
}

Response:
{
  "success": true,
  "data": {
    "reportId": "report_id",
    "status": "assigned",
    "assignedTo": "worker_id"
  }
}
```

**Get AI Generated Reports**
```http
GET /damage/reports/generated-from-ai
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "reports": [
      {
        "id": "report_id",
        "aiReportId": "ai_report_id",
        "status": "assigned",
        "createdAt": "2024-06-17T10:30:00Z"
      },
      // ...more reports
    ]
  }
}
```

**Assign Repair Task**
```http
PATCH /damage/reports/:reportId/assign
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "workerId": "worker_id",
  "priority": "high",
  "notes": "Please fix ASAP"
}

Response:
{
  "success": true,
  "message": "Repair task assigned successfully"
}
```

**Unassign Repair Task**
```http
PATCH /damage/reports/:reportId/unassign
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "message": "Repair task unassigned successfully"
}
```

**Update Report Status**
```http
PATCH /damage/reports/:reportId/status
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "status": "completed",
  "notes": "Repaired on June 17, 2025"
}

Response:
{
  "success": true,
  "message": "Report status updated successfully"
}
```

**Update Report**
```http
PUT /damage/report/:reportId
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "description": "Updated description",
  "severity": "medium",
  "priority": "medium"
}

Response:
{
  "success": true,
  "message": "Report updated successfully"
}
```

**Delete Report**
```http
DELETE /damage/report/:reportId
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "message": "Report deleted successfully"
}
```

#### Field Worker Damage Management

**Get Field Worker Dashboard**
```http
GET /fieldworker/damage/dashboard
Authorization: Bearer <worker_token>

Response:
{
  "success": true,
  "data": {
    "assignedTasks": 5,
    "completedTasks": 15,
    "pendingTasks": [
      {
        "id": "task_id",
        "description": "Fix pothole on main street",
        "priority": "high",
        "location": {"latitude": 40.7128, "longitude": -74.0060}
      },
      // ...more tasks
    ]
  }
}
```

**Get Field Worker Reports**
```http
GET /fieldworker/damage/reports
Authorization: Bearer <worker_token>

Response:
{
  "success": true,
  "data": {
    "reports": [
      {
        "id": "report_id",
        "description": "Pothole on main street",
        "status": "assigned",
        "priority": "high",
        "location": {"latitude": 40.7128, "longitude": -74.0060},
        "assignedAt": "2024-06-17T10:30:00Z"
      },
      // ...more reports
    ]
  }
}
```

**Update Repair Status (Field Worker)**
```http
PATCH /fieldworker/damage/reports/:reportId/status
Content-Type: application/json
Authorization: Bearer <worker_token>

{
  "status": "completed",
  "notes": "Fixed the pothole and resurfaced the area",
  "completionImages": ["base64_image_1", "base64_image_2"]
}

Response:
{
  "success": true,
  "message": "Repair status updated successfully"
}
```

**Upload Damage Report (Field Worker)**
```http
POST /fieldworker/damage/reports/upload
Content-Type: multipart/form-data
Authorization: Bearer <worker_token>

FormData:
- image: [file]
- location: {"latitude": 40.7128, "longitude": -74.0060}
- description: "New pothole discovered during repair work"
- severity: "medium"

Response:
{
  "success": true,
  "data": {
    "id": "report_id",
    "status": "pending",
    "createdAt": "2024-06-17T10:30:00Z"
  }
}
```

### Image Processing Endpoints

**Test AI Server Connection**
```http
GET /images/test-ai-server
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "status": "connected",
    "message": "AI server is operational"
  }
}
```

**Upload Image for Analysis**
```http
POST /images/upload
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "image": "base64_encoded_image",
  "metadata": {"source": "mobile_app", "quality": "high"}
}

Response:
{
  "success": true,
  "data": {
    "imageId": "image_id",
    "url": "http://your-server/uploads/image.jpg",
    "aiAnalysis": {
      "classification": "D30",
      "confidence": 0.95
    }
  }
}
```

**Get Image by Email**
```http
GET /images/email/:email
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "images": [
      {
        "id": "image_id",
        "url": "http://your-server/uploads/image.jpg",
        "metadata": {"source": "mobile_app", "quality": "high"},
        "uploadedAt": "2024-06-17T10:30:00Z"
      },
      // ...more images
    ]
  }
}
```

**Get Image by ID**
```http
GET /images/id/:imageId
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "id": "image_id",
    "url": "http://your-server/uploads/image.jpg",
    "metadata": {"source": "mobile_app", "quality": "high"},
    "uploadedAt": "2024-06-17T10:30:00Z"
  }
}
```

**Get Image Reports**
```http
GET /images/reports
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "reports": [
      {
        "id": "report_id",
        "imageId": "image_id",
        "classification": "D30",
        "confidence": 0.95,
        "createdAt": "2024-06-17T10:30:00Z"
      },
      // ...more reports
    ]
  }
}
```

**Get Image Report by ID**
```http
GET /images/reports/:reportId
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "id": "report_id",
    "imageId": "image_id",
    "classification": "D30",
    "confidence": 0.95,
    "createdAt": "2024-06-17T10:30:00Z",
    "image": {
      "id": "image_id",
      "url": "http://your-server/uploads/image.jpg"
    }
  }
}
```

### Field Worker Management Endpoints

**Add Field Worker (Admin Only)**
```http
POST /field/add
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "name": "John Doe",
  "workerId": "FW002",
  "email": "john.doe@safestreets.worker",
  "phone": "555-123-4567",
  "personalEmail": "john.personal@email.com",
  "specialization": "Road Repair",
  "region": "North District",
  "status": "Available"
}

Response:
{
  "success": true,
  "data": {
    "_id": "worker_db_id",
    "name": "John Doe",
    "email": "john.doe@safestreets.worker",
    "workerId": "FW002",
    "specialization": "Road Repair",
    "region": "North District",
    "profile": {
      "phone": "555-123-4567",
      "personalEmail": "john.personal@email.com",
      "receiveDailyUpdates": true,
      "isActive": true,
      "totalReportsHandled": 0
    },
    "status": "Available",
    "activeAssignments": 0
  }
}
```

**Get All Field Workers (Admin Only)**
```http
GET /field/workers
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "workers": [
      {
        "id": "worker_id_1",
        "name": "John Doe",
        "email": "john.doe@safestreets.worker",
        "workerId": "FW001",
        "assignedTasks": 5,
        "completedTasks": 20
      },
      // ...more workers
    ]
  }
}
```

**Get Field Worker by ID (Admin Only)**
```http
GET /field/:workerId
Authorization: Bearer <admin_token>

Response:
{
  "name": "John Doe",
  "email": "john.doe@safestreets.worker",
  "workerId": "FW001",
  "specialization": "Road Repair",
  "region": "North District",
  "profile": {
    "phone": "555-123-4567",
    "personalEmail": "john.personal@email.com",
    "receiveDailyUpdates": true,
    "isActive": true,
    "totalReportsHandled": 15,
    "lastActive": "2025-07-11T09:30:45.123Z"
  },
  "activeAssignments": 2,
  "status": "Available"
}
```

**Get Field Worker Assignments (Admin Only)**
```http
GET /field/:workerId/assignments
Authorization: Bearer <admin_token>

Response:
{
  "fieldWorker": {
    "id": "worker_db_id",
    "name": "John Doe",
    "workerId": "FW001",
    "email": "john.doe@safestreets.worker"
  },
  "assignments": [
    {
      "_id": "assignment_id_1",
      "reportId": "REP-001",
      "location": {
        "address": "123 Main Street",
        "coordinates": [35.12345, -82.54321]
      },
      "damageType": "Pothole",
      "severity": "High",
      "status": "In Progress",
      "createdAt": "2025-07-01T10:30:00.000Z",
      "assignedAt": "2025-07-01T14:15:00.000Z"
    },
    {
      "_id": "assignment_id_2",
      "reportId": "REP-002",
      "location": {
        "address": "456 Elm Street",
        "coordinates": [35.12675, -82.55421]
      },
      "damageType": "Surface Crack",
      "severity": "Medium",
      "status": "Under Review",
      "createdAt": "2025-07-03T09:20:00.000Z",
      "assignedAt": "2025-07-03T11:45:00.000Z"
    }
  ],
  "count": 2
}
```

**Update Field Worker (Admin Only)**
```http
PUT /field/:workerId
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "name": "John Smith",
  "specialization": "Pothole Repair",
  "region": "South District",
  "phone": "555-987-6543",
  "personalEmail": "john.smith.personal@email.com",
  "receiveDailyUpdates": true,
  "status": "Available"
}

Response:
{
  "_id": "worker_db_id",
  "name": "John Smith",
  "email": "john.doe@safestreets.worker",
  "workerId": "FW002",
  "specialization": "Pothole Repair",
  "region": "South District",
  "profile": {
    "phone": "555-987-6543",
    "personalEmail": "john.smith.personal@email.com",
    "receiveDailyUpdates": true,
    "isActive": true,
    "totalReportsHandled": 0,
    "lastActive": "2025-07-10T15:30:45.123Z"
  },
  "status": "Available",
  "activeAssignments": 0
}
```

### Analytics Endpoints

#### System Overview
```http
GET /analytics/overview
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "totalReports": 1250,
    "pendingReports": 45,
    "completedRepairs": 1100,
    "averageRepairTime": "3.2 days",
    "topDamageTypes": [
      {"type": "D30", "count": 523},
      {"type": "D20", "count": 312},
      {"type": "D10", "count": 215}
    ]
  }
}
```

#### Geographic Hotspots
```http
GET /analytics/hotspots
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "hotspots": [
      {
        "location": {"lat": 40.7128, "lng": -74.0060},
        "damageCount": 15,
        "severity": "high",
        "area": "Downtown District"
      },
      // ...more hotspots
    ]
  }
}
```

#### Time Series Data
```http
GET /analytics/timeseries?period=month
Authorization: Bearer <admin_token>

Response:
{
  "success": true,
  "data": {
    "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "datasets": [
      {
        "label": "Reports",
        "data": [65, 78, 90, 81, 56, 55]
      },
      {
        "label": "Completed",
        "data": [40, 65, 75, 70, 50, 40]
      }
    ]
  }
}
```

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  }
}
```

### Common Error Codes
- `AUTHENTICATION_ERROR` - Invalid or expired token
- `AUTHORIZATION_ERROR` - Insufficient permissions
- `VALIDATION_ERROR` - Invalid input data
- `RESOURCE_NOT_FOUND` - Requested resource not found
- `SERVER_ERROR` - Internal server error

### Response Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error

## Data Formats

### Date Time Format
All timestamps are in ISO 8601 format (UTC):
```
YYYY-MM-DDThh:mm:ssZ
```

### Location Format
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060
}
```

### Damage Classification Types
- `D00` - Longitudinal cracks
- `D10` - Transverse cracks
- `D20` - Alligator cracks
- `D30` - Potholes
- `D40` - Line cracks
- `D43` - Cross walk blur
- `D44` - Whiteline blur
- `D50` - Manhole covers

### Severity Levels
- `low` - Low severity
- `medium` - Medium severity
- `high` - High severity
- `critical` - Critical severity

