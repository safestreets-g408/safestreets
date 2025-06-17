# Field Worker Authentication System

## Overview
The field worker authentication system has been implemented to provide secure access to the Safe Streets user application. The system automatically generates email addresses and passwords based on the field worker's name and worker ID.

## Authentication Flow

### 1. Field Worker Model
- **Email Generation**: `firstname.lastname@safestreets.worker`
- **Password Generation**: `first3lettersofname + workerID`
- **Example**: 
  - Name: "John Doe", Worker ID: "FW001"
  - Email: `john.doe@safestreets.worker`
  - Password: `johFW001`

### 2. Backend API Endpoints

#### Field Worker Authentication Routes (`/api/fieldworker/auth/`)
- `POST /register` - Register a new field worker
- `POST /login` - Login field worker
- `GET /profile` - Get field worker profile (protected)
- `PUT /profile` - Update field worker profile (protected)

#### Field Worker Management Routes (`/api/field/`)
- `POST /add` - Add field worker (admin only)
- `GET /workers` - Get all field workers (admin only)
- `GET /:workerId` - Get field worker by ID (admin only)
- `PUT /:workerId` - Update field worker (admin only)

### 3. Frontend Implementation

#### AuthContext
The app uses React Context for state management:
- `isAuthenticated` - Boolean indicating if user is logged in
- `fieldWorker` - Current field worker data
- `login()` - Login function
- `logout()` - Logout function
- `updateFieldWorker()` - Update field worker data

#### Login Screen
- Email and password input fields
- Credential format information displayed
- Automatic token and data storage

#### Profile Screen
- Display field worker information
- Edit profile functionality
- Work statistics
- Logout option

### 4. Security Features
- JWT tokens with 24-hour expiration
- bcrypt password hashing
- Protected routes with middleware
- Token validation on each protected request
- Automatic token storage in AsyncStorage

### 5. Usage Instructions

#### For Administrators:
1. Create field workers using the admin portal
2. Field workers are automatically assigned email and password
3. Share credentials with field workers

#### For Field Workers:
1. Open the user app
2. Enter provided email and password
3. View credential format on login screen for reference
4. Update profile information as needed

### 6. Testing

To create a test field worker:
```bash
cd backend
node create-test-worker.js
```

This will create a test worker with:
- Name: John Doe
- Worker ID: FW001
- Email: john.doe@safestreets.worker
- Password: johFW001

### 7. Configuration

Update the API base URL in `/apps/user-app/utils/auth.js`:
```javascript
const API_BASE_URL = 'http://your-backend-url/api';
```

For Expo development, use your machine's IP address:
```javascript
const API_BASE_URL = 'http://192.168.1.x:5000/api';
```

### 8. Dependencies

#### Backend:
- bcryptjs (password hashing)
- jsonwebtoken (JWT tokens)
- mongoose (MongoDB integration)

#### Frontend:
- @react-native-async-storage/async-storage (token storage)
- React Navigation (navigation)
- React Native Paper (UI components)

## Error Handling
- Invalid credentials show appropriate error messages
- Network errors are handled gracefully
- Token expiration triggers automatic logout
- Validation errors display helpful information
