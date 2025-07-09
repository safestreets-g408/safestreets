# SafeStreets - Developer Guide

This guide provides developers with information on the codebase organization, development workflows, and best practices for contributing to the SafeStreets project.

## Project Structure

The SafeStreets project is organized into several main components:

```
safestreets/
├── ai_models_server/       # Python Flask server for AI models
├── apps/                   # Frontend applications
│   ├── admin-portal/       # React-based admin dashboard
│   └── user-app/           # React Native mobile app for field workers
├── backend/                # Node.js Express backend API
├── docs/                   # Project documentation
└── notebooks/              # Jupyter notebooks for AI model development
```

### AI Models Server Structure

The AI Models Server follows a clean architecture pattern:

```
ai_models_server/
├── app.py                  # Main application entry point
├── models/                 # Pre-trained AI models
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── api/                # API endpoints and handlers
│   ├── core/               # Core business logic
│   ├── models/             # Model management and inference
│   └── utils/              # Utility functions
└── static/                 # Static files and uploads
```

### Backend Structure

The backend follows a modular, controller-based architecture:

```
backend/
├── controllers/            # Request handlers
├── middleware/             # Express middleware functions
├── models/                 # MongoDB data models
├── routes/                 # API route definitions
├── utils/                  # Utility functions
└── scripts/                # Utility scripts
```

### Admin Portal Structure

The admin portal is a React application with Material-UI:

```
apps/admin-portal/
├── public/                 # Static public assets
└── src/                    # Source code
    ├── components/         # Reusable UI components
    ├── context/            # React context providers
    ├── hooks/              # Custom React hooks
    ├── pages/              # Page components
    ├── services/           # API services
    └── utils/              # Utility functions
```

### Mobile App Structure

The mobile app is a React Native application with Expo:

```
apps/user-app/
├── assets/                 # Static assets (images, fonts)
├── components/             # Reusable UI components
├── context/                # React context providers
├── hooks/                  # Custom React hooks
├── screens/                # Screen components
└── utils/                  # Utility functions
```

## Development Workflow

### Environment Setup

1. Set up the required tools and dependencies as described in the [Setup Guide](./setup-guide.md)
2. Configure your editor (VS Code recommended) with the following extensions:
   - ESLint
   - Prettier
   - Python
   - MongoDB for VS Code
   - React Native Tools

### Running Components

#### Backend Development

```bash
cd backend
npm run dev  # Runs with nodemon for auto-reload
```

#### Admin Portal Development

```bash
cd apps/admin-portal
npm start
```

#### Mobile App Development

```bash
cd apps/user-app
npx expo start
```

#### AI Models Server Development

```bash
cd ai_models_server
source .venv/bin/activate  # Activate virtual environment
python app.py
```

### VS Code Tasks

The project includes VS Code tasks for common operations:

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Type "Tasks: Run Task"
3. Select one of the available tasks:
   - "Start User App" - Start the mobile app
   - Other tasks as configured in your VSCode workspace

## Code Conventions

### JavaScript/TypeScript

- Use ES6+ features
- Follow Airbnb JavaScript Style Guide
- Use async/await for asynchronous operations
- Document functions with JSDoc comments

Example:

```javascript
/**
 * Fetches damage reports for the specified tenant
 * @param {string} tenantId - The tenant ID
 * @param {Object} options - Query options
 * @returns {Promise<Array>} Array of damage reports
 */
async function getDamageReports(tenantId, options = {}) {
  try {
    return await DamageReport.find({ tenant: tenantId, ...options });
  } catch (error) {
    console.error('Error fetching damage reports:', error);
    throw error;
  }
}
```

### Python

- Follow PEP 8 style guide
- Use type hints
- Document functions with docstrings
- Use virtual environments for dependency management

Example:

```python
def classify_image(image_path: str, confidence_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Classifies a road image using the Vision Transformer model.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score to accept classification
        
    Returns:
        Dictionary with classification results
    """
    # Implementation...
```

### API Design

- Follow RESTful API design principles
- Use proper HTTP methods (GET, POST, PUT, DELETE)
- Return consistent JSON response structures
- Include proper error handling and status codes
- Document all endpoints

## Testing

### Backend Testing

```bash
cd backend
npm test
```

### AI Models Testing

```bash
cd ai_models_server
pytest
```

### Frontend Testing

```bash
cd apps/admin-portal
npm test
```

## Deployment

### Production Configuration

For production deployment:

1. Set environment variables to production mode
2. Configure secure database connections
3. Set up HTTPS/TLS
4. Configure proper logging

### Docker Deployment (Optional)

The project can be containerized for easier deployment. Docker configuration files are available in each component directory.

```bash
# Example Docker commands
docker-compose build
docker-compose up -d
```

## Multi-Tenant Development

When developing features for the multi-tenant architecture:

1. Always include tenant context in database operations
2. Use the tenant middleware for API routes
3. Test with multiple tenants to ensure isolation
4. Consider tenant-specific configurations

## AI Model Development

For working on the AI models:

1. Use the Jupyter notebooks in the `notebooks/` directory
2. Follow the existing model structure
3. Export models in the proper format for the AI Models Server
4. Document model performance and parameters
5. Test thoroughly with diverse road images

## Troubleshooting

### Common Issues

1. **MongoDB Connection Issues**
   - Check MongoDB service is running
   - Verify connection string in `.env`
   - Check network connectivity

2. **Mobile App Development Issues**
   - Ensure Expo CLI is up to date
   - Check device/emulator connection
   - Verify API endpoint configuration in `config.js`

3. **AI Model Server Issues**
   - Verify Python version compatibility (3.8+ required)
   - Check if model files are downloaded correctly
   - Ensure CUDA setup for GPU acceleration (if applicable)

4. **Admin Portal Issues**
   - Check for JavaScript console errors
   - Verify API endpoint configuration
   - Clear browser cache if needed

## Contributing Guidelines

1. Follow the established code style
2. Write tests for new features
3. Update documentation for significant changes
4. Use meaningful commit messages
5. Create pull requests with clear descriptions

By following these guidelines, developers can effectively contribute to the SafeStreets project while maintaining code quality and consistency.
