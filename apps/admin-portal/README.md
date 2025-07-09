# SafeStreets Admin Portal

The SafeStreets Admin Portal is a comprehensive web dashboard that provides administrators with tools to manage road damage reports, field workers, and maintenance operations.

## Overview

This admin portal serves as the central management interface for the SafeStreets system, allowing administrators to:

- View and process damage reports
- Manage field worker accounts and assignments
- Generate analytics and insights
- Configure tenant settings (for multi-tenant deployments)
- Monitor system performance and status

## Project Structure

The admin portal follows a modular React architecture:

```
admin-portal/
├── public/              # Static assets
└── src/                 # Source code
    ├── components/      # Reusable UI components
    │   ├── dashboard/   # Dashboard-specific components
    │   ├── layout/      # Layout components
    │   ├── reports/     # Report management components
    │   ├── users/       # User management components
    │   └── ui/          # Generic UI components
    ├── context/         # React context providers
    ├── hooks/           # Custom React hooks
    ├── pages/           # Page components
    │   ├── Dashboard/   # Main dashboard page
    │   ├── Reports/     # Report management pages
    │   ├── FieldWorkers/# Field worker management pages
    │   ├── Analytics/   # Analytics pages
    │   ├── Settings/    # Settings pages
    │   └── Auth/        # Authentication pages
    ├── services/        # API services
    ├── utils/           # Utility functions
    ├── App.js           # Main application component
    └── index.js         # Entry point
```

## Features

### Dashboard

- Interactive overview of system status
- Recent activity feed
- Key performance indicators
- Quick action buttons

### Report Management

- Interactive map view of damage reports
- Filterable list of reports
- Detailed report viewer with AI analysis results
- Report assignment and status management

### Field Worker Management

- Field worker list and profiles
- Assignment tracking
- Performance metrics
- Account management

### Analytics

- Damage trend analysis
- Geographic distribution visualizations
- Field worker performance metrics
- System usage statistics

### Multi-tenant Support

- Tenant management (for super admins)
- Tenant-specific configuration
- Role-based access control

## Development

### Prerequisites

- Node.js (v14.x or higher)
- npm or yarn

### Available Scripts

In the project directory, you can run:

#### `npm start`

Runs the app in development mode at [http://localhost:3000](http://localhost:3000)

#### `npm test`

Launches the test runner in interactive watch mode

#### `npm run build`

Builds the app for production to the `build` folder

### Environment Configuration

Create a `.env` file in the project root with the following variables:

```
REACT_APP_API_URL=http://localhost:5030/api
REACT_APP_MAPS_API_KEY=your_mapbox_api_key
```

## Deployment

### Build for Production

```bash
npm run build
```

This creates optimized production files in the `build` folder that can be deployed to any static hosting service.

### Deployment Options

1. **Static Hosting**: Deploy the built files to services like Netlify, Vercel, or AWS S3
2. **Docker**: Use the provided Dockerfile to build and deploy a containerized version
3. **Server Deployment**: Deploy the built files to a web server like Nginx or Apache

## Integration Points

The admin portal integrates with:

1. **Backend API**: For data retrieval and management
2. **AI Models Server**: For accessing AI analysis capabilities
3. **MapBox/Leaflet**: For geographic visualization
4. **External services**: Weather data, etc.

## Learn More

For more detailed information, refer to the main project documentation:

- [Setup Guide](../../docs/setup-guide.md)
- [Architecture Overview](../../docs/architecture-overview.md)
- [Developer Guide](../../docs/developer-guide.md)
