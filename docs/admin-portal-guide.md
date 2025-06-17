# Admin Portal Guide

This document provides detailed information about the SafeStreets Admin Portal, which allows administrators to monitor road damage reports, manage field workers, and analyze data.

## Overview

The SafeStreets Admin Portal is a web application built with React and Material-UI (MUI) that provides a comprehensive dashboard for managing the road damage detection and repair system. It includes features for monitoring reports, assigning tasks to field workers, viewing analytics, and managing system users.

## Accessing the Admin Portal

### URL
The admin portal is accessible at:
```
http://your-server-address:3000
```

### Authentication
1. Navigate to the login page
2. Enter your admin credentials:
   - Email: provided by system administrator
   - Password: provided by system administrator

### Password Security
- For security reasons, change your password after the first login
- Passwords must contain at least 8 characters, including uppercase letters, lowercase letters, numbers, and special characters
- Your account will be temporarily locked after 5 consecutive failed login attempts

## Dashboard Overview

The admin portal dashboard is divided into several key sections:

### Main Dashboard

The default landing page after login provides an overview of system statistics:

- **Summary Cards**:
  - Total reports
  - Pending reports
  - In-progress repairs
  - Completed repairs
  
- **Recent Activity**:
  - Latest damage reports
  - Recent repair completions
  - System notifications

- **Quick Stats**:
  - Average repair time
  - Reports by damage type
  - System health indicators

### Navigation

The main navigation sidebar provides access to:

1. **Dashboard** - System overview
2. **Reports** - Comprehensive report management
3. **Map View** - Geographical visualization of damages
4. **Field Workers** - User management
5. **Analytics** - Advanced data analysis
6. **Settings** - System configuration
7. **Profile** - Admin account management

## Core Features

### Report Management

The Reports section allows administrators to:

#### View and Filter Reports
- Filter by status (Pending, Analyzed, Assigned, In Progress, Completed)
- Filter by damage type (D00, D10, D20, etc.)
- Filter by date range
- Filter by severity level
- Filter by location
- Search by report ID or description

#### Report Details
Clicking on a report provides detailed information:

- Full-size images
- AI analysis results:
  - Damage classification
  - Confidence score
  - Severity assessment
- Location details with map
- Timeline of actions taken
- Field worker assignments
- Status updates

#### Task Assignment
For analyzed reports:

1. Select a report that requires attention
2. Click "Assign Task"
3. Choose a field worker from the dropdown
4. Set priority level and deadline (optional)
5. Add special instructions (optional)
6. Save assignment

### Map View

The interactive map provides a geographical visualization of damage reports:

- Heat map showing damage concentration
- Color-coded markers for different damage types
- Cluster visualization for dense areas
- Filtering options similar to report management
- Click on markers to view report details
- Select area to view grouped statistics

#### Map Controls
- Zoom and pan controls
- Layer toggles (satellite, street, hybrid)
- Drawing tools for selecting areas
- Export selected area data

### Field Worker Management

The Field Workers section allows administrators to:

#### View Field Workers
- List of all registered field workers
- Status indicators (available, busy, offline)
- Performance metrics
- Current and completed tasks

#### Add New Field Worker
1. Click "Add Field Worker"
2. Enter field worker details:
   - Name
   - Worker ID
   - Contact information
3. Submit to create the account
4. System generates email and password automatically

#### Worker Performance
View detailed performance metrics for each worker:
- Reports processed
- Average completion time
- Quality ratings
- Activity timeline

### Analytics Dashboard

The Analytics section provides data visualization and insights:

#### Data Visualization
- Damage type distribution
- Geographical hotspot analysis
- Repair completion rates
- Severity trends over time
- Field worker performance comparison

#### Custom Reports
1. Select report type
2. Configure parameters
3. Choose visualization style
4. Generate report
5. Export as PDF, CSV, or Excel

#### Trend Analysis
- Monthly/weekly/daily trends
- Seasonal patterns
- Year-over-year comparison
- Predictive analysis (where available)

## Administrative Tools

### System Settings
The Settings section allows configuration of:

- System notification preferences
- Default priorities and deadlines
- AI classification thresholds
- User role permissions
- Integration settings

### User Management
Manage admin users:
- Create new admin accounts
- Modify permissions
- Reset passwords
- View activity logs

### Audit Trail
Complete log of system activities:
- User logins
- Report status changes
- Task assignments
- System configuration changes

## Best Practices

### Report Triage
1. Focus on high-severity damages first
2. Group repairs by geographical proximity
3. Consider weather conditions for scheduling
4. Balance workload among field workers

### Task Assignment
1. Consider field worker specialization
2. Account for geographic proximity
3. Maintain balanced workloads
4. Set appropriate deadlines
5. Monitor progress regularly

### Data Analysis
1. Review analytics weekly
2. Identify recurring damage patterns
3. Monitor repair efficiency
4. Track seasonal trends
5. Generate monthly summary reports

## Keyboard Shortcuts

For power users, the admin portal supports keyboard shortcuts:

| Shortcut | Action |
|----------|--------|
| `Alt + D` | Go to Dashboard |
| `Alt + R` | Go to Reports |
| `Alt + M` | Go to Map View |
| `Alt + F` | Go to Field Workers |
| `Alt + A` | Go to Analytics |
| `Alt + S` | Go to Settings |
| `Ctrl + F` | Open search |
| `Esc` | Close current popup/modal |
| `F5` | Refresh data |

## Troubleshooting

### Common Issues

#### Login Problems
- Verify your email and password
- Check for caps lock
- Clear browser cache and cookies
- Reset password if necessary

#### Data Not Loading
- Check your internet connection
- Verify API server status
- Refresh the page
- Clear browser cache

#### Report Assignment Failures
- Ensure the field worker is active in the system
- Check for maximum task limits
- Verify you have the required permissions
- Ensure the report is in the correct status

### Support Contact

If you encounter persistent issues:
- Email: admin-support@safestreets.com

## Browser Compatibility

The admin portal is optimized for:
- Google Chrome (recommended)
- Mozilla Firefox
- Microsoft Edge
- Safari

For the best experience, use the latest version of Google Chrome.
