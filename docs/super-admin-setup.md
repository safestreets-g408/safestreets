# Super Admin Setup for SafeStreets

This document explains how to set up and use super admin accounts in the SafeStreets application.

## Default Super Admin

When the server first starts, it automatically creates a default super admin account if one doesn't already exist:

```
Email: superadmin@safestreets.com
Password: superadmin123
```

This account has full system access and can manage all tenants and settings.

## Creating Additional Super Admins

You can create additional super admin accounts using the provided script. The script allows you to specify custom name, email, and password.

### Method 1: Using Default Values

Run:

```bash
cd backend
node scripts/createSuperAdmin.js
```

This creates a super admin with default credentials:
- Name: System Administrator
- Email: superadmin@safestreets.com
- Password: superadmin123

### Method 2: Custom Super Admin

Specify custom values:

```bash
cd backend
node scripts/createSuperAdmin.js "Your Name" "your.email@example.com" "yourpassword"
```

### Method 3: Updating Existing User to Super Admin

If you provide an email that already exists, the script will update that user's role to super-admin.

## Super Admin Capabilities

As a super admin, you can:

1. Access the tenant management page at `/tenants`
2. Create new tenants with the following:
   - Tenant basic information (name, code, description)
   - Tenant owner/admin details
   - Tenant settings (admin limits, branding colors)
3. Manage existing tenants:
   - View and edit tenant details
   - Configure tenant settings
   - Access tenant data and reports
4. View data across all tenants
5. Configure system-wide settings

### Tenant Creation Process

1. Navigate to Tenant Management
2. Click "Create New Tenant"
3. Fill in required tenant information:
   - Name and unique code
   - Description (optional)
   - Tenant admin details
   - System settings
4. Submit to create the tenant
5. After creation, the tenant admin can:
   - Log in to their account
   - Add and manage field workers
   - Configure tenant-specific settings

## Security Recommendations

1. Change the default super admin password immediately after first login
2. Create a dedicated super admin account for each system administrator
3. Use strong, unique passwords for super admin accounts
4. Regularly audit the list of super admin users

## Troubleshooting

If you're having issues with super admin access:

1. Check that the MongoDB connection is working correctly
2. Verify that the super admin account exists in the database
3. Ensure the JWT secret is properly configured in the environment variables
4. Check the logs for any authentication errors
