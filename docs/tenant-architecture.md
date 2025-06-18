# SafeStreets - Multi-Tenant Architecture

## Overview

SafeStreets now implements a multi-tenant architecture that allows:

1. **Super Admins** - Can manage all tenants, create new tenants, and access data across all tenants
2. **Tenant Owners** - Can manage their own tenant, add admin users, and access all data within their tenant
3. **Tenant Admins** - Can access and manage data only within their tenant
4. **Field Workers** - Associated with a specific tenant and can only access/report data for that tenant

## Getting Started

### Default Accounts

When the server first starts, it automatically creates:

- **Super Admin**:
  - Email: `superadmin@safestreets.com`
  - Password: `superadmin123`

To set up a demo tenant with demo accounts, run:

```bash
node backend/utils/setupDemoTenant.js
```

This will create:
- Demo Tenant: "Demo City"
- Tenant Owner:
  - Email: `admin@democity.com`
  - Password: `demo123`
  
After the demo tenant is created, you can add field workers through the admin portal.

## Tenant Management

Only Super Admins can:
- Create new tenants
- See all tenants in the system
- Delete tenants
- Access data across tenant boundaries
- Modify tenant settings and limits

Each tenant has:
- A unique code (used for identification)
- Custom settings:
  - Maximum number of admins
  - Brand colors (primary and secondary)
  - Other tenant-specific configurations
- Isolated data that cannot be accessed by other tenants

Field workers must be added after tenant creation through the tenant management interface. Each field worker requires:
- Name
- Unique Worker ID
- Email
- Password
- Specialization
- Assigned Region
- Phone (optional)

## Data Isolation

The system enforces strict tenant isolation:

1. All data models (DamageReports, FieldWorkers, etc.) include a tenant reference
2. Middleware automatically filters queries by tenant
3. Validation checks prevent cross-tenant operations
4. Authentication tokens include tenant information

## User Hierarchy

1. **Super Admin**: System-wide access, manages tenants
2. **Tenant Owner**: Full access within their tenant
3. **Admin**: Management access within their tenant
4. **Field Worker**: Limited access to tenant-specific assignments

## UI Features

- Different navigation options based on user role
- Tenant management screens (for super admins)
- Tenant-specific dashboards and reports

## Security Notes

- JWT tokens include tenant information for validation
- Role-based authorization is enforced on all API endpoints
- Tenant isolation is checked on every data operation

## Development Usage

When developing new features, always:
1. Include the tenant ID in all data models
2. Apply tenant isolation middleware to routes
3. Check tenant access rights in controllers
4. Use the tenant context in frontend components
