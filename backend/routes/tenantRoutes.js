const express = require('express');
const { 
  createTenant, 
  getAllTenants, 
  getTenantById, 
  updateTenant, 
  deleteTenant 
} = require('../controllers/tenantController');
const { protectAdmin, restrictTo } = require('../middleware/adminAuthMiddleware');

const router = express.Router();

// Protect all routes
router.use(protectAdmin);

// Restrict to super-admin only
router.use(restrictTo('super-admin'));

// Create tenant
router.post('/', createTenant);

// Get all tenants
router.get('/', getAllTenants);

// Get tenant by ID
router.get('/:id', getTenantById);

// Update tenant
router.put('/:id', updateTenant);

// Delete tenant
router.delete('/:id', deleteTenant);

module.exports = router;
