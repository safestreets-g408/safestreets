const express = require('express');
const {
  createAccessRequest,
  getAllAccessRequests,
  getAccessRequestById,
  updateAccessRequestStatus,
  deleteAccessRequest,
  markTenantCreated
} = require('../controllers/accessRequestController');
const { protectAdmin, restrictTo } = require('../middleware/adminAuthMiddleware');

const router = express.Router();

// Public route - anyone can submit an access request
router.post('/', createAccessRequest);

// Protected routes - only super-admin can access
router.get('/', protectAdmin, restrictTo('super-admin'), getAllAccessRequests);
router.get('/:id', protectAdmin, restrictTo('super-admin'), getAccessRequestById);
router.patch('/:id', protectAdmin, restrictTo('super-admin'), updateAccessRequestStatus);
router.delete('/:id', protectAdmin, restrictTo('super-admin'), deleteAccessRequest);
router.patch('/:id/mark-tenant-created', protectAdmin, restrictTo('super-admin'), markTenantCreated);

module.exports = router;
