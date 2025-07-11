const AccessRequest = require('../models/AccessRequest');
const { sendMail } = require('../utils/emailService');

// Create new access request
const createAccessRequest = async (req, res) => {
  try {
    const { organizationName, contactName, email, phone, region, reason } = req.body;
    
    // Validate required fields
    if (!organizationName || !contactName || !email || !phone || !region || !reason) {
      return res.status(400).json({ message: 'All fields are required' });
    }

    // Check if request from this email already exists
    const existingRequest = await AccessRequest.findOne({ email });
    if (existingRequest) {
      return res.status(400).json({ message: 'A request from this email already exists. Please wait for approval or contact support.' });
    }

    const accessRequest = await AccessRequest.create({
      organizationName,
      contactName,
      email,
      phone,
      region,
      reason
    });

    // Notify super admins via email (implementation depends on your email service)
    try {
      await sendMail({
        to: process.env.ADMIN_EMAIL || 'admin@safestreets.com',
        subject: 'New SafeStreets Access Request',
        text: `A new access request has been submitted by ${organizationName}. Please review it in the admin portal.`
      });
    } catch (emailError) {
      console.error('Failed to send email notification:', emailError);
      // Continue processing even if email fails
    }

    res.status(201).json({
      success: true,
      message: 'Access request submitted successfully. Our team will review your request.',
      data: accessRequest
    });

  } catch (error) {
    console.error('Error creating access request:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get all access requests (for super-admin only)
const getAllAccessRequests = async (req, res) => {
  try {
    console.log('Fetching access requests for admin:', req.admin._id, 'Role:', req.admin.role);
    
    const accessRequests = await AccessRequest.find()
      .sort({ createdAt: -1 })
      .populate('reviewedBy', 'name email');
    
    console.log('Access requests found:', accessRequests.length);
    
    res.status(200).json({
      success: true,
      count: accessRequests.length,
      data: accessRequests
    });
  } catch (error) {
    console.error('Error fetching access requests:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Get single access request by ID (for super-admin only)
const getAccessRequestById = async (req, res) => {
  try {
    const accessRequest = await AccessRequest.findById(req.params.id)
      .populate('reviewedBy', 'name email');

    if (!accessRequest) {
      return res.status(404).json({ message: 'Access request not found' });
    }

    res.status(200).json({
      success: true,
      data: accessRequest
    });
  } catch (error) {
    console.error('Error fetching access request:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Update access request status (for super-admin only)
const updateAccessRequestStatus = async (req, res) => {
  try {
    const { status, reviewNotes } = req.body;
    
    if (!['pending', 'approved', 'rejected'].includes(status)) {
      return res.status(400).json({ message: 'Invalid status value' });
    }

    const accessRequest = await AccessRequest.findById(req.params.id);
    if (!accessRequest) {
      return res.status(404).json({ message: 'Access request not found' });
    }

    accessRequest.status = status;
    accessRequest.reviewNotes = reviewNotes || '';
    accessRequest.reviewedBy = req.admin._id;
    accessRequest.reviewedAt = Date.now();

    await accessRequest.save();

    // Send email notification based on status
    try {
      let emailSubject, emailText;

      if (status === 'approved') {
        emailSubject = 'Your SafeStreets Access Request Has Been Approved';
        emailText = `Dear ${accessRequest.contactName},\n\nWe're pleased to inform you that your access request to SafeStreets has been approved. Please check your email for instructions on setting up your account.\n\nBest regards,\nThe SafeStreets Team`;
      } else if (status === 'rejected') {
        emailSubject = 'Update on Your SafeStreets Access Request';
        emailText = `Dear ${accessRequest.contactName},\n\nThank you for your interest in SafeStreets. After careful review, we regret to inform you that we cannot approve your access request at this time. ${accessRequest.reviewNotes ? `\n\nReview notes: ${accessRequest.reviewNotes}` : ''}\n\nIf you have any questions, please contact our support team.\n\nBest regards,\nThe SafeStreets Team`;
      }

      if (emailSubject) {
        await sendMail({
          to: accessRequest.email,
          subject: emailSubject,
          text: emailText
        });
      }
    } catch (emailError) {
      console.error('Failed to send status notification email:', emailError);
      // Continue processing even if email fails
    }

    res.status(200).json({
      success: true,
      message: `Access request ${status}`,
      data: accessRequest
    });
  } catch (error) {
    console.error('Error updating access request:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Delete access request (for super-admin only)
const deleteAccessRequest = async (req, res) => {
  try {
    const accessRequest = await AccessRequest.findById(req.params.id);
    
    if (!accessRequest) {
      return res.status(404).json({ message: 'Access request not found' });
    }

    await AccessRequest.findByIdAndDelete(req.params.id);

    res.status(200).json({
      success: true,
      message: 'Access request deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting access request:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Mark an access request as having a tenant created
const markTenantCreated = async (req, res) => {
  try {
    const accessRequest = await AccessRequest.findById(req.params.id);
    
    if (!accessRequest) {
      return res.status(404).json({ message: 'Access request not found' });
    }
    
    // If the status is not approved, don't allow tenant creation
    if (accessRequest.status !== 'approved') {
      return res.status(400).json({ 
        message: 'Cannot create tenant for a request that is not approved'
      });
    }
    
    accessRequest.tenantCreated = true;
    await accessRequest.save();
    
    res.status(200).json({
      success: true,
      message: 'Access request marked as tenant created',
      data: accessRequest
    });
  } catch (error) {
    console.error('Error updating access request:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

module.exports = {
  createAccessRequest,
  getAllAccessRequests,
  getAccessRequestById,
  updateAccessRequestStatus,
  deleteAccessRequest,
  markTenantCreated
};
