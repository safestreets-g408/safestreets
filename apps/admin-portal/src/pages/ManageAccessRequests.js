import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Button,
  TextField,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Alert,
  Snackbar,
  Tooltip,
  Divider
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Check as ApproveIcon,
  Close as RejectIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Business as BusinessIcon,
  Email as EmailIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { API_BASE_URL, TOKEN_KEY } from '../config/constants';
import axios from 'axios';
import { useAuth } from '../hooks/useAuth';
import { format } from 'date-fns';

const ManageAccessRequests = () => {
  const { isAuthenticated } = useAuth();
  const token = localStorage.getItem(TOKEN_KEY);
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showAlert, setShowAlert] = useState(false);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [statusDialogOpen, setStatusDialogOpen] = useState(false);
  const [createTenantDialogOpen, setCreateTenantDialogOpen] = useState(false);
  const [dialogAction, setDialogAction] = useState('');
  const [reviewNotes, setReviewNotes] = useState('');
  const [actionLoading, setActionLoading] = useState(false);
  const [tenantFormData, setTenantFormData] = useState({
    name: '',
    code: '',
    description: '',
    adminName: '',
    adminEmail: '',
    adminPassword: '',
    settings: {
      maxFieldWorkers: 10,
      maxAdmins: 2,
      primaryColor: '#1976d2',
      secondaryColor: '#f50057'
    }
  });
  const [stats, setStats] = useState({
    total: 0,
    pending: 0,
    approved: 0,
    rejected: 0
  });

  useEffect(() => {
    if (isAuthenticated && token) {
      fetchAccessRequests();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, token]);

  const fetchAccessRequests = async () => {
    try {
      setLoading(true);
      console.log('Fetching access requests from:', `${API_BASE_URL}/access-requests`);
      console.log('Using token:', token ? `Yes (${token.substring(0, 10)}...)` : 'No');
      
      if (!token) {
        throw new Error('No authentication token found. Please login again.');
      }
      
      const response = await axios.get(`${API_BASE_URL}/access-requests`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      console.log('Response data:', response.data);
      const requestData = response.data.data || [];
      setRequests(requestData);
      
      // Calculate statistics
      const total = requestData.length;
      const pending = requestData.filter(req => req.status === 'pending').length;
      const approved = requestData.filter(req => req.status === 'approved').length;
      const rejected = requestData.filter(req => req.status === 'rejected').length;
      
      setStats({
        total,
        pending,
        approved,
        rejected
      });

      setError('');
    } catch (err) {
      console.error('Error fetching access requests:', err);
      if (err.response?.status === 401) {
        setError('Authentication failed. Please log out and log in again.');
      } else {
        setError(err.response?.data?.message || err.message || 'Failed to fetch access requests');
      }
      setShowAlert(true);
    } finally {
      setLoading(false);
    }
  };

  const handleViewRequest = (request) => {
    setSelectedRequest(request);
    setViewDialogOpen(true);
  };

  const handleStatusAction = (request, action) => {
    setSelectedRequest(request);
    setDialogAction(action);
    setReviewNotes('');
    setStatusDialogOpen(true);
  };

  const handleStatusUpdate = async () => {
    if (!selectedRequest) return;
    
    try {
      setActionLoading(true);
      
      await axios.patch(
        `${API_BASE_URL}/access-requests/${selectedRequest._id}`,
        {
          status: dialogAction,
          reviewNotes
        },
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      setSuccess(`Request ${dialogAction === 'approved' ? 'approved' : 'rejected'} successfully`);
      setShowAlert(true);
      setStatusDialogOpen(false);
      
      // If approved, open the tenant creation dialog
      if (dialogAction === 'approved') {
        // Pre-fill the tenant form with data from the access request
        setError(''); // Clear any previous errors
        const uniqueCode = selectedRequest.organizationName.toLowerCase()
          .replace(/\s+/g, '-')
          .replace(/[^a-z0-9-]/g, '')
          .substring(0, 8) + '-' + Math.random().toString(36).substring(2, 6);
          
        setTenantFormData({
          name: selectedRequest.organizationName,
          code: uniqueCode,
          description: `Created from access request by ${selectedRequest.contactName}`,
          adminName: selectedRequest.contactName,
          adminEmail: selectedRequest.email,
          adminPassword: generatePassword(12),
          settings: {
            maxFieldWorkers: 10,
            maxAdmins: 2,
            primaryColor: '#1976d2',
            secondaryColor: '#f50057'
          }
        });
        
        setCreateTenantDialogOpen(true);
      }
      
      fetchAccessRequests();
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to update request status');
      setShowAlert(true);
    } finally {
      setActionLoading(false);
    }
  };
  
  // Helper to generate a strong password
  const generatePassword = (length = 12) => {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*";
    let password = "";
    for (let i = 0; i < length; i++) {
      const randomIndex = Math.floor(Math.random() * charset.length);
      password += charset[randomIndex];
    }
    return password;
  };
  
  const generateUniqueCode = () => {
    const timestamp = new Date().getTime().toString(36).substring(0, 4);
    const randomChars = Math.random().toString(36).substring(2, 6);
    return `${tenantFormData.name?.toLowerCase().replace(/\s+/g, '-').substring(0, 8) || 'tenant'}-${timestamp}${randomChars}`;
  };

  const handleCompleteTenantCreation = (request) => {
    setSelectedRequest(request);
    
    // Pre-fill the tenant form with data from the access request
    setError(''); // Clear any previous errors
    const uniqueCode = request.organizationName.toLowerCase()
      .replace(/\s+/g, '-')
      .replace(/[^a-z0-9-]/g, '')
      .substring(0, 8) + '-' + Math.random().toString(36).substring(2, 6);
      
    setTenantFormData({
      name: request.organizationName,
      code: uniqueCode,
      description: `Created from access request by ${request.contactName}`,
      adminName: request.contactName,
      adminEmail: request.email,
      adminPassword: generatePassword(12),
      settings: {
        maxFieldWorkers: 10,
        maxAdmins: 2,
        primaryColor: '#1976d2',
        secondaryColor: '#f50057'
      }
    });
    
    // Open the tenant creation dialog
    setCreateTenantDialogOpen(true);
  };

  const handleDeleteRequest = async (requestId) => {
    if (!window.confirm('Are you sure you want to delete this request? This action cannot be undone.')) {
      return;
    }
    
    try {
      setActionLoading(true);
      
      await axios.delete(
        `${API_BASE_URL}/access-requests/${requestId}`,
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      setSuccess('Request deleted successfully');
      setShowAlert(true);
      fetchAccessRequests();
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to delete request');
      setShowAlert(true);
    } finally {
      setActionLoading(false);
    }
  };

  const handleCloseAlert = () => {
    setShowAlert(false);
  };
  
  const handleTenantFormChange = (e) => {
    const { name, value } = e.target;
    
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setTenantFormData({
        ...tenantFormData,
        [parent]: {
          ...tenantFormData[parent],
          [child]: value
        }
      });
    } else {
      setTenantFormData({
        ...tenantFormData,
        [name]: value
      });
    }
  };
  
  const handleCreateTenant = async () => {
    // Enhanced validation
    const validationErrors = [];
    if (!tenantFormData.name) validationErrors.push('Tenant Name is required');
    if (!tenantFormData.code) validationErrors.push('Tenant Code is required');
    if (!tenantFormData.adminName) validationErrors.push('Admin Name is required');
    if (!tenantFormData.adminEmail) validationErrors.push('Admin Email is required');
    if (!tenantFormData.adminPassword) validationErrors.push('Admin Password is required');
    
    if (validationErrors.length > 0) {
      setError(validationErrors.join(', '));
      setShowAlert(true);
      return;
    }
    
    try {
      setActionLoading(true);
      
      console.log('Creating tenant with data:', JSON.stringify(tenantFormData, null, 2));
      
      // Clean the data before submission
      const cleanedData = {
        ...tenantFormData,
        code: tenantFormData.code.trim().toLowerCase().replace(/[^a-z0-9-]/g, '-'),
        settings: {
          ...tenantFormData.settings,
          maxFieldWorkers: Number(tenantFormData.settings.maxFieldWorkers) || 10,
          maxAdmins: Number(tenantFormData.settings.maxAdmins) || 2
        }
      };
      
      // Create the tenant
      const response = await axios.post(
        `${API_BASE_URL}/admin/tenants`,
        cleanedData,
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      console.log('Tenant creation response:', response.data);
      
      // Send email with credentials to the original email from the access request
      // Use selectedRequest.email to ensure we're sending to the original requester
      await axios.post(
        `${API_BASE_URL}/admin/send-tenant-credentials`,
        {
          email: selectedRequest.email, // Use original email from access request
          tenantName: tenantFormData.name,
          adminName: tenantFormData.adminName,
          password: tenantFormData.adminPassword,
          loginUrl: window.location.origin + '/login'
        },
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      // Mark the access request as having a tenant created
      if (selectedRequest && selectedRequest._id) {
        await axios.patch(
          `${API_BASE_URL}/access-requests/${selectedRequest._id}/mark-tenant-created`,
          {},
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );
      }
      
      setSuccess('Tenant created successfully and credentials sent by email');
      setShowAlert(true);
      setCreateTenantDialogOpen(false);
      // Refresh the access requests to update the UI
      fetchAccessRequests();
    } catch (err) {
      console.error('Tenant creation error:', err);
      const errorData = err.response?.data;
      console.error('Error response data:', errorData);
      setError(errorData?.message || err.message || 'Failed to create tenant');
      setShowAlert(true);
    } finally {
      setActionLoading(false);
    }
  };

  const getStatusChip = (status) => {
    switch (status) {
      case 'pending':
        return <Chip size="small" label="Pending" color="warning" />;
      case 'approved':
        return <Chip size="small" label="Approved" color="success" />;
      case 'rejected':
        return <Chip size="small" label="Rejected" color="error" />;
      default:
        return <Chip size="small" label={status} />;
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return format(new Date(dateString), 'MMM d, yyyy • h:mm a');
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 600 }}>
        Manage Access Requests
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Total Requests
                </Typography>
                <Typography variant="h4" component="div">
                  {stats.total}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                <Typography color="inherit" gutterBottom>
                  Pending
                </Typography>
                <Typography variant="h4" component="div" color="inherit">
                  {stats.pending}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ bgcolor: 'success.light', color: 'success.contrastText' }}>
                <Typography color="inherit" gutterBottom>
                  Approved
                </Typography>
                <Typography variant="h4" component="div" color="inherit">
                  {stats.approved}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ bgcolor: 'error.light', color: 'error.contrastText' }}>
                <Typography color="inherit" gutterBottom>
                  Rejected
                </Typography>
                <Typography variant="h4" component="div" color="inherit">
                  {stats.rejected}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Button
          startIcon={<RefreshIcon />}
          onClick={fetchAccessRequests}
          disabled={loading}
          variant="outlined"
        >
          Refresh
        </Button>
      </Box>

      <Paper>
        <TableContainer>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : requests.length === 0 ? (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="body1">No access requests found.</Typography>
            </Box>
          ) : (
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Organization</TableCell>
                  <TableCell>Contact</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Region</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {requests.map((request) => (
                  <TableRow key={request._id}>
                    <TableCell>{request.organizationName}</TableCell>
                    <TableCell>{request.contactName}</TableCell>
                    <TableCell>{request.email}</TableCell>
                    <TableCell>{request.region}</TableCell>
                    <TableCell>{formatDate(request.createdAt)}</TableCell>
                    <TableCell>{getStatusChip(request.status)}</TableCell>
                    <TableCell align="center">
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewRequest(request)}
                          color="primary"
                        >
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                      
                      {request.status === 'pending' && (
                        <>
                          <Tooltip title="Approve">
                            <IconButton
                              size="small"
                              onClick={() => handleStatusAction(request, 'approved')}
                              color="success"
                            >
                              <ApproveIcon />
                            </IconButton>
                          </Tooltip>
                          
                          <Tooltip title="Reject">
                            <IconButton
                              size="small"
                              onClick={() => handleStatusAction(request, 'rejected')}
                              color="error"
                            >
                              <RejectIcon />
                            </IconButton>
                          </Tooltip>
                        </>
                      )}
                      
                      {request.status === 'approved' && !request.tenantCreated && (
                        <Tooltip title="Complete Tenant Creation">
                          <IconButton
                            size="small"
                            onClick={() => handleCompleteTenantCreation(request)}
                            color="primary"
                          >
                            <BusinessIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                      
                      {request.status === 'approved' && request.tenantCreated && (
                        <Tooltip title="Tenant Created">
                          <IconButton
                            size="small"
                            disabled
                            color="success"
                          >
                            <CheckCircleIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                      
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          onClick={() => handleDeleteRequest(request._id)}
                          color="default"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </TableContainer>
      </Paper>

      {/* View Request Dialog */}
      <Dialog
        open={viewDialogOpen}
        onClose={() => setViewDialogOpen(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>Request Details</DialogTitle>
        <DialogContent>
          {selectedRequest && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Organization Name
                  </Typography>
                  <Typography variant="body1">
                    {selectedRequest.organizationName}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Contact Person
                  </Typography>
                  <Typography variant="body1">
                    {selectedRequest.contactName}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Email Address
                  </Typography>
                  <Typography variant="body1">
                    {selectedRequest.email}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Phone Number
                  </Typography>
                  <Typography variant="body1">
                    {selectedRequest.phone}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Region
                  </Typography>
                  <Typography variant="body1">
                    {selectedRequest.region}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Date Submitted
                  </Typography>
                  <Typography variant="body1">
                    {formatDate(selectedRequest.createdAt)}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Reason for Access
                  </Typography>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {selectedRequest.reason}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Status
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    {getStatusChip(selectedRequest.status)}
                  </Box>
                </Box>
              </Grid>
              
              {selectedRequest.status === 'approved' && (
                <Grid item xs={12}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Tenant Status
                    </Typography>
                    <Box sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                      {selectedRequest.tenantCreated ? (
                        <Chip 
                          icon={<CheckCircleIcon />} 
                          label="Tenant created" 
                          color="success" 
                          variant="outlined" 
                        />
                      ) : (
                        <Chip 
                          icon={<BusinessIcon />} 
                          label="Pending tenant creation" 
                          color="primary" 
                          variant="outlined" 
                        />
                      )}
                    </Box>
                  </Box>
                </Grid>
              )}
              
              {selectedRequest.reviewNotes && (
                <Grid item xs={12}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Review Notes
                    </Typography>
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                      {selectedRequest.reviewNotes}
                    </Typography>
                  </Box>
                </Grid>
              )}
              
              {selectedRequest.reviewedAt && (
                <Grid item xs={12}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Reviewed At
                    </Typography>
                    <Typography variant="body1">
                      {formatDate(selectedRequest.reviewedAt)}
                    </Typography>
                  </Box>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Status Update Dialog */}
      <Dialog
        open={statusDialogOpen}
        onClose={() => setStatusDialogOpen(false)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>
          {dialogAction === 'approved' ? 'Approve Request' : 'Reject Request'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Review Notes"
              multiline
              rows={4}
              value={reviewNotes}
              onChange={(e) => setReviewNotes(e.target.value)}
              placeholder={
                dialogAction === 'approved'
                  ? 'Enter any additional notes or instructions for the approved request...'
                  : 'Enter the reason for rejecting this request...'
              }
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStatusDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleStatusUpdate}
            color={dialogAction === 'approved' ? 'success' : 'error'}
            disabled={actionLoading}
            variant="contained"
          >
            {actionLoading ? <CircularProgress size={24} /> : dialogAction === 'approved' ? 'Approve' : 'Reject'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Tenant Creation Dialog */}
      <Dialog
        open={createTenantDialogOpen}
        onClose={() => {
          setError('');
          setCreateTenantDialogOpen(false);
        }}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <BusinessIcon sx={{ mr: 1 }} />
            Create Tenant for {selectedRequest?.organizationName}
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <Typography variant="body2" color="textSecondary" paragraph>
              Create a tenant for the approved access request. Login credentials will be sent to the tenant administrator by email.
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Tenant Name *"
                  name="name"
                  value={tenantFormData.name}
                  onChange={handleTenantFormChange}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                  <TextField
                    fullWidth
                    label="Tenant Code *"
                    name="code"
                    value={tenantFormData.code}
                    onChange={handleTenantFormChange}
                    margin="normal"
                    helperText="Unique code for the tenant, used in URLs and APIs"
                  />
                  <Button 
                    variant="outlined" 
                    size="small" 
                    onClick={() => setTenantFormData({...tenantFormData, code: generateUniqueCode()})}
                    sx={{ mt: 2 }}
                  >
                    Generate
                  </Button>
                </Box>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  name="description"
                  value={tenantFormData.description}
                  onChange={handleTenantFormChange}
                  margin="normal"
                  multiline
                  rows={2}
                />
              </Grid>
            </Grid>

            <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
              Admin User
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Admin Name *"
                  name="adminName"
                  value={tenantFormData.adminName}
                  onChange={handleTenantFormChange}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Admin Email *"
                  name="adminEmail"
                  value={tenantFormData.adminEmail}
                  onChange={handleTenantFormChange}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                  <TextField
                    fullWidth
                    label="Admin Password *"
                    name="adminPassword"
                    value={tenantFormData.adminPassword}
                    onChange={handleTenantFormChange}
                    margin="normal"
                    type="text"
                    helperText="Password will be sent by email to the admin"
                  />
                  <Button 
                    variant="outlined" 
                    size="small" 
                    onClick={() => setTenantFormData({...tenantFormData, adminPassword: generatePassword(12)})}
                    sx={{ mt: 2 }}
                  >
                    Generate
                  </Button>
                </Box>
              </Grid>
            </Grid>
            {error && (
              <Box sx={{ mt: 3 }}>
                <Alert severity="error">{error}</Alert>
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setError('');
            setCreateTenantDialogOpen(false);
          }}>Cancel</Button>
          <Button
            onClick={handleCreateTenant}
            variant="contained"
            color="primary"
            disabled={actionLoading}
            startIcon={actionLoading ? null : <EmailIcon />}
          >
            {actionLoading ? <CircularProgress size={24} /> : 'Create Tenant & Send Email'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar 
        open={showAlert} 
        autoHideDuration={6000} 
        onClose={handleCloseAlert}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseAlert} 
          severity={error ? 'error' : 'success'} 
          sx={{ width: '100%' }}
        >
          {error || success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ManageAccessRequests;
