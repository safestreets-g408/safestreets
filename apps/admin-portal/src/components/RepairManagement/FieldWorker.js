import { useState } from 'react';
import {
  Box, Typography, Grid, Card, CardContent, CardActions,
  Button, Divider, IconButton, Tooltip,
  Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Paper, Alert, Snackbar
} from '@mui/material';
import {
  Assignment as AssignmentIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Person as PersonIcon,
  Email as EmailIcon,
  Phone as PhoneIcon
} from '@mui/icons-material';

function FieldWorker({ 
  fieldWorkers = [], 
  onAddWorker = () => {}, 
  onEditWorker = () => {}, 
  onDeleteWorker = () => {},
  onViewAssignments = () => {}
}) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [selectedWorker, setSelectedWorker] = useState(null);
  const [errors, setErrors] = useState({});
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [workerData, setWorkerData] = useState({
    name: '',
    workerId: '',
    email: '',
    phone: '',
    personalEmail: '',
    specialization: '',
    region: '',
    status: 'Available'
  });

  const resetWorkerData = () => ({
    name: '',
    workerId: '',
    email: '',
    phone: '',
    personalEmail: '',
    specialization: '',
    region: '',
    status: 'Available'
  });

  const handleDialogOpen = (mode, worker = null) => {
    setEditMode(mode === 'edit');
    setSelectedWorker(worker);
    setErrors({});

    if (mode === 'edit' && worker) {
      setWorkerData({
        name: worker.name || '',
        workerId: worker.workerId || '',
        email: worker.email || '',
        phone: worker.profile?.phone || '',
        personalEmail: worker.profile?.personalEmail || worker.personalEmail || '',
        specialization: worker.specialization || '',
        region: worker.region || '',
        status: typeof worker.status === 'object' ? 'Available' : worker.status || 'Available'
      });
    } else {
      setWorkerData(resetWorkerData());
    }

    setDialogOpen(true);
  };

  const handleDialogClose = () => {
    setDialogOpen(false);
    setSelectedWorker(null);
    setWorkerData(resetWorkerData());
    setErrors({});
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Clear error for this field when user types
    if (errors[name]) {
      setErrors(prev => ({...prev, [name]: ''}));
    }

    setWorkerData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const validateForm = () => {
    const newErrors = {};
    
    // Required fields validation
    if (!workerData.name.trim()) newErrors.name = 'Name is required';
    if (!workerData.specialization.trim()) newErrors.specialization = 'Specialization is required';
    if (!workerData.region.trim()) newErrors.region = 'Region is required';
    
    // Only validate workerId and email for new workers
    if (!editMode) {
      if (!workerData.workerId.trim()) newErrors.workerId = 'Worker ID is required';
      if (!workerData.email.trim()) {
        newErrors.email = 'Email is required';
      } else if (!/\S+@\S+\.\S+/.test(workerData.email)) {
        newErrors.email = 'Email is invalid';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validateForm()) return;
    
    if (editMode && selectedWorker) {
      // Use the correct field for the worker ID - could be id or workerId
      const workerId = selectedWorker.workerId || selectedWorker.id;
      if (!workerId) {
        console.error('Cannot edit worker: Missing worker ID', selectedWorker);
        setErrors({ general: 'Missing worker ID' });
        return;
      }
      
      console.log('Editing worker with ID:', workerId, workerData);
      onEditWorker(workerId, workerData);
      
      setSnackbar({ 
        open: true, 
        message: `Personnel record updated: ${workerData.name}`, 
        severity: 'success' 
      });
    } else {
      // For new workers, make sure we have a workerId
      if (!workerData.workerId) {
        setErrors({ workerId: 'Worker ID is required' });
        return;
      }
      
      console.log('Adding new worker:', workerData);
      onAddWorker(workerData);
      
      setSnackbar({ 
        open: true, 
        message: `Personnel added: ${workerData.name}`, 
        severity: 'success' 
      });
    }
    
    handleDialogClose();
  };

  const handleDeleteWorker = (workerId) => {
    if (!workerId) {
      console.error('Cannot delete worker: Missing worker ID');
      setSnackbar({
        open: true,
        message: 'Cannot remove record: Missing ID',
        severity: 'error'
      });
      return;
    }
    
    if (window.confirm('Are you sure you want to remove this personnel record?')) {
      console.log('Deleting worker with ID:', workerId);
      onDeleteWorker(workerId);
      setSnackbar({ open: true, message: 'Personnel record removed', severity: 'info' });
    }
  };

  // Optionally, add a refresh handler if needed
  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <Paper sx={{ p: 2, borderRadius: 1, boxShadow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
        <Typography variant="h6" sx={{ fontWeight: 500 }}>
          Personnel Management
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<PersonIcon sx={{ fontSize: '1rem' }} />}
            size="small"
            onClick={() => handleDialogOpen('add')}
          >
            Add Personnel
          </Button>
          <Tooltip title="Refresh">
            <IconButton size="small" onClick={handleRefresh}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      <Divider sx={{ mb: 1.5 }} />

      <Grid container spacing={1.5}>
        {fieldWorkers.length === 0 ? (
          <Grid item xs={12}>
            <Box sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="body1" sx={{ mb: 1, fontWeight: 500 }}>
                No personnel records found
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Click "Add Personnel" to add a new record
              </Typography>
            </Box>
          </Grid>
        ) : (
          fieldWorkers.map((worker) => (
            <Grid item xs={12} sm={6} md={4} key={worker.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  boxShadow: 1
                }}
              >
                <CardContent sx={{ flexGrow: 1, pb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                      {worker.name || 'Unnamed'}
                    </Typography>
                    <Typography 
                      variant="caption"
                      sx={{ 
                        fontWeight: 500, 
                        color: typeof worker.status === 'object' || worker.status === 'Available' ? 'success.main' : 'warning.main'
                      }}
                    >
                      {typeof worker.status === 'object' ? 'Available' : worker.status}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    ID: {worker.workerId}
                  </Typography>

                                  <Divider sx={{ my: 0.5 }} />

                  <Box sx={{ mt: 0.5 }}>
                    <Typography variant="body2" sx={{ mb: 0.5, display: 'flex', alignItems: 'center' }}>
                      <EmailIcon fontSize="small" sx={{ mr: 0.5, fontSize: '1rem' }} />
                      <Box component="span" sx={{ fontSize: '0.875rem' }}>
                        {worker.email || 'No email'}
                      </Box>
                    </Typography>

                    <Typography variant="body2" sx={{ mb: 0.5, display: 'flex', alignItems: 'center' }}>
                      <PhoneIcon fontSize="small" sx={{ mr: 0.5, fontSize: '1rem' }} />
                      <Box component="span" sx={{ fontSize: '0.875rem' }}>
                        {worker.profile?.phone || 'No phone'}
                      </Box>
                    </Typography>

                    {(worker.profile?.personalEmail || worker.personalEmail) && (
                      <Typography variant="body2" sx={{ mb: 0.5, display: 'flex', alignItems: 'center' }}>
                        <EmailIcon fontSize="small" sx={{ mr: 0.5, fontSize: '1rem', color: 'text.secondary' }} />
                        <Tooltip title="Personal email for daily updates">
                          <Box component="span" sx={{ fontSize: '0.875rem' }}>
                            {worker.profile?.personalEmail || worker.personalEmail}
                          </Box>
                        </Tooltip>
                      </Typography>
                    )}

                    <Box sx={{ mt: 0.5 }}>
                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                        <Box component="span" sx={{ color: 'text.secondary', width: '5rem', display: 'inline-block' }}>Specialty:</Box>
                        {worker.specialization || '-'}
                      </Typography>

                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                        <Box component="span" sx={{ color: 'text.secondary', width: '5rem', display: 'inline-block' }}>Region:</Box>
                        {worker.region || '-'}
                      </Typography>

                      <Typography variant="body2">
                        <Box component="span" sx={{ color: 'text.secondary', width: '5rem', display: 'inline-block' }}>Assigned:</Box>
                        {worker.activeAssignments ?? 0} task(s)
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
                <CardActions sx={{ justifyContent: 'space-between', pt: 0, pb: 1, px: 2 }}>
                  <Button
                    size="small"
                    variant="text"
                    startIcon={<AssignmentIcon sx={{ fontSize: '1rem' }} />}
                    onClick={() => onViewAssignments(worker.id)}
                    color="primary"
                    sx={{ fontSize: '0.75rem' }}
                  >
                    Assignments
                  </Button>
                  <Box>
                    <Tooltip title="Edit">
                      <IconButton size="small" onClick={() => handleDialogOpen('edit', worker)}>
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteWorker(worker.workerId || worker.id)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </CardActions>
              </Card>
            </Grid>
          ))
        )}
      </Grid>

      {/* Add/Edit Worker Dialog */}
      <Dialog open={dialogOpen} onClose={handleDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>{editMode ? 'Edit Personnel' : 'Add New Personnel'}</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Full Name"
              name="name"
              fullWidth
              value={workerData.name}
              onChange={handleInputChange}
              required
              error={!!errors.name}
              helperText={errors.name}
              autoFocus
            />

            {!editMode && (
              <TextField
                label="Worker ID"
                name="workerId"
                fullWidth
                value={workerData.workerId}
                onChange={handleInputChange}
                placeholder="Unique identifier (FW001, etc.)"
                required
                error={!!errors.workerId}
                helperText={errors.workerId}
              />
            )}

            <TextField
              label="Email"
              name="email"
              type="email"
              fullWidth
              value={workerData.email}
              onChange={handleInputChange}
              disabled={editMode} // Can't change email in edit mode
              required={!editMode}
              error={!!errors.email}
              helperText={errors.email || (editMode ? 'Email cannot be changed' : '')}
            />

            <TextField
              label="Phone Number"
              name="phone"
              fullWidth
              value={workerData.phone}
              onChange={handleInputChange}
              placeholder="Contact phone number"
            />
            
            <TextField
              label="Personal Email (For Daily Updates)"
              name="personalEmail"
              type="email"
              fullWidth
              value={workerData.personalEmail}
              onChange={handleInputChange}
              placeholder="Personal email address"
              helperText="Daily assignment reports will be sent to this email"
            />

            <TextField
              label="Specialization"
              name="specialization"
              fullWidth
              value={workerData.specialization}
              onChange={handleInputChange}
              placeholder="Electrical, Plumbing, Structural, etc."
              required
              error={!!errors.specialization}
              helperText={errors.specialization}
            />

            <TextField
              label="Region"
              name="region"
              fullWidth
              value={workerData.region}
              onChange={handleInputChange}
              placeholder="North, South, East, West, Central"
              required
              error={!!errors.region}
              helperText={errors.region}
            />

            <TextField
              select
              label="Status"
              name="status"
              fullWidth
              value={workerData.status}
              onChange={handleInputChange}
              SelectProps={{
                native: true,
              }}
            >
              <option value="Available">Available</option>
              <option value="Busy">Busy</option>
              <option value="On Leave">On Leave</option>
            </TextField>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose} size="small" variant="text">Cancel</Button>
          <Button
            variant="outlined"
            onClick={handleSubmit}
            size="small"
            color="primary"
          >
            {editMode ? 'Update Record' : 'Add Record'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar notifications */}      <Snackbar
        open={snackbar.open}
        autoHideDuration={5000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          variant="outlined"
          sx={{ 
            width: '100%',
            '& .MuiAlert-icon': { fontSize: '1.2rem' },
            '& .MuiAlert-message': { fontSize: '0.9rem' }
          }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Paper>
  );
}

export default FieldWorker;
