import React, { useState } from 'react';
import {
  Box, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemAvatar, 
  Avatar, 
  Chip, 
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Grid,
  CircularProgress,
  IconButton
} from '@mui/material';
import EngineeringIcon from '@mui/icons-material/Engineering';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import { TOKEN_KEY, API_BASE_URL, API_ENDPOINTS } from '../../config/constants';

const FieldWorkersList = ({ tenantId, fieldWorkers, setFieldWorkers }) => {
  const [openAddFieldWorkerDialog, setOpenAddFieldWorkerDialog] = useState(false);
  const [dialogLoading, setDialogLoading] = useState(false);
  const [dialogError, setDialogError] = useState('');
  const [newFieldWorker, setNewFieldWorker] = useState({ 
    name: '', 
    email: '', 
    password: '', 
    phone: '',
    workerId: '',
    specialization: '',
    region: ''
  });
  
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewFieldWorker(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleAddFieldWorker = async () => {
    try {
      setDialogLoading(true);
      setDialogError('');
      
      if (!newFieldWorker.name || !newFieldWorker.email || !newFieldWorker.password || 
          !newFieldWorker.workerId || !newFieldWorker.specialization || !newFieldWorker.region) {
        setDialogError('All required fields must be filled out');
        setDialogLoading(false);
        return;
      }
      
      const token = localStorage.getItem(TOKEN_KEY);
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ADMIN}/tenants/${tenantId}/field-workers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(newFieldWorker)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to add field worker');
      }
      
      const newWorkerData = await response.json();
      setFieldWorkers([...fieldWorkers, newWorkerData]);
      
      // Reset form and close dialog
      setNewFieldWorker({ 
        name: '', 
        email: '', 
        password: '', 
        phone: '',
        workerId: '',
        specialization: '',
        region: ''
      });
      setOpenAddFieldWorkerDialog(false);
    } catch (err) {
      setDialogError(err.message);
    } finally {
      setDialogLoading(false);
    }
  };

  return (
    <>
      {fieldWorkers.length > 0 ? (
        <List>
          {fieldWorkers.map((worker) => (
            <ListItem
              key={worker._id}
              secondaryAction={
                <Box>
                  <Chip 
                    label={worker.active ? 'Active' : 'Inactive'} 
                    color={worker.active ? 'success' : 'default'}
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  <IconButton size="small">
                    <EditIcon fontSize="small" />
                  </IconButton>
                  <IconButton size="small" color="error">
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
              }
              sx={{ borderBottom: '1px solid #f0f0f0' }}
            >
              <ListItemAvatar>
                <Avatar>
                  <EngineeringIcon />
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={worker.name}
                secondary={worker.email}
              />
            </ListItem>
          ))}
        </List>
      ) : (
        <Typography>No field workers for this tenant yet.</Typography>
      )}
      
      <Box mt={2} sx={{ display: 'flex', gap: 2 }}>
        <Button 
          variant="contained" 
          color="primary"
          onClick={() => setOpenAddFieldWorkerDialog(true)}
        >
          Add Field Worker
        </Button>
      </Box>

      {/* Add Field Worker Dialog */}
      <Dialog open={openAddFieldWorkerDialog} onClose={() => setOpenAddFieldWorkerDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Field Worker</DialogTitle>
        <DialogContent dividers>
          {dialogError && (
            <Box mb={2}>
              <Typography color="error">{dialogError}</Typography>
            </Box>
          )}
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Name"
                name="name"
                value={newFieldWorker.name}
                onChange={handleInputChange}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Worker ID"
                name="workerId"
                value={newFieldWorker.workerId}
                onChange={handleInputChange}
                required
                helperText="Unique identifier for this worker"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Email"
                name="email"
                type="email"
                value={newFieldWorker.email}
                onChange={handleInputChange}
                required
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Password"
                name="password"
                type="password"
                value={newFieldWorker.password}
                onChange={handleInputChange}
                required
                helperText="Minimum 6 characters"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Specialization"
                name="specialization"
                value={newFieldWorker.specialization}
                onChange={handleInputChange}
                required
                helperText="E.g., Pothole Repair, Street Light Maintenance"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Region"
                name="region"
                value={newFieldWorker.region}
                onChange={handleInputChange}
                required
                helperText="Geographical working area"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField 
                fullWidth
                label="Phone Number"
                name="phone"
                value={newFieldWorker.phone}
                onChange={handleInputChange}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAddFieldWorkerDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleAddFieldWorker}
            disabled={dialogLoading}
          >
            {dialogLoading ? <CircularProgress size={24} /> : 'Add Field Worker'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default FieldWorkersList;
