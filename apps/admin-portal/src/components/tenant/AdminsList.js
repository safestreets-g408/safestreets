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
import PersonIcon from '@mui/icons-material/Person';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import { TOKEN_KEY, API_BASE_URL, API_ENDPOINTS } from '../../config/constants';

const AdminsList = ({ tenantId, admins, setAdmins }) => {
  const [openAddAdminDialog, setOpenAddAdminDialog] = useState(false);
  const [dialogLoading, setDialogLoading] = useState(false);
  const [dialogError, setDialogError] = useState('');
  const [newAdmin, setNewAdmin] = useState({ name: '', email: '', password: '' });
  
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewAdmin(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleAddAdmin = async () => {
    try {
      setDialogLoading(true);
      setDialogError('');
      
      if (!newAdmin.name || !newAdmin.email || !newAdmin.password) {
        setDialogError('All fields are required');
        setDialogLoading(false);
        return;
      }
      
      const token = localStorage.getItem(TOKEN_KEY);
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ADMIN}/tenants/${tenantId}/admins`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(newAdmin)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to add admin');
      }
      
      const newAdminData = await response.json();
      setAdmins([...admins, newAdminData]);
      
      // Reset form and close dialog
      setNewAdmin({ name: '', email: '', password: '' });
      setOpenAddAdminDialog(false);
    } catch (err) {
      setDialogError(err.message);
    } finally {
      setDialogLoading(false);
    }
  };

  return (
    <>
      {admins.length > 0 ? (
        <List>
          {admins.map((admin) => (
            <ListItem
              key={admin._id}
              secondaryAction={
                <Box>
                  <Chip 
                    label={admin.role === 'tenant-owner' ? 'Owner' : 'Admin'} 
                    color={admin.role === 'tenant-owner' ? 'primary' : 'default'}
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  {admin.role !== 'tenant-owner' && (
                    <>
                      <IconButton size="small">
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton size="small" color="error">
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </>
                  )}
                </Box>
              }
              sx={{ borderBottom: '1px solid #f0f0f0' }}
            >
              <ListItemAvatar>
                <Avatar>
                  <PersonIcon />
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={admin.name}
                secondary={admin.email}
              />
            </ListItem>
          ))}
        </List>
      ) : (
        <Typography>No admin users for this tenant yet.</Typography>
      )}
      
      <Box mt={2} sx={{ display: 'flex', gap: 2 }}>
        <Button 
          variant="contained" 
          color="primary"
          onClick={() => setOpenAddAdminDialog(true)}
        >
          Add Admin
        </Button>
      </Box>

      {/* Add Admin Dialog */}
      <Dialog open={openAddAdminDialog} onClose={() => setOpenAddAdminDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Admin</DialogTitle>
        <DialogContent dividers>
          {dialogError && (
            <Box mb={2}>
              <Typography color="error">{dialogError}</Typography>
            </Box>
          )}
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField 
                fullWidth
                label="Name"
                name="name"
                value={newAdmin.name}
                onChange={handleInputChange}
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField 
                fullWidth
                label="Email"
                name="email"
                type="email"
                value={newAdmin.email}
                onChange={handleInputChange}
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField 
                fullWidth
                label="Password"
                name="password"
                type="password"
                value={newAdmin.password}
                onChange={handleInputChange}
                required
                helperText="Minimum 6 characters"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAddAdminDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleAddAdmin}
            disabled={dialogLoading}
          >
            {dialogLoading ? <CircularProgress size={24} /> : 'Add Admin'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default AdminsList;
