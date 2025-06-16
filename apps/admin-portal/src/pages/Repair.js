import React, { useState, useEffect } from 'react';
import { 
  Box, Typography,  TextField, Button, 
  Select, MenuItem, FormControl, InputLabel,Avatar, 
   Tab, Tabs, Badge, 
  Dialog, DialogTitle, DialogContent, DialogActions,
  FormHelperText, useTheme, CircularProgress,
  Snackbar, Alert
} from '@mui/material';
import { 
  Build as BuildIcon,
  Assignment as AssignmentIcon,
  Person as PersonIcon,
} from '@mui/icons-material';
import { api } from '../utils/api';

import ActiveRepairs from '../components/RepairManagement/ActiveRepairs';
import PendingAssignments from '../components/RepairManagement/PendingAssignments';
import FieldWorker from '../components/RepairManagement/FieldWorker';

function Repair() {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [selectedRepair, setSelectedRepair] = useState(null);
  const [assignDialogOpen, setAssignDialogOpen] = useState(false);
  const [selectedWorker, setSelectedWorker] = useState('');
  const [assignmentNotes, setAssignmentNotes] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fieldWorkers, setFieldWorkers] = useState([]);
  const [pendingRepairs, setPendingRepairs] = useState([]);
  const [assignedRepairs, setAssignedRepairs] = useState([]);
  
  // Snackbar for notifications
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success'); // success, error, warning, info

  const fetchRepairs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch pending reports (status: Pending)
      const pendingResponse = await api.get('/damage/reports?status=Pending');
      setPendingRepairs(pendingResponse);

      // Fetch all assigned reports (status: not Pending)
      const assignedResponse = await api.get('/damage/reports?status=!Pending');
      setAssignedRepairs(assignedResponse);
    } catch (err) {
      console.error('Error fetching repairs:', err);
      setError(err.message || 'Failed to fetch repairs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true);
        
        // Fetch both field workers and repairs in parallel
        await Promise.all([
          (async () => {
            const response = await api.get('/field/workers');
            const transformedWorkers = response.map(worker => ({
              id: worker._id,
              workerId: worker.workerId,
              name: worker.name,
              specialization: worker.specialization,
              region: worker.region,
              activeAssignments: worker.activeAssignments || 0,
              status: (worker.activeAssignments >= 3) ? 'Busy' : 'Available'
            }));
            setFieldWorkers(transformedWorkers);
          })(),
          fetchRepairs()
        ]);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message || 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchAllData();
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleAssignDialogOpen = (repair) => {
    console.log('Opening assign dialog with repair:', repair);
    setSelectedRepair(repair);
    setAssignDialogOpen(true);
  };

  const handleAssignDialogClose = () => {
    setAssignDialogOpen(false);
    setSelectedWorker('');
    setAssignmentNotes('');
  };

  const handleAssignRepair = async () => {
    if (!selectedRepair || !selectedWorker) {
      console.error('Cannot assign: Missing repair or worker', { 
        repair: selectedRepair, 
        worker: selectedWorker 
      });
      return;
    }
    
    const reportId = selectedRepair._id || selectedRepair.id;
    console.log('Assigning repair', { reportId, workerId: selectedWorker, notes: assignmentNotes });
    
    try {
      const response = await api.patch(`/damage/reports/${reportId}/assign`, {
        workerId: selectedWorker,
        notes: assignmentNotes
      });
      
      if (response && response.success) {
        // Show success message
        console.log('Repair assigned successfully', response);
        setError(null);
        handleAssignDialogClose();
        fetchRepairs(); // Refresh data after API call
      }
    } catch (err) {
      console.error('Error assigning repair:', err);
      setError(`Failed to assign repair: ${err.message || 'Unknown error'}`);
    }
  };

  const handleStatusChange = async (repairId, newStatus) => {
    try {
      setError(null);
      console.log(`Updating repair ${repairId} status to ${newStatus}`);
      
      // Convert frontend status format to backend format if needed
      let backendStatus = newStatus;
      if (newStatus === 'In-Progress') backendStatus = 'In Progress';
      
      const response = await api.patch(`/damage/reports/${repairId}/status`, {
        status: backendStatus
      });
      
      if (response && response.success) {
        console.log(`Repair status updated successfully to ${newStatus}`, response);
        
        // Show success notification
        setSnackbarMessage(`Repair status updated to ${newStatus}`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        fetchRepairs(); // Refresh data after API call
        return true; // Indicate success
      } else {
        console.warn('Status update response missing success flag:', response);
        setError('Response received but operation may have failed');
        
        // Show warning notification
        setSnackbarMessage('Status update may have failed. Please check and try again.');
        setSnackbarSeverity('warning');
        setSnackbarOpen(true);
        
        return false; // Indicate issues
      }
    } catch (err) {
      console.error('Error updating repair status:', err);
      setError(`Failed to update status: ${err.message || 'Unknown error'}`);
      
      // Show error notification
      setSnackbarMessage(`Failed to update status: ${err.message || 'Unknown error'}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
      
      throw err; // Propagate the error for proper Promise handling
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Low': return theme.palette.info.main;
      case 'Medium': return theme.palette.warning.main;
      case 'High': return theme.palette.error.main;
      case 'Critical': return theme.palette.secondary.main;
      default: return theme.palette.grey[500];
    }
  };

  // Helper to close the snackbar
  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') return;
    setSnackbarOpen(false);
  };

  return (
    <>
      
      <Tabs 
        value={tabValue} 
        onChange={handleTabChange} 
        sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab 
          label={
            <Badge badgeContent={pendingRepairs.length} color="warning" max={99}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <AssignmentIcon sx={{ mr: 1 }} />
                Pending Assignments
              </Box>
            </Badge>
          } 
        />
        <Tab 
          label={
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <BuildIcon sx={{ mr: 1 }} />
              Active Repairs
            </Box>
          } 
        />
        <Tab 
          label={
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <PersonIcon sx={{ mr: 1 }} />
              Field Workers
            </Box>
          } 
        />
      </Tabs>
      
      {/* Pending Assignments Tab */}
      {tabValue === 0 && (
        <PendingAssignments 
          pendingRepairs={pendingRepairs}
          getSeverityColor={getSeverityColor}
          handleAssignDialogOpen={handleAssignDialogOpen}
        />
      )}
      
      {/* Active Repairs Tab */}
      {tabValue === 1 && (
        <ActiveRepairs 
          assignedRepairs={assignedRepairs}
          fieldWorkers={fieldWorkers}
          onStatusChange={handleStatusChange}
          theme={theme}
        />
      )}
      
      {/* Field Workers Tab */}
      {tabValue === 2 && (
        <>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Box sx={{ textAlign: 'center', color: 'error.main', py: 3 }}>
              <Typography>{error}</Typography>
              <Button 
                variant="contained" 
                sx={{ mt: 2 }}
                onClick={() => window.location.reload()}
              >
                Retry
              </Button>
            </Box>
          ) : (
            <FieldWorker 
              fieldWorkers={fieldWorkers}
              theme={theme}
            />
          )}
        </>
      )}
      
      {/* Assignment Dialog */}
      <Dialog open={assignDialogOpen} onClose={handleAssignDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>Assign Repair Task</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2, mt: 1 }}>
            <Typography variant="subtitle1">
              Damage Report: {selectedRepair?.id}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {selectedRepair?.description}
            </Typography>
          </Box>
          
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>Assign to Field Worker</InputLabel>
            <Select
              value={selectedWorker}
              label="Assign to Field Worker"
              onChange={(e) => setSelectedWorker(e.target.value)}
            >
              {fieldWorkers.map(worker => (
                <MenuItem 
                  key={worker.id} 
                  value={worker.name}
                  disabled={worker.status !== 'Available'}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Avatar 
                      sx={{ 
                        width: 24, 
                        height: 24, 
                        mr: 1, 
                        bgcolor: worker.status === 'Available' ? theme.palette.success.main : theme.palette.warning.main 
                      }}
                    >
                      {worker.name.split(' ').map(n => n[0]).join('')}
                    </Avatar>
                    <Box>
                      {worker.name}
                      <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                        {worker.specialization} • {worker.region} • {worker.activeAssignments} active tasks
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
            <FormHelperText>Select an available field worker to assign this repair task</FormHelperText>
          </FormControl>
          
          <TextField
            label="Assignment Notes"
            multiline
            rows={4}
            fullWidth
            value={assignmentNotes}
            onChange={(e) => setAssignmentNotes(e.target.value)}
            placeholder="Add any specific instructions or notes for the field worker..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleAssignDialogClose}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={handleAssignRepair}
            disabled={!selectedWorker}
          >
            Assign Task
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Snackbar for status updates and notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbarSeverity} 
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </>
  );
}

export default Repair;
