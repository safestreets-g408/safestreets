import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Button, 
  Tab, Tabs, Badge, 
  useTheme, CircularProgress,
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

  // No longer need handleAssignDialogClose as we're using the dialog in PendingAssignments component

  const handleAssignRepair = async (reportId, workerId, notes) => {
    if (!reportId || !workerId) {
      console.error('Cannot assign: Missing repair or worker', { 
        reportId, 
        workerId 
      });
      return;
    }
    
    console.log('Assigning repair', { reportId, workerId, notes });
    
    try {
      // Find worker details to get worker name for success message
      const worker = fieldWorkers.find(w => w.id === workerId);
      if (!worker) {
        console.error('Worker not found with ID:', workerId);
        setError(`Worker not found with ID: ${workerId}`);
        return;
      }
      
      const response = await api.patch(`/damage/reports/${reportId}/assign`, {
        workerId,
        notes
      });
      
      if (response && response.success) {
        // Show success message
        console.log('Repair assigned successfully', response);
        setError(null);
        
        // Show success notification
        setSnackbarMessage(`Repair assigned to ${worker.name} successfully`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        fetchRepairs(); // Refresh data after API call
      }
    } catch (err) {
      console.error('Error assigning repair:', err);
      setError(`Failed to assign repair: ${err.message || 'Unknown error'}`);
      
      // Show error notification
      setSnackbarMessage(`Failed to assign repair: ${err.message || 'Unknown error'}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
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

  // Severity color handling is now done in child components

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
          fieldWorkers={fieldWorkers}
          onAssignRepair={handleAssignRepair}
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
      
      {/* Assignment Dialog is now handled within the PendingAssignments component */}
      
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
