import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Button, 
  Tab, Tabs, 
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

  // The backend now calculates active assignments directly

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true);
        
        // Fetch both field workers and repairs in parallel
        await Promise.all([
          (async () => {
            // The backend now calculates active assignments correctly
            const response = await api.get('/field/workers');
            console.log('Field workers response:', response);
            
            // Transform the workers with the data from backend
            const transformedWorkers = response.map(worker => ({
              id: worker._id,
              workerId: worker.workerId,
              name: worker.name,
              email: worker.email,
              phone: worker.profile?.phone || '',
              personalEmail: worker.profile?.personalEmail || '',
              specialization: worker.specialization,
              region: worker.region,
              profile: worker.profile || {},
              // The backend now calculates active assignments correctly
              activeAssignments: worker.activeAssignments || 0,
              status: typeof worker.status === 'object' ? 'Available' : worker.status || (worker.activeAssignments >= 3 ? 'Busy' : 'Available')
            }));
            
            // Set workers data
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
        setSnackbarMessage(`Assigned to ${worker.name}`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        fetchRepairs(); // Refresh data after API call
      }
    } catch (err) {
      console.error('Error assigning repair:', err);
      setError(`Failed to assign repair: ${err.message || 'Unknown error'}`);
      
      // Show error notification
      setSnackbarMessage(`Failed: ${err.message || 'Try again'}`);
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
        setSnackbarMessage(`Status: ${newStatus}`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        fetchRepairs(); // Refresh data after API call
        return true; // Indicate success
      } else {
        console.warn('Status update response missing success flag:', response);
        setError('Response received but operation may have failed');
        
        // Show warning notification
        setSnackbarMessage('Verification needed');
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

  const handleAddFieldWorker = async (workerData) => {
    try {
      console.log('Adding field worker:', workerData);
      setError(null);
      
      // Prepare the data for the API
      const workerPayload = {
        name: workerData.name,
        workerId: workerData.workerId,
        email: workerData.email,
        specialization: workerData.specialization,
        region: workerData.region,
        phone: workerData.phone,
        personalEmail: workerData.personalEmail,
        status: workerData.status
      };
      
      const response = await api.post('/field/add', workerPayload);
      
      if (response) {
        console.log('Field worker added successfully:', response);
        
        // Add the new worker to the state with a derived status
        const newWorker = {
          id: response._id,
          workerId: response.workerId,
          name: response.name,
          email: response.email,
          specialization: response.specialization,
          region: response.region,
          activeAssignments: response.activeAssignments || 0,
          profile: response.profile || {},
          personalEmail: response.profile?.personalEmail || workerData.personalEmail || '',
          phone: response.profile?.phone || workerData.phone || '',
          status: workerData.status || 'Available'
        };
        
        setFieldWorkers(prevWorkers => [...prevWorkers, newWorker]);
        
        // Show success notification
        setSnackbarMessage(`Added: ${workerData.name}`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        return true;
      }
    } catch (err) {
      console.error('Error adding field worker:', err);
      setError(`Failed to add field worker: ${err.message || 'Unknown error'}`);
      
      // Show error notification
      setSnackbarMessage(`Error: ${err.message || 'Could not add'}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
      
      return false;
    }
  };

  const handleEditFieldWorker = async (workerId, workerData) => {
    try {
      console.log('Editing field worker:', workerId, workerData);
      console.log('Personal email to update:', workerData.personalEmail);
      setError(null);
      
      // Prepare the data for the API - send fields as the backend expects them
      const workerPayload = {
        name: workerData.name,
        specialization: workerData.specialization,
        region: workerData.region,
        phone: workerData.phone,
        personalEmail: workerData.personalEmail,
        // If personalEmail is provided, enable daily updates by default
        receiveDailyUpdates: workerData.personalEmail ? true : false,
        status: workerData.status
      };
      
      console.log('Sending payload to update worker:', workerPayload);
      const response = await api.put(`/field/${workerId}`, workerPayload);
      
      if (response) {
        console.log('Field worker updated successfully:', response);
        
        console.log('Updated field worker response:', response);
        
        // Update the worker in the state with data from response
        setFieldWorkers(prevWorkers => 
          prevWorkers.map(worker => 
            worker.workerId === workerId 
              ? {
                  ...worker,
                  name: response.name || workerData.name,
                  specialization: response.specialization || workerData.specialization,
                  region: response.region || workerData.region,
                  profile: {
                    ...(worker.profile || {}),
                    ...(response.profile || {}),
                    phone: response.profile?.phone || workerData.phone,
                    personalEmail: response.profile?.personalEmail || workerData.personalEmail
                  },
                  // Keep these fields separately for easy access in UI
                  personalEmail: response.profile?.personalEmail || workerData.personalEmail, 
                  phone: response.profile?.phone || workerData.phone,
                  status: workerData.status
                }
              : worker
          )
        );
        
        // Show success notification
        setSnackbarMessage(`Updated: ${workerData.name}`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        return true;
      }
    } catch (err) {
      console.error('Error updating field worker:', err);
      setError(`Failed to update field worker: ${err.message || 'Unknown error'}`);
      
      // Show error notification
      setSnackbarMessage(`Error: ${err.message || 'Could not update'}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
      
      return false;
    }
  };
  
  const handleDeleteFieldWorker = async (workerId) => {
    try {
      console.log('Deleting field worker:', workerId);
      setError(null);
      
      const worker = fieldWorkers.find(w => w.workerId === workerId);
      if (!worker) {
        throw new Error(`Worker with ID ${workerId} not found`);
      }
      
      // Since there's no delete endpoint, we'll just mark the worker as inactive
      // by updating their status. In a real app, you would have a proper delete endpoint
      const workerPayload = {
        status: 'Inactive'
      };
      
      const response = await api.put(`/field/${workerId}`, workerPayload);
      
      if (response) {
        console.log('Field worker deleted successfully:', response);
        
        // Remove the worker from the state
        setFieldWorkers(prevWorkers => 
          prevWorkers.filter(worker => worker.workerId !== workerId)
        );
        
        // Show success notification
        setSnackbarMessage(`Removed: ${worker.name}`);
        setSnackbarSeverity('success');
        setSnackbarOpen(true);
        
        return true;
      }
    } catch (err) {
      console.error('Error deleting field worker:', err);
      setError(`Failed to delete field worker: ${err.message || 'Unknown error'}`);
      
      // Show error notification
      setSnackbarMessage(`Error: ${err.message || 'Could not remove'}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
      
      return false;
    }
  };

  // Function to handle viewing a worker's assignments
  const handleViewAssignments = async (workerId) => {
    try {
      setError(null);
      
      const worker = fieldWorkers.find(w => w.id === workerId);
      if (!worker) {
        throw new Error(`Worker with ID ${workerId} not found`);
      }
      
      // Use the new dedicated endpoint to get assignments for this worker
      const response = await api.get(`/field/${worker.workerId}/assignments`);
      const assignments = response?.assignments || [];
      
      if (assignments && assignments.length > 0) {
        // Format a message with assignment details in a minimal format
        const assignmentDetails = assignments.map((a, idx) => 
          `${a.reportId}: ${a.damageType} (${a.status})`
        ).join('\n');
        
        setSnackbarSeverity('info');
        setSnackbarMessage(`${worker.name}: ${assignments.length} assigned\n${assignmentDetails}`);
      } else {
        setSnackbarSeverity('info');
        setSnackbarMessage(`${worker.name}: No assignments`);
      }
      
      setSnackbarOpen(true);
      return assignments;
    } catch (err) {
      console.error('Error fetching worker assignments:', err);
      setError(`Failed to fetch assignments: ${err.message || 'Unknown error'}`);
      
      setSnackbarSeverity('error');
      setSnackbarMessage(`Error: Could not fetch assignments`);
      setSnackbarOpen(true);
      
      return [];
    }
  };

  return (
    <>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="standard"
        >
          <Tab 
            label="Pending" 
            icon={<AssignmentIcon sx={{ fontSize: '1rem' }} />}
            iconPosition="start"
            sx={{ minHeight: '40px', fontSize: '0.875rem', textTransform: 'none', fontWeight: 500 }}
          />
          <Tab 
            label="Active" 
            icon={<BuildIcon sx={{ fontSize: '1rem' }} />}
            iconPosition="start"
            sx={{ minHeight: '40px', fontSize: '0.875rem', textTransform: 'none', fontWeight: 500 }}
          />
          <Tab 
            label="Personnel" 
            icon={<PersonIcon sx={{ fontSize: '1rem' }} />}
            iconPosition="start"
            sx={{ minHeight: '40px', fontSize: '0.875rem', textTransform: 'none', fontWeight: 500 }}
          />
        </Tabs>
        {pendingRepairs.length > 0 && (
          <Typography 
            variant="caption" 
            color="text.secondary" 
            sx={{ ml: 2, fontSize: '0.75rem', position: 'relative', top: '-4px' }}
          >
            {pendingRepairs.length} pending
          </Typography>
        )}
      </Box>
      
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
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 120 }}>
              <CircularProgress size={20} thickness={3} />
            </Box>
          ) : error ? (
            <Box sx={{ textAlign: 'left', py: 2, px: 1 }}>
              <Typography variant="caption" color="error" sx={{ display: 'block', mb: 1, fontWeight: 500 }}>
                {error}
              </Typography>
              <Button 
                variant="text" 
                size="small"
                color="primary"
                onClick={() => window.location.reload()}
                sx={{ fontSize: '0.75rem', p: 0.5 }}
              >
                Reload
              </Button>
            </Box>
          ) : (
            <FieldWorker 
              fieldWorkers={fieldWorkers}
              onAddWorker={handleAddFieldWorker}
              onEditWorker={handleEditFieldWorker}
              onDeleteWorker={handleDeleteFieldWorker}
              onViewAssignments={handleViewAssignments}
            />
          )}
        </>
      )}
      
      {/* Assignment Dialog is now handled within the PendingAssignments component */}
      
      {/* Snackbar for status updates and notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={4000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbarSeverity} 
          variant="standard"
          sx={{ 
            py: 0.5,
            width: '100%', 
            maxWidth: '320px',
            whiteSpace: 'pre-line', 
            overflowWrap: 'break-word',
            '& .MuiAlert-icon': { fontSize: '0.875rem', py: 0.5 },
            '& .MuiAlert-message': { fontSize: '0.8125rem', py: 0.5 },
            '& .MuiAlert-action': { pt: 0, pb: 0, ml: 0.5 }
          }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </>
  );
}

export default Repair;
