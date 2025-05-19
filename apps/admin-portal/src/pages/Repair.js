import React, { useState } from 'react';
import { 
  Box, Typography,  TextField, Button, 
  Select, MenuItem, FormControl, InputLabel,Avatar, 
   Tab, Tabs, Badge, 
  Dialog, DialogTitle, DialogContent, DialogActions,
  FormHelperText, useTheme,
} from '@mui/material';
import { 
  Build as BuildIcon,
  Assignment as AssignmentIcon,
  AssignmentInd as AssignmentIndIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Pending as PendingIcon,
  Person as PersonIcon,
} from '@mui/icons-material';

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
  const [statusFilter, setStatusFilter] = useState('all');
  const [workerFilter, setWorkerFilter] = useState('all');
  const [regionFilter, setRegionFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Mock data - would be replaced with API calls
  const pendingRepairs = [
    { id: 'DR-2023-001', dateReported: '2023-06-15', severity: 'High', region: 'North', damageType: 'Structural', reporter: 'John Doe', status: 'Pending', description: 'Major structural damage to building facade' },
    { id: 'DR-2023-003', dateReported: '2023-06-17', severity: 'Critical', region: 'Central', damageType: 'Flooding', reporter: 'Mike Johnson', status: 'Pending', description: 'Severe flooding in basement level' },
    { id: 'DR-2023-005', dateReported: '2023-06-19', severity: 'Medium', region: 'West', damageType: 'Electrical', reporter: 'Lisa Brown', status: 'Pending', description: 'Electrical outage affecting multiple units' },
  ];

  const assignedRepairs = [
    { id: 'DR-2023-002', dateReported: '2023-06-16', severity: 'Medium', region: 'South', damageType: 'Electrical', reporter: 'Jane Smith', status: 'Assigned', assignedTo: 'Robert Chen', assignedDate: '2023-06-17', notes: 'Check all circuit breakers' },
    { id: 'DR-2023-006', dateReported: '2023-06-20', severity: 'High', region: 'East', damageType: 'Plumbing', reporter: 'David Wilson', status: 'In-Progress', assignedTo: 'Maria Garcia', assignedDate: '2023-06-21', notes: 'Bring extra pipe fittings' },
    { id: 'DR-2023-004', dateReported: '2023-06-18', severity: 'Low', region: 'East', damageType: 'Plumbing', reporter: 'Sarah Williams', status: 'Resolved', assignedTo: 'James Wilson', assignedDate: '2023-06-19', completedDate: '2023-06-22', notes: 'Minor repair completed' },
    { id: 'DR-2023-007', dateReported: '2023-06-21', severity: 'Critical', region: 'North', damageType: 'Fire', reporter: 'Emily Davis', status: 'On Hold', assignedTo: 'Robert Chen', assignedDate: '2023-06-22', notes: 'Waiting for safety inspection' },
  ];

  const fieldWorkers = [
    { id: 'FW-001', name: 'Robert Chen', specialization: 'Electrical', activeAssignments: 2, region: 'North', status: 'Available' },
    { id: 'FW-002', name: 'Maria Garcia', specialization: 'Plumbing', activeAssignments: 1, region: 'East', status: 'Busy' },
    { id: 'FW-003', name: 'James Wilson', specialization: 'Structural', activeAssignments: 0, region: 'Central', status: 'Available' },
    { id: 'FW-004', name: 'Sarah Johnson', specialization: 'General', activeAssignments: 3, region: 'South', status: 'Busy' },
    { id: 'FW-005', name: 'Michael Brown', specialization: 'Electrical', activeAssignments: 0, region: 'West', status: 'Available' },
  ];

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleAssignDialogOpen = (repair) => {
    setSelectedRepair(repair);
    setAssignDialogOpen(true);
  };

  const handleAssignDialogClose = () => {
    setAssignDialogOpen(false);
    setSelectedWorker('');
    setAssignmentNotes('');
  };

  const handleAssignRepair = () => {
    // In a real app, this would make an API call to assign the repair
    console.log(`Assigning repair ${selectedRepair.id} to worker ${selectedWorker} with notes: ${assignmentNotes}`);
    handleAssignDialogClose();
    // Would refresh data after API call
  };

  const handleStatusChange = (repairId, newStatus) => {
    // In a real app, this would make an API call to update the status
    console.log(`Updating repair ${repairId} status to ${newStatus}`);
    // Would refresh data after API call
  };

  const getStatusIcon = (status) => {
    switch(status) {
      case 'Pending': return <PendingIcon color="warning" />;
      case 'Assigned': return <AssignmentIndIcon color="info" />;
      case 'In-Progress': return <BuildIcon color="primary" />;
      case 'On Hold': return <PendingIcon color="error" />;
      case 'Resolved': return <CheckCircleIcon color="success" />;
      case 'Rejected': return <CancelIcon color="error" />;
      default: return <PendingIcon color="warning" />;
    }
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'Pending': return theme.palette.warning.main;
      case 'Assigned': return theme.palette.info.main;
      case 'In-Progress': return theme.palette.primary.main;
      case 'On Hold': return theme.palette.error.light;
      case 'Resolved': return theme.palette.success.main;
      case 'Rejected': return theme.palette.error.main;
      default: return theme.palette.warning.main;
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

  const filteredAssignedRepairs = assignedRepairs.filter(repair => {
    const matchesStatus = statusFilter === 'all' || repair.status === statusFilter;
    const matchesWorker = workerFilter === 'all' || repair.assignedTo === workerFilter;
    const matchesRegion = regionFilter === 'all' || repair.region === regionFilter;
    const matchesSearch = searchQuery === '' || 
      repair.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      repair.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      repair.reporter.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (repair.assignedTo && repair.assignedTo.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesStatus && matchesWorker && matchesRegion && matchesSearch;
  });

  return (
    <>
      <Typography variant="h4" gutterBottom sx={{fontWeight: 'semi-bold'}}>
        Repair Management
      </Typography>
      
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
          assignedRepairs={filteredAssignedRepairs}
          fieldWorkers={fieldWorkers}
          statusFilter={statusFilter}
          workerFilter={workerFilter}
          regionFilter={regionFilter}
          searchQuery={searchQuery}
          setStatusFilter={setStatusFilter}
          setWorkerFilter={setWorkerFilter}
          setRegionFilter={setRegionFilter}
          setSearchQuery={setSearchQuery}
          getStatusColor={getStatusColor}
          getStatusIcon={getStatusIcon}
          handleStatusChange={handleStatusChange}
          theme={theme}
        />
      )}
      
      {/* Field Workers Tab */}
      {tabValue === 2 && (
        <FieldWorker 
          fieldWorkers={fieldWorkers}
          theme={theme}
        />
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
    </>
  );
}

export default Repair;
