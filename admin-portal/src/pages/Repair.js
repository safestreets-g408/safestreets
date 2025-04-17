import React, { useState } from 'react';
import { 
  Box, Typography, Paper, Grid, TextField, Button, 
  Select, MenuItem, FormControl, InputLabel, Divider,
  Card, CardContent, CardActions, Chip, Avatar, 
  IconButton, Tooltip, Tab, Tabs, Badge, 
  Dialog, DialogTitle, DialogContent, DialogActions,
  FormHelperText, useTheme, alpha
} from '@mui/material';
import { 
  Build as BuildIcon,
  Assignment as AssignmentIcon,
  AssignmentInd as AssignmentIndIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Pending as PendingIcon,
  Search as SearchIcon,
  Person as PersonIcon,
  LocationOn as LocationOnIcon,
  AccessTime as AccessTimeIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon
} from '@mui/icons-material';

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
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Pending Damage Reports
                </Typography>
                <Tooltip title="Refresh">
                  <IconButton>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Box>
              <Divider sx={{ mb: 2 }} />
              
              {pendingRepairs.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 3 }}>
                  <CheckCircleIcon color="success" sx={{ fontSize: 48, mb: 2 }} />
                  <Typography variant="h6">No pending repairs</Typography>
                  <Typography variant="body2" color="textSecondary">
                    All damage reports have been assigned
                  </Typography>
                </Box>
              ) : (
                <Grid container spacing={2}>
                  {pendingRepairs.map((repair) => (
                    <Grid item xs={12} md={6} lg={4} key={repair.id}>
                      <Card 
                        sx={{ 
                          height: '100%', 
                          display: 'flex', 
                          flexDirection: 'column',
                          borderLeft: `4px solid ${getSeverityColor(repair.severity)}`,
                          transition: 'transform 0.2s',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: 4
                          }
                        }}
                      >
                        <CardContent sx={{ flexGrow: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="subtitle1" fontWeight="bold">
                              {repair.id}
                            </Typography>
                            <Chip 
                              size="small" 
                              label={repair.severity} 
                              sx={{ 
                                bgcolor: alpha(getSeverityColor(repair.severity), 0.1),
                                color: getSeverityColor(repair.severity),
                                fontWeight: 'bold'
                              }} 
                            />
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            <LocationOnIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                            {repair.region} Region
                          </Typography>
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            <AccessTimeIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                            Reported: {new Date(repair.dateReported).toLocaleDateString()}
                          </Typography>
                          
                          <Divider sx={{ my: 1 }} />
                          
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            <strong>Type:</strong> {repair.damageType}
                          </Typography>
                          
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            <strong>Reporter:</strong> {repair.reporter}
                          </Typography>
                          
                          <Typography variant="body2">
                            <strong>Description:</strong>
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            {repair.description}
                          </Typography>
                        </CardContent>
                        <CardActions>
                          <Button 
                            variant="contained" 
                            startIcon={<AssignmentIndIcon />}
                            fullWidth
                            onClick={() => handleAssignDialogOpen(repair)}
                          >
                            Assign Repair
                          </Button>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}
      
      {/* Active Repairs Tab */}
      {tabValue === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Active Repair Tasks
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    placeholder="Search repairs..."
                    size="small"
                    InputProps={{
                      startAdornment: <SearchIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />,
                    }}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    sx={{ width: 200 }}
                  />
                  <Tooltip title="Refresh">
                    <IconButton>
                      <RefreshIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              
              <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <FormControl size="small" sx={{ minWidth: 150 }}>
                  <InputLabel>Status</InputLabel>
                  <Select
                    value={statusFilter}
                    label="Status"
                    onChange={(e) => setStatusFilter(e.target.value)}
                  >
                    <MenuItem value="all">All Statuses</MenuItem>
                    <MenuItem value="Assigned">Assigned</MenuItem>
                    <MenuItem value="In-Progress">In Progress</MenuItem>
                    <MenuItem value="On Hold">On Hold</MenuItem>
                    <MenuItem value="Resolved">Resolved</MenuItem>
                    <MenuItem value="Rejected">Rejected</MenuItem>
                  </Select>
                </FormControl>
                
                <FormControl size="small" sx={{ minWidth: 150 }}>
                  <InputLabel>Field Worker</InputLabel>
                  <Select
                    value={workerFilter}
                    label="Field Worker"
                    onChange={(e) => setWorkerFilter(e.target.value)}
                  >
                    <MenuItem value="all">All Workers</MenuItem>
                    {fieldWorkers.map(worker => (
                      <MenuItem key={worker.id} value={worker.name}>{worker.name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl size="small" sx={{ minWidth: 150 }}>
                  <InputLabel>Region</InputLabel>
                  <Select
                    value={regionFilter}
                    label="Region"
                    onChange={(e) => setRegionFilter(e.target.value)}
                  >
                    <MenuItem value="all">All Regions</MenuItem>
                    <MenuItem value="North">North</MenuItem>
                    <MenuItem value="South">South</MenuItem>
                    <MenuItem value="East">East</MenuItem>
                    <MenuItem value="West">West</MenuItem>
                    <MenuItem value="Central">Central</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              
              <Divider sx={{ mb: 2 }} />
              
              {filteredAssignedRepairs.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 3 }}>
                  <SearchIcon color="action" sx={{ fontSize: 48, mb: 2 }} />
                  <Typography variant="h6">No matching repairs found</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Try adjusting your search or filter criteria
                  </Typography>
                </Box>
              ) : (
                <Grid container spacing={2}>
                  {filteredAssignedRepairs.map((repair) => (
                    <Grid item xs={12} md={6} lg={4} key={repair.id}>
                      <Card 
                        sx={{ 
                          height: '100%', 
                          display: 'flex', 
                          flexDirection: 'column',
                          borderLeft: `4px solid ${getStatusColor(repair.status)}`,
                          transition: 'transform 0.2s',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: 4
                          }
                        }}
                      >
                        <CardContent sx={{ flexGrow: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="subtitle1" fontWeight="bold">
                              {repair.id}
                            </Typography>
                            <Chip 
                              size="small" 
                              icon={getStatusIcon(repair.status)}
                              label={repair.status} 
                              sx={{ 
                                bgcolor: alpha(getStatusColor(repair.status), 0.1),
                                color: getStatusColor(repair.status),
                                fontWeight: 'bold'
                              }} 
                            />
                          </Box>
                          
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Avatar sx={{ width: 24, height: 24, mr: 1, bgcolor: theme.palette.primary.main }}>
                              <PersonIcon sx={{ fontSize: 16 }} />
                            </Avatar>
                            <Typography variant="body2">
                              {repair.assignedTo}
                            </Typography>
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            <LocationOnIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                            {repair.region} Region
                          </Typography>
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            <AccessTimeIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                            Assigned: {new Date(repair.assignedDate).toLocaleDateString()}
                          </Typography>
                          
                          <Divider sx={{ my: 1 }} />
                          
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            <strong>Type:</strong> {repair.damageType}
                          </Typography>
                          
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            <strong>Severity:</strong> {repair.severity}
                          </Typography>
                          
                          <Typography variant="body2">
                            <strong>Notes:</strong>
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            {repair.notes}
                          </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'space-between' }}>
                          <FormControl size="small" sx={{ minWidth: 120 }}>
                            <InputLabel>Update Status</InputLabel>
                            <Select
                              label="Update Status"
                              defaultValue=""
                              onChange={(e) => handleStatusChange(repair.id, e.target.value)}
                            >
                              <MenuItem value="Assigned">Assigned</MenuItem>
                              <MenuItem value="In-Progress">In Progress</MenuItem>
                              <MenuItem value="On Hold">On Hold</MenuItem>
                              <MenuItem value="Resolved">Resolved</MenuItem>
                              <MenuItem value="Rejected">Rejected</MenuItem>
                            </Select>
                          </FormControl>
                          <Tooltip title="View Details">
                            <IconButton color="primary">
                              <MoreVertIcon />
                            </IconButton>
                          </Tooltip>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}
      
      {/* Field Workers Tab */}
      {tabValue === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Field Worker Management
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button 
                    variant="contained" 
                    startIcon={<AddIcon />}
                    size="small"
                  >
                    Add Worker
                  </Button>
                  <Tooltip title="Refresh">
                    <IconButton>
                      <RefreshIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <Divider sx={{ mb: 2 }} />
              
              <Grid container spacing={2}>
                {fieldWorkers.map((worker) => (
                  <Grid item xs={12} sm={6} md={4} key={worker.id}>
                    <Card 
                      sx={{ 
                        height: '100%', 
                        display: 'flex', 
                        flexDirection: 'column',
                        transition: 'transform 0.2s',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 4
                        }
                      }}
                    >
                      <CardContent sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Avatar 
                            sx={{ 
                              bgcolor: theme.palette.primary.main, 
                              width: 56, 
                              height: 56,
                              mr: 2
                            }}
                          >
                            {worker.name.split(' ').map(n => n[0]).join('')}
                          </Avatar>
                          <Box>
                            <Typography variant="h6">{worker.name}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              ID: {worker.id}
                            </Typography>
                            <Chip 
                              size="small" 
                              label={worker.status} 
                              color={worker.status === 'Available' ? 'success' : 'warning'}
                              sx={{ mt: 0.5 }}
                            />
                          </Box>
                        </Box>
                        
                        <Divider sx={{ my: 1 }} />
                        
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Specialization:</strong> {worker.specialization}
                        </Typography>
                        
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Region:</strong> {worker.region}
                        </Typography>
                        
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Active Assignments:</strong> {worker.activeAssignments}
                        </Typography>
                      </CardContent>
                      <CardActions sx={{ justifyContent: 'space-between' }}>
                        <Button 
                          size="small" 
                          startIcon={<AssignmentIcon />}
                          disabled={worker.status !== 'Available'}
                        >
                          View Assignments
                        </Button>
                        <Box>
                          <Tooltip title="Edit">
                            <IconButton size="small">
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton size="small" color="error">
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        </Grid>
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
