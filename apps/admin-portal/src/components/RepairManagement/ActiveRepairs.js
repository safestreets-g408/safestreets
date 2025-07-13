import React, { useState } from 'react';
import {
  Box, Typography, Grid, Card, CardContent, CardActions,
  Divider, IconButton, Tooltip,
  TextField, FormControl, InputLabel, Select, MenuItem,
  useTheme, Paper
} from '@mui/material';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import RefreshIcon from '@mui/icons-material/Refresh';
import CircularProgress from '@mui/material/CircularProgress';
import SearchIcon from '@mui/icons-material/Search';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import AssignmentIndIcon from '@mui/icons-material/AssignmentInd';
import BuildIcon from '@mui/icons-material/Build';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';

function ActiveRepairs({ assignedRepairs = [], fieldWorkers = [], onStatusChange }) {
  const theme = useTheme();
  const [statusFilter, setStatusFilter] = useState('all');
  const [workerFilter, setWorkerFilter] = useState('all');
  const [regionFilter, setRegionFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [changingStatusIds, setChangingStatusIds] = useState([]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'Pending': return <HourglassEmptyIcon color="warning" />;
      case 'Assigned': return <AssignmentIndIcon color="info" />;
      case 'In-Progress': return <BuildIcon color="primary" />;
      case 'On Hold': return <HourglassEmptyIcon color="error" />;
      case 'Resolved': return <CheckCircleIcon color="success" />;
      case 'Rejected': return <CancelIcon color="error" />;
      default: return <HourglassEmptyIcon color="warning" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Pending': return theme.palette.warning.main;
      case 'Assigned': return theme.palette.info.main;
      case 'In-Progress': return theme.palette.primary.main;
      case 'On Hold': return theme.palette.error.light;
      case 'Resolved': return theme.palette.success.main;
      case 'Rejected': return theme.palette.error.main;
      default: return theme.palette.warning.main;
    }
  };

  const filteredAssignedRepairs = assignedRepairs.filter(repair => {
    const matchesStatus = statusFilter === 'all' || repair.status === statusFilter;
    const assignedWorkerName = repair.assignedTo ? repair.assignedTo.name : null;
    const matchesWorker = workerFilter === 'all' || assignedWorkerName === workerFilter;
    const matchesRegion = regionFilter === 'all' || repair.region === regionFilter;
    const matchesSearch = searchQuery === '' ||
      (repair.id && repair.id.toString().toLowerCase().includes(searchQuery.toLowerCase())) ||
      (repair.description && repair.description.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (repair.reporter && repair.reporter.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (assignedWorkerName && assignedWorkerName.toLowerCase().includes(searchQuery.toLowerCase()));

    return matchesStatus && matchesWorker && matchesRegion && matchesSearch;
  });

  return (
    <Paper sx={{ p: 2, borderRadius: 1, boxShadow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 500 }}>
          Active Repair Tasks
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            placeholder="Search"
            size="small"
            variant="outlined"
            InputProps={{
              startAdornment: (
                <SearchIcon fontSize="small" sx={{ mr: 0.5, color: 'text.secondary' }} />
              ),
            }}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            sx={{ width: 180 }}
          />
          <Tooltip title="Refresh">
            <IconButton size="small">
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Box sx={{ mb: 2, display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        <FormControl size="small" sx={{ minWidth: 140 }} variant="outlined">
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

        <FormControl size="small" sx={{ minWidth: 140 }} variant="outlined">
          <InputLabel>Personnel</InputLabel>
          <Select
            value={workerFilter}
            label="Personnel"
            onChange={(e) => setWorkerFilter(e.target.value)}
          >
            <MenuItem value="all">All Personnel</MenuItem>
            {fieldWorkers.map(worker => (
              <MenuItem key={worker.id} value={worker.name}>{worker.name}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 140 }} variant="outlined">
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
        <Box sx={{ textAlign: 'center', py: 2 }}>
          <Typography variant="body1" sx={{ mb: 1, fontWeight: 500 }}>No records found</Typography>
          <Typography variant="body2" color="text.secondary">
            Adjust filters to view matching records
          </Typography>
        </Box>
      ) : (
        <Grid container spacing={1.5}>
          {filteredAssignedRepairs.map((repair) => (
            <Grid item xs={12} md={6} lg={4} key={repair._id || repair.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  borderLeft: `3px solid ${getStatusColor(repair.status)}`,
                  boxShadow: 1
                }}
              >
                <CardContent sx={{ flexGrow: 1, pb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                      Report #{repair.reportId || repair.id}
                    </Typography>
                    <Typography 
                      variant="caption" 
                      sx={{
                        color: getStatusColor(repair.status),
                        fontWeight: 500
                      }}
                    >
                      {repair.status}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mr: 0.5 }}>
                      Personnel:
                    </Typography>
                    <Typography variant="body2">
                      {repair.assignedTo ? repair.assignedTo.name : 'Unassigned'}
                    </Typography>
                  </Box>

                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <LocationOnIcon fontSize="small" sx={{ verticalAlign: 'text-top', mr: 0.5, fontSize: '0.9rem' }} />
                    {repair.region} Region
                  </Typography>

                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <AccessTimeIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                    Assigned: {repair.assignedDate ? new Date(repair.assignedDate).toLocaleDateString() : 'N/A'}
                  </Typography>

                  <Divider sx={{ my: 0.5 }} />

                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    <Box component="span" sx={{ color: 'text.secondary', width: '4.5rem', display: 'inline-block' }}>Type:</Box> 
                    {repair.damageType}
                  </Typography>

                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    <Box component="span" sx={{ color: 'text.secondary', width: '4.5rem', display: 'inline-block' }}>Severity:</Box>
                    {repair.severity}
                  </Typography>

                  {repair.notes && (
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                      <Box component="span" sx={{ color: 'text.secondary', width: '4.5rem', display: 'inline-block' }}>Notes:</Box>
                      {repair.notes?.length > 50 ? repair.notes.substring(0, 50) + '...' : repair.notes}
                    </Typography>
                  )}
                </CardContent>
                <CardActions sx={{ pt: 0, pb: 1, px: 2 }}>
                  <FormControl size="small" sx={{ minWidth: 130, position: 'relative' }} variant="outlined">
                    <InputLabel>Status</InputLabel>
                    {changingStatusIds.includes(repair._id || repair.id) && (
                      <CircularProgress
                        size={16}
                        sx={{
                          position: 'absolute',
                          top: '50%',
                          left: '50%',
                          marginTop: '-8px',
                          marginLeft: '-8px',
                          zIndex: 1,
                        }}
                      />
                    )}
                    <Select
                      label="Status"
                      value={repair.status || ''}
                      onChange={(e) => {
                        const newStatus = e.target.value;
                        const repairId = repair._id || repair.id;
                        
                        // Show visual feedback that something is happening
                        setChangingStatusIds(prev => [...prev, repairId]);
                        
                        // Call the status change handler
                        if (onStatusChange) {
                          onStatusChange(repairId, newStatus)
                            .finally(() => {
                              // Remove from loading state whether successful or not
                              setChangingStatusIds(prev => prev.filter(id => id !== repairId));
                            });
                        }
                      }}
                      disabled={changingStatusIds.includes(repair._id || repair.id)}
                      MenuProps={{
                        PaperProps: {
                          elevation: 3,
                          sx: { maxHeight: 200 }
                        }
                      }}
                    >
                      <MenuItem value="Assigned">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <AssignmentIndIcon fontSize="small" sx={{ mr: 1, color: theme.palette.info.main }} />
                          Assigned
                        </Box>
                      </MenuItem>
                      <MenuItem value="In-Progress">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <BuildIcon fontSize="small" sx={{ mr: 1, color: theme.palette.primary.main }} />
                          In Progress
                        </Box>
                      </MenuItem>
                      <MenuItem value="On Hold">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <HourglassEmptyIcon fontSize="small" sx={{ mr: 1, color: theme.palette.error.light }} />
                          On Hold
                        </Box>
                      </MenuItem>
                      <MenuItem value="Resolved">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <CheckCircleIcon fontSize="small" sx={{ mr: 1, color: theme.palette.success.main }} />
                          Resolved
                        </Box>
                      </MenuItem>
                      <MenuItem value="Rejected">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <CancelIcon fontSize="small" sx={{ mr: 1, color: theme.palette.error.main }} />
                          Rejected
                        </Box>
                      </MenuItem>
                    </Select>
                  </FormControl>
                  <Tooltip title="View Details">
                    <IconButton color="primary">
                      {/* More options icon */}
                    </IconButton>
                  </Tooltip>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Paper>
  );
}

export default ActiveRepairs;
