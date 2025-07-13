import React, { useState } from 'react';
import {
  Box, Typography, Grid, Card, CardContent, CardActions,
  Button, Divider, IconButton, Tooltip,
  Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, FormControl, InputLabel, Select, MenuItem,
  FormHelperText, useTheme, Paper
} from '@mui/material';
import {
  AssignmentInd,
  LocationOn,
  AccessTime,
  Refresh
} from '@mui/icons-material';

function PendingAssignments({ pendingRepairs = [], fieldWorkers = [], onAssignRepair = () => {} }) {
  const theme = useTheme();
  const [assignDialogOpen, setAssignDialogOpen] = useState(false);
  const [selectedRepair, setSelectedRepair] = useState(null);
  const [selectedWorker, setSelectedWorker] = useState('');
  const [assignmentNotes, setAssignmentNotes] = useState('');

  const handleAssignDialogOpen = (repair) => {
    setSelectedRepair(repair);
    setAssignDialogOpen(true);
  };

  const handleAssignDialogClose = () => {
    setAssignDialogOpen(false);
    setSelectedWorker('');
    setAssignmentNotes('');
    setSelectedRepair(null);
  };

  const handleAssignRepair = () => {
    if (!selectedRepair || !selectedWorker) return;
    
    // Call the parent handler with the correct parameters
    onAssignRepair(selectedRepair._id || selectedRepair.id, selectedWorker, assignmentNotes);
    handleAssignDialogClose();
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'Low': return theme.palette.info.main;
      case 'Medium': return theme.palette.warning.main;
      case 'High': return theme.palette.error.main;
      case 'Critical': return theme.palette.secondary.main;
      default: return theme.palette.grey[500];
    }
  };

  return (
    <Paper sx={{ p: 2, borderRadius: 1, boxShadow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
        <Typography variant="h6" sx={{ fontWeight: 500 }}>
          Pending Reports
        </Typography>
        <Tooltip title="Refresh">
          <IconButton size="small">
            <Refresh fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      <Divider sx={{ mb: 1.5 }} />

      {pendingRepairs.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 2 }}>
          <Typography variant="body1" sx={{ mb: 1, fontWeight: 500 }}>No pending reports</Typography>
          <Typography variant="body2" color="text.secondary">
            All damage reports have been assigned
          </Typography>
        </Box>
      ) : (
        <Grid container spacing={1.5}>
          {pendingRepairs.map((repair) => (
            <Grid item xs={12} md={6} lg={4} key={repair._id || repair.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  borderLeft: `3px solid ${getSeverityColor(repair.severity)}`,
                  boxShadow: 1
                }}
              >
                <CardContent sx={{ flexGrow: 1, pb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                      Report #{repair.reportId || repair._id || repair.id}
                    </Typography>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        fontWeight: 500, 
                        color: getSeverityColor(repair.severity) 
                      }}
                    >
                      {repair.severity}
                    </Typography>
                  </Box>

                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <LocationOn fontSize="small" sx={{ verticalAlign: 'text-top', mr: 0.5, fontSize: '0.9rem' }} />
                    {repair.region} Region
                  </Typography>

                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <AccessTime fontSize="small" sx={{ verticalAlign: 'text-top', mr: 0.5, fontSize: '0.9rem' }} />
                    {repair.dateReported ? new Date(repair.dateReported).toLocaleDateString() : '-'}
                  </Typography>

                  <Divider sx={{ my: 0.5 }} />

                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    <Box component="span" sx={{ color: 'text.secondary', width: '4.5rem', display: 'inline-block' }}>Type:</Box> 
                    {repair.damageType}
                  </Typography>

                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    <Box component="span" sx={{ color: 'text.secondary', width: '4.5rem', display: 'inline-block' }}>Reporter:</Box> 
                    {repair.reporter}
                  </Typography>

                  {repair.description && (
                    <Typography variant="body2">
                      <Box component="span" sx={{ color: 'text.secondary', width: '4.5rem', display: 'inline-block', verticalAlign: 'top' }}>Details:</Box>
                      <Box component="span" sx={{ display: 'inline-block', maxWidth: 'calc(100% - 4.5rem)' }}>
                        {repair.description.length > 60 
                          ? `${repair.description.substring(0, 60)}...` 
                          : repair.description}
                      </Box>
                    </Typography>
                  )}
                </CardContent>
                <CardActions sx={{ pt: 0, pb: 1, px: 2 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<AssignmentInd sx={{ fontSize: '1rem' }} />}
                    fullWidth
                    onClick={() => handleAssignDialogOpen(repair)}
                  >
                    Assign
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Assignment Dialog */}
      <Dialog open={assignDialogOpen} onClose={handleAssignDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>Assign Report</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2, mt: 0.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
              Report #{selectedRepair?.id}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {selectedRepair?.description && selectedRepair.description.length > 100
                ? selectedRepair.description.substring(0, 100) + '...'
                : selectedRepair?.description}
            </Typography>
            {selectedRepair && (
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Typography variant="caption" color="text.secondary">
                  Type: {selectedRepair.damageType}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Severity: {selectedRepair.severity}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Region: {selectedRepair.region}
                </Typography>
              </Box>
            )}
          </Box>

          <FormControl fullWidth sx={{ mb: 2 }} variant="outlined" size="small">
            <InputLabel>Assign to Personnel</InputLabel>
            <Select
              value={selectedWorker}
              label="Assign to Personnel"
              onChange={(e) => {
                console.log('Selected worker:', e.target.value);
                setSelectedWorker(e.target.value);
              }}
            >
              {fieldWorkers.map(worker => (
                <MenuItem
                  key={worker.id}
                  value={worker.id}
                  disabled={worker.status === 'Busy'}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box sx={{ 
                      width: 10, 
                      height: 10, 
                      borderRadius: '50%',
                      bgcolor: worker.status === 'Available' ? 'success.main' : 'warning.main',
                      mr: 1
                    }} />
                    <Box>
                      {worker.name}
                      <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.75rem' }}>
                        {worker.specialization} - {worker.region} - {worker.activeAssignments ?? 0} active
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
            <FormHelperText>Select available personnel to assign this task</FormHelperText>
          </FormControl>

          <TextField
            label="Assignment Notes"
            multiline
            rows={3}
            fullWidth
            size="small"
            variant="outlined"
            value={assignmentNotes}
            onChange={(e) => setAssignmentNotes(e.target.value)}
            placeholder="Add any specific instructions for the field worker"
            sx={{ fontSize: '0.875rem' }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleAssignDialogClose} size="small">Cancel</Button>
          <Button
            variant="outlined"
            size="small" 
            onClick={handleAssignRepair}
            disabled={!selectedWorker}
          >
            Assign Task
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
}

export default PendingAssignments;
