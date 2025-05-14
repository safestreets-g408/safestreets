import React, { useState } from 'react';
import {
  Box, Typography, Grid, Card, CardContent, CardActions,
  Button, Chip, Divider, IconButton, Tooltip, Avatar,
  Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, useTheme, Paper
} from '@mui/material';
import {
  Assignment as AssignmentIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Person as PersonIcon
} from '@mui/icons-material';

function FieldWorker({ fieldWorkers = [], onAddWorker = () => {}, onEditWorker = () => {}, onDeleteWorker = () => {} }) {
  const theme = useTheme();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [selectedWorker, setSelectedWorker] = useState(null);
  const [workerData, setWorkerData] = useState({
    name: '',
    specialization: '',
    region: '',
    status: 'Available'
  });

  const resetWorkerData = () => ({
    name: '',
    specialization: '',
    region: '',
    status: 'Available'
  });

  const handleDialogOpen = (mode, worker = null) => {
    setEditMode(mode === 'edit');
    setSelectedWorker(worker);

    if (mode === 'edit' && worker) {
      setWorkerData({
        name: worker.name || '',
        specialization: worker.specialization || '',
        region: worker.region || '',
        status: worker.status || 'Available'
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
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setWorkerData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = () => {
    if (!workerData.name || !workerData.specialization || !workerData.region) return;
    if (editMode && selectedWorker) {
      onEditWorker(selectedWorker.id, workerData);
    } else {
      onAddWorker(workerData);
    }
    handleDialogClose();
  };

  const handleDeleteWorker = (workerId) => {
    if (window.confirm('Are you sure you want to delete this worker?')) {
      onDeleteWorker(workerId);
    }
  };

  // Optionally, add a refresh handler if needed
  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <Paper sx={{ p: 2, borderRadius: 2, boxShadow: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Field Worker Management
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<PersonIcon />}
            size="small"
            onClick={() => handleDialogOpen('add')}
          >
            Add Worker
          </Button>
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      <Divider sx={{ mb: 2 }} />

      <Grid container spacing={2}>
        {fieldWorkers.length === 0 ? (
          <Grid item xs={12}>
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="text.secondary">
                No field workers found.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Click "Add Worker" to create a new field worker.
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
                      {worker.name
                        ? worker.name.split(' ').map(n => n[0]).join('')
                        : '?'}
                    </Avatar>
                    <Box>
                      <Typography variant="h6">{worker.name || 'Unnamed'}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        ID: {worker.id}
                      </Typography>
                      <Chip
                        size="small"
                        label={worker.status}
                        color={worker.status === 'Available' ? 'success' : worker.status === 'Busy' ? 'warning' : 'default'}
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Box>

                  <Divider sx={{ my: 1 }} />

                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Specialization:</strong> {worker.specialization || '-'}
                  </Typography>

                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Region:</strong> {worker.region || '-'}
                  </Typography>

                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Active Assignments:</strong> {worker.activeAssignments ?? 0}
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
                      <IconButton size="small" onClick={() => handleDialogOpen('edit', worker)}>
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteWorker(worker.id)}
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
        <DialogTitle>{editMode ? 'Edit Field Worker' : 'Add New Field Worker'}</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Full Name"
              name="name"
              fullWidth
              value={workerData.name}
              onChange={handleInputChange}
              required
              autoFocus
            />

            <TextField
              label="Specialization"
              name="specialization"
              fullWidth
              value={workerData.specialization}
              onChange={handleInputChange}
              placeholder="e.g., Electrical, Plumbing, Structural"
              required
            />

            <TextField
              label="Region"
              name="region"
              fullWidth
              value={workerData.region}
              onChange={handleInputChange}
              placeholder="e.g., North, South, East, West, Central"
              required
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
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={!workerData.name || !workerData.specialization || !workerData.region}
          >
            {editMode ? 'Save Changes' : 'Add Worker'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
}

export default FieldWorker;
