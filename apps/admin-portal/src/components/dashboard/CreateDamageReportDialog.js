import React from 'react';
import { 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  IconButton, 
  Typography, 
  Box,
  Button, 
  Grid, 
  TextField, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  CircularProgress,
  Alert,
  FormHelperText
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import AssignmentIcon from '@mui/icons-material/Assignment';

const CreateDamageReportDialog = ({ 
  open, 
  onClose, 
  onSubmit,
  selectedAiReport,
  formData,
  onFormChange,
  fieldWorkers,
  onFieldWorkerChange,
  selectedFieldWorker,
  loading,
  error
}) => {
  // Helper function to check if a field is empty
  const isFieldEmpty = (fieldName) => {
    return formData[fieldName] === '';
  };

  // Required fields for form validation
  const requiredFields = ['region', 'location', 'damageType', 'severity', 'priority'];
  
  // Check if form is valid
  const isFormValid = !requiredFields.some(isFieldEmpty);

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        borderBottom: '1px solid #e5e7eb'
      }}>
        <Typography variant="h6">Create Damage Report</Typography>
        <IconButton onClick={onClose} size="small">
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 3 }}>
        {selectedAiReport?.annotatedImageBase64 && (
          <Box sx={{ mb: 3, textAlign: 'center' }}>
            <img
              src={`data:image/jpeg;base64,${selectedAiReport.annotatedImageBase64}`}
              alt="Damage"
              style={{ 
                maxWidth: '100%', 
                maxHeight: '200px',
                objectFit: 'contain',
                borderRadius: '4px',
              }}
            />
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Region"
              name="region"
              value={formData.region}
              onChange={onFormChange}
              fullWidth
              required
              error={isFieldEmpty('region')}
              helperText={isFieldEmpty('region') ? 'Region is required' : ''}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Location"
              name="location"
              value={formData.location}
              onChange={onFormChange}
              fullWidth
              required
              error={isFieldEmpty('location')}
              helperText={isFieldEmpty('location') ? 'Location is required' : ''}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Damage Type"
              name="damageType"
              value={formData.damageType}
              onChange={onFormChange}
              fullWidth
              required
              disabled={!!selectedAiReport}
              error={isFieldEmpty('damageType')}
              helperText={isFieldEmpty('damageType') ? 'Damage type is required' : ''}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth required error={isFieldEmpty('severity')}>
              <InputLabel>Severity</InputLabel>
              <Select
                label="Severity"
                name="severity"
                value={formData.severity}
                onChange={onFormChange}
                disabled={!!selectedAiReport}
              >
                <MenuItem value="LOW">Low</MenuItem>
                <MenuItem value="MEDIUM">Medium</MenuItem>
                <MenuItem value="HIGH">High</MenuItem>
              </Select>
              {isFieldEmpty('severity') && (
                <FormHelperText>Severity is required</FormHelperText>
              )}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Priority (1-10)"
              name="priority"
              value={formData.priority}
              onChange={onFormChange}
              fullWidth
              required
              type="number"
              InputProps={{ inputProps: { min: 1, max: 10 } }}
              disabled={!!selectedAiReport}
              error={isFieldEmpty('priority')}
              helperText={isFieldEmpty('priority') ? 'Priority is required' : ''}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Assign to Field Worker</InputLabel>
              <Select
                label="Assign to Field Worker"
                value={selectedFieldWorker}
                onChange={onFieldWorkerChange}
              >
                <MenuItem value="">
                  <em>None (Unassigned)</em>
                </MenuItem>
                {fieldWorkers.map(worker => (
                  <MenuItem key={worker._id} value={worker._id}>
                    {worker.name} - {worker.specialization}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              label="Description"
              name="description"
              value={formData.description}
              onChange={onFormChange}
              fullWidth
              multiline
              rows={3}
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 2, borderTop: '1px solid #e5e7eb' }}>
        <Button 
          onClick={onClose} 
          color="inherit"
        >
          Cancel
        </Button>
        <Button
          onClick={(e) => {
            // Prevent passing the entire event object to avoid circular references
            e.preventDefault();
            // Only pass the formData to the onSubmit handler
            onSubmit(formData);
          }}
          variant="contained"
          disabled={loading || !isFormValid}
          startIcon={loading ? <CircularProgress size={20} /> : <AssignmentIcon />}
        >
          {loading ? 'Creating...' : 'Create Report'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CreateDamageReportDialog;
