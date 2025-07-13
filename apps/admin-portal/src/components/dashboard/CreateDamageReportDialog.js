import React, { useState } from 'react';
import { useTheme, alpha } from '@mui/material/styles';
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
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { aiServices } from '../../utils/aiServices';

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
  const theme = useTheme();
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(null);

  // Helper function to check if a field is empty
  const isFieldEmpty = (fieldName) => {
    return formData[fieldName] === '';
  };

  // Required fields for form validation
  const requiredFields = ['region', 'location', 'damageType', 'severity', 'priority'];
  
  // Check if form is valid
  const isFormValid = !requiredFields.some(isFieldEmpty);
  
  // Generate AI summary for description
  const handleGenerateAiSummary = async () => {
    // Required fields for AI summary
    const requiredAiFields = ['location', 'damageType', 'severity', 'priority'];
    
    // Check if required fields are filled
    const missingFields = requiredAiFields.filter(isFieldEmpty);
    if (missingFields.length > 0) {
      setAiError(`Please fill in ${missingFields.join(', ')} before generating a summary`);
      return;
    }
    
    setAiLoading(true);
    setAiError(null);
    
    try {
      // Add a little delay to show the "Generating..." message
      const summaryResponse = await aiServices.generateDamageSummary({
        location: formData.location,
        damageType: formData.damageType,
        severity: formData.severity,
        priority: formData.priority
      });
      
      console.log('Summary response received:', summaryResponse);
      
      if (!summaryResponse || !summaryResponse.summary) {
        throw new Error('No summary generated. Please try again.');
      }
      
      // Update the form data with the generated summary
      onFormChange({
        target: {
          name: 'description',
          value: summaryResponse.summary
        }
      });
      
    } catch (error) {
      setAiError(error.message || 'Failed to generate AI summary. Please try again or enter description manually.');
      console.error('Error generating summary:', error);
    } finally {
      setAiLoading(false);
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 1,
          maxHeight: '90vh',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)'
        }
      }}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        borderBottom: `1px solid ${theme.palette.divider}`,
        py: 1.5,
        px: 2.5,
        bgcolor: theme.palette.background.paper
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AssignmentIcon fontSize="small" sx={{ color: theme.palette.text.secondary }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
            Create Damage Report
          </Typography>
          {selectedAiReport && (
            <Typography 
              variant="caption" 
              sx={{ 
                bgcolor: alpha(theme.palette.primary.main, 0.08),
                color: theme.palette.primary.main, 
                px: 1, 
                py: 0.25, 
                borderRadius: 0.5,
                fontWeight: 500,
                fontSize: '0.7rem'
              }}
            >
              AI-Assisted
            </Typography>
          )}
        </Box>
        <IconButton 
          onClick={onClose} 
          size="small"
          edge="end"
          sx={{ color: theme.palette.text.secondary }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 2, px: 2.5 }}>
        {/* AI Detected Image Section */}
        {((selectedAiReport?.annotatedImageBase64 && 
           typeof selectedAiReport.annotatedImageBase64 === 'string' && 
           selectedAiReport.annotatedImageBase64.length > 0) ||
          (selectedAiReport?.annotatedImage && 
           typeof selectedAiReport.annotatedImage === 'string' && 
           selectedAiReport.annotatedImage.length > 0)) && (
          <Box sx={{ 
            mb: 2, 
            p: 1.5, 
            bgcolor: alpha(theme.palette.background.paper, 0.5), 
            borderRadius: 1,
            border: `1px solid ${theme.palette.divider}`
          }}>
            <Typography variant="caption" sx={{ 
              display: 'block', 
              mb: 1, 
              color: theme.palette.text.secondary,
              fontWeight: 500
            }}>
              AI Detected Damage
            </Typography>
            <Box sx={{ textAlign: 'center', mb: 0.5 }}>
              <img
                src={selectedAiReport.annotatedImageBase64 ? 
                  `data:image/jpeg;base64,${selectedAiReport.annotatedImageBase64}` :
                  selectedAiReport.annotatedImage?.startsWith('data:') ? 
                    selectedAiReport.annotatedImage :
                    `data:image/jpeg;base64,${selectedAiReport.annotatedImage}`
                }
                alt="AI Detected Damage"
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '180px',
                  objectFit: 'contain',
                  borderRadius: '4px'
                }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ 
              display: 'block', 
              textAlign: 'center',
              fontSize: '0.7rem'
            }}>
              This image will be attached as the "before" image in your damage report
            </Typography>
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 2, py: 0.5, fontSize: '0.8rem' }}>
            {error}
          </Alert>
        )}

        {/* Form Section */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.5, color: theme.palette.text.primary, fontWeight: 500 }}>
            Report Details
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Region"
                name="region"
                value={formData.region}
                onChange={onFormChange}
                fullWidth
                required
                size="small"
                error={isFieldEmpty('region')}
                helperText={isFieldEmpty('region') ? 'Region is required' : ''}
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': { borderRadius: 1 },
                  '& .MuiInputLabel-root': { fontSize: '0.875rem' },
                  '& .MuiInputBase-input': { fontSize: '0.875rem' }
                }}
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
                size="small"
                error={isFieldEmpty('location')}
                helperText={isFieldEmpty('location') ? 'Location is required' : ''}
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': { borderRadius: 1 },
                  '& .MuiInputLabel-root': { fontSize: '0.875rem' },
                  '& .MuiInputBase-input': { fontSize: '0.875rem' }
                }}
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
                size="small"
                disabled={!!selectedAiReport}
                error={isFieldEmpty('damageType')}
                helperText={isFieldEmpty('damageType') ? 'Damage type is required' : (selectedAiReport ? 'Pre-filled by AI' : '')}
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': { borderRadius: 1 },
                  '& .MuiInputLabel-root': { fontSize: '0.875rem' },
                  '& .MuiInputBase-input': { fontSize: '0.875rem' },
                  '& .MuiInputBase-input:disabled': { 
                    backgroundColor: alpha(theme.palette.action.disabledBackground, 0.5),
                    color: theme.palette.text.disabled
                  },
                  '& .MuiFormHelperText-root': {
                    fontSize: '0.7rem'
                  }
                }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth required error={isFieldEmpty('severity')} size="small">
                <InputLabel sx={{ fontSize: '0.875rem' }}>Severity</InputLabel>
                <Select
                  label="Severity"
                  name="severity"
                  value={formData.severity}
                  onChange={onFormChange}
                  disabled={!!selectedAiReport}
                  sx={{ 
                    borderRadius: 1,
                    fontSize: '0.875rem',
                    '& .MuiSelect-select:disabled': { 
                      backgroundColor: alpha(theme.palette.action.disabledBackground, 0.5)
                    }
                  }}
                >
                  <MenuItem value="LOW" sx={{ fontSize: '0.875rem' }}>Low</MenuItem>
                  <MenuItem value="MEDIUM" sx={{ fontSize: '0.875rem' }}>Medium</MenuItem>
                  <MenuItem value="HIGH" sx={{ fontSize: '0.875rem' }}>High</MenuItem>
                </Select>
                {isFieldEmpty('severity') && (
                  <FormHelperText sx={{ fontSize: '0.7rem' }}>Severity is required</FormHelperText>
                )}
                {selectedAiReport && !isFieldEmpty('severity') && (
                  <FormHelperText sx={{ color: theme.palette.text.secondary, fontSize: '0.7rem' }}>Pre-filled by AI</FormHelperText>
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
                size="small"
                type="number"
                InputProps={{ inputProps: { min: 1, max: 10 } }}
                disabled={!!selectedAiReport}
                error={isFieldEmpty('priority')}
                helperText={isFieldEmpty('priority') ? 'Priority is required' : (selectedAiReport ? 'Pre-filled by AI' : '')}
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': { borderRadius: 1 },
                  '& .MuiInputLabel-root': { fontSize: '0.875rem' },
                  '& .MuiInputBase-input': { fontSize: '0.875rem' },
                  '& .MuiInputBase-input:disabled': { 
                    backgroundColor: alpha(theme.palette.action.disabledBackground, 0.5),
                    color: theme.palette.text.disabled
                  },
                  '& .MuiFormHelperText-root': {
                    fontSize: '0.7rem'
                  }
                }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: '0.875rem' }}>Assign to Field Worker</InputLabel>
                <Select
                  label="Assign to Field Worker"
                  value={selectedFieldWorker}
                  onChange={onFieldWorkerChange}
                  sx={{ 
                    borderRadius: 1,
                    fontSize: '0.875rem' 
                  }}
                >
                  <MenuItem value="" sx={{ fontSize: '0.875rem' }}>
                    <em>None (Unassigned)</em>
                  </MenuItem>
                  {fieldWorkers.map(worker => (
                    <MenuItem key={worker._id} value={worker._id} sx={{ fontSize: '0.875rem' }}>
                      {worker.name} - {worker.specialization}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Box>

        {/* Description Section */}
        <Box sx={{ mt: 1 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, color: theme.palette.text.primary, fontWeight: 500, display: 'flex', alignItems: 'center' }}>
            Description
            {aiLoading && <CircularProgress size={14} sx={{ ml: 1 }} />}
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            {aiLoading ? (
              <TextField
                value="Generating professional description..."
                disabled
                fullWidth
                multiline
                rows={3}
                size="small"
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': { borderRadius: 1 },
                  '& .MuiInputBase-input': { 
                    fontStyle: 'italic',
                    fontSize: '0.875rem'
                  }
                }}
              />
            ) : (
              <TextField
                name="description"
                value={formData.description}
                onChange={onFormChange}
                fullWidth
                multiline
                rows={3}
                size="small"
                placeholder="Enter a detailed description of the damage or use the AI enhancement button below..."
                variant="outlined"
                sx={{ 
                  '& .MuiOutlinedInput-root': { borderRadius: 1 },
                  '& .MuiInputBase-input': { fontSize: '0.875rem' }
                }}
              />
            )}
            
            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                onClick={handleGenerateAiSummary}
                disabled={aiLoading || loading || !isFormValid}
                variant="outlined"
                color="primary"
                startIcon={<AutoAwesomeIcon sx={{ fontSize: '0.8rem' }} />}
                size="small"
                sx={{ 
                  borderRadius: 1,
                  textTransform: 'none',
                  fontWeight: 500,
                  fontSize: '0.75rem',
                  px: 1.5
                }}
              >
                Enhance with AI
              </Button>
            </Box>
            
            {aiError && (
              <Alert severity="error" sx={{ borderRadius: 1, py: 0.5, fontSize: '0.8rem' }}>
                {aiError}
              </Alert>
            )}
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ 
        px: 2.5, 
        py: 1.5, 
        borderTop: `1px solid ${theme.palette.divider}`,
        bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.paper, 0.6) : alpha(theme.palette.grey[50], 0.8),
        gap: 1
      }}>
        <Button 
          onClick={onClose} 
          color="inherit"
          size="small"
          sx={{ 
            borderRadius: 1,
            textTransform: 'none',
            fontWeight: 500,
            px: 2,
            fontSize: '0.8rem'
          }}
        >
          Cancel
        </Button>
        <Button
          onClick={(e) => {
            e.preventDefault();
            onSubmit(formData);
          }}
          variant="contained"
          disabled={loading || !isFormValid}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : null}
          size="small"
          sx={{ 
            borderRadius: 1,
            textTransform: 'none',
            fontWeight: 500,
            px: 2,
            minWidth: 100,
            fontSize: '0.8rem'
          }}
        >
          {loading ? 'Creating...' : 'Create Report'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CreateDamageReportDialog;
