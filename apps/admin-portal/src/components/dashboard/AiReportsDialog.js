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
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Stack,
  Chip
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import AssignmentIcon from '@mui/icons-material/Assignment';

const AiReportsDialog = ({ 
  open, 
  onClose, 
  reports = [], 
  loading, 
  error,
  onSelectReport 
}) => {
  // Helper function to get color based on severity
  const getSeverityColor = (severity) => {
    switch(severity?.toUpperCase()) {
      case 'HIGH':
        return 'error';
      case 'MEDIUM':
        return 'warning';
      case 'LOW':
        return 'success';
      default:
        return 'primary';
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        borderBottom: '1px solid #e5e7eb'
      }}>
        <Typography variant="h6">AI Generated Reports</Typography>
        <IconButton onClick={onClose} size="small">
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 3 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ my: 2 }}>
            {error}
          </Alert>
        ) : reports.length === 0 ? (
          <Alert severity="info" sx={{ my: 2 }}>
            No AI reports found. Upload images in the AI Analysis section to generate reports.
          </Alert>
        ) : (
          <Grid container spacing={3}>
            {reports.map((report) => (
              <Grid item xs={12} sm={6} md={4} key={report._id}>
                <Card 
                  sx={{ 
                    borderRadius: 2,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                    transition: 'transform 0.2s ease-in-out',
                    opacity: report.damageReportGenerated ? 0.7 : 1,
                    border: report.damageReportGenerated ? '2px solid #4caf50' : 'none',
                    '&:hover': {
                      transform: report.damageReportGenerated ? 'none' : 'translateY(-4px)',
                      boxShadow: report.damageReportGenerated 
                        ? '0 4px 12px rgba(0, 0, 0, 0.05)' 
                        : '0 12px 24px rgba(0, 0, 0, 0.1)',
                    }
                  }}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="h6" gutterBottom sx={{ mb: 0 }}>
                        {report.damageType || 'Damage Report'}
                      </Typography>
                      {report.damageReportGenerated && (
                        <Chip 
                          label="Used" 
                          size="small" 
                          color="success"
                          sx={{ fontWeight: 'bold' }}
                        />
                      )}
                    </Box>
                    
                    <Stack spacing={1} sx={{ mb: 2 }}>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Severity:
                        </Typography>
                        <Chip 
                          label={report.severity} 
                          size="small"
                          color={getSeverityColor(report.severity)}
                        />
                      </Stack>
                      
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Priority:
                        </Typography>
                        <Typography variant="body2" fontWeight={500}>
                          {report.priority}/10
                        </Typography>
                      </Stack>
                      
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Date:
                        </Typography>
                        <Typography variant="body2">
                          {new Date(report.createdAt).toLocaleDateString()}
                        </Typography>
                      </Stack>
                    </Stack>
                    
                    <Box sx={{ mt: 2 }}>
                      {report.annotatedImageBase64 && (
                        <Box sx={{ mb: 2, position: 'relative', overflow: 'hidden', borderRadius: 1 }}>
                          <img
                            src={`data:image/jpeg;base64,${report.annotatedImageBase64}`}
                            alt="Damage"
                            style={{ 
                              width: '100%', 
                              height: '150px',
                              objectFit: 'cover',
                              borderRadius: '4px',
                            }}
                          />
                        </Box>
                      )}
                      
                      {report.damageReportGenerated ? (
                        <Button
                          variant="outlined"
                          fullWidth
                          disabled
                          sx={{ 
                            mt: 1,
                            borderColor: '#4caf50',
                            color: '#4caf50',
                            '&.Mui-disabled': {
                              borderColor: '#4caf50',
                              color: '#4caf50',
                            }
                          }}
                        >
                          ✓ Already Generated
                        </Button>
                      ) : (
                        <Button
                          variant="contained"
                          fullWidth
                          startIcon={<AssignmentIcon />}
                          onClick={() => onSelectReport(report)}
                          sx={{ 
                            mt: 1,
                            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          }}
                        >
                          Generate Damage Report
                        </Button>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 2, borderTop: '1px solid #e5e7eb' }}>
        <Button 
          onClick={onClose} 
          color="inherit"
        >
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AiReportsDialog;
