import React from 'react';
import { 
  Dialog, 
  DialogTitle, 
  DialogContent, 
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
import { formatLocation, getCoordinatesString } from '../../utils/formatters';

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
        borderBottom: '1px solid',
        borderColor: 'divider',
        py: 1.5,
        px: 2.5
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
            AI Generated Reports
          </Typography>
        </Box>
        <IconButton 
          onClick={onClose} 
          size="small"
          edge="end"
          sx={{ color: 'text.secondary' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 2, px: 2.5 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
            <CircularProgress size={24} />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ my: 1.5, py: 0.5, fontSize: '0.8rem' }}>
            {error}
          </Alert>
        ) : reports.length === 0 ? (
          <Alert severity="info" sx={{ my: 1.5, py: 0.5, fontSize: '0.8rem' }}>
            No AI reports found. Upload images in the AI Analysis section to generate reports.
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {reports.map((report) => (
              <Grid item xs={12} sm={6} md={4} key={report._id}>
                <Card 
                  sx={{ 
                    borderRadius: 1,
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
                    transition: 'transform 0.15s ease-in-out',
                    opacity: report.damageReportGenerated ? 0.8 : 1,
                    border: report.damageReportGenerated ? '1px solid' : '1px solid',
                    borderColor: report.damageReportGenerated ? 'success.light' : 'divider',
                    '&:hover': {
                      transform: report.damageReportGenerated ? 'none' : 'translateY(-2px)',
                      boxShadow: report.damageReportGenerated 
                        ? '0 2px 8px rgba(0, 0, 0, 0.05)' 
                        : '0 4px 12px rgba(0, 0, 0, 0.08)',
                    }
                  }}
                >
                  <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                        {report.damageType || 'Damage Report'}
                      </Typography>
                      {report.damageReportGenerated && (
                        <Chip 
                          label="Report Generated"
                          color="success"
                          size="small"
                          sx={{ height: '18px', '& .MuiChip-label': { px: 0.8, fontSize: '0.65rem' } }}
                        />
                      )}
                    </Box>
                    <Stack spacing={0.5}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                        Priority: {report.priority}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                        Location: {formatLocation(report.location)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                        Coordinates: {getCoordinatesString(report.location)}
                      </Typography>
                      <Box sx={{ mt: 0.5 }}>
                        <Chip
                          label={`Severity: ${report.severity}`}
                          color={getSeverityColor(report.severity)}
                          size="small"
                          sx={{ height: '20px', '& .MuiChip-label': { px: 0.8, fontSize: '0.7rem' } }}
                        />
                      </Box>
                    </Stack>
                    <Box sx={{ mt: 1.5, display: 'flex', justifyContent: 'flex-end' }}>
                      <Button
                        variant="contained"
                        size="small"
                        sx={{ 
                          fontSize: '0.75rem', 
                          py: 0.5,
                          px: 1.5, 
                          minWidth: '0',
                          borderRadius: 0.75,
                          textTransform: 'none',
                          fontWeight: 500
                        }}
                        onClick={(e) => {
                          // Prevent passing the entire event object to avoid circular references
                          e.preventDefault();
                          
                          console.log('Original AI report:', report);
                          
                          // Pass only the data needed from the report, but ensure annotatedImageBase64 is included
                          const cleanReport = {
                            _id: report._id || report.id,
                            id: report._id || report.id,
                            imageId: report.imageId,
                            damageType: report.damageType,
                            severity: report.severity,
                            priority: report.priority,
                            predictionClass: report.predictionClass,
                            location: report.location,
                            annotatedImageBase64: report.annotatedImageBase64,
                            createdAt: report.createdAt
                          };
                          
                          console.log('Clean report before passing to parent:', {
                            ...cleanReport,
                            hasAnnotatedImage: !!cleanReport.annotatedImageBase64,
                            imageLength: cleanReport.annotatedImageBase64?.length || 0
                          });
                          
                          onSelectReport(cleanReport);
                        }}
                        disabled={report.damageReportGenerated}
                      >
                        Generate Report
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default AiReportsDialog;
