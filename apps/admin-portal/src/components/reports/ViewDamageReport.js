import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Grid,
  Box,
  Chip,
  Divider,
  IconButton
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

function ViewDamageReport({ open, onClose, report }) {
  if (!report) return null;

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Mild': return '#2196f3';
      case 'Moderate': return '#ff9800';
      case 'Severe': return '#f44336';
      default: return '#757575';
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      scroll="paper"
    >
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Damage Report Details</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent dividers>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="h5">Report ID: {report.reportId}</Typography>
              <Chip
                label={report.severity}
                sx={{
                  bgcolor: getSeverityColor(report.severity),
                  color: 'white'
                }}
              />
            </Box>
            <Typography color="textSecondary" gutterBottom>
              Reported on {new Date(report.createdAt).toLocaleString()}
            </Typography>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>Location Details</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography><strong>Region:</strong> {report.region}</Typography>
              {report.address && (
                <Typography><strong>Address:</strong> {report.address}</Typography>
              )}
              {report.coordinates && (
                <Typography>
                  <strong>Coordinates:</strong> {report.coordinates.lat}, {report.coordinates.lng}
                </Typography>
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>Damage Information</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography><strong>Type:</strong> {report.damageType}</Typography>
              <Typography><strong>Priority:</strong> {report.priority}</Typography>
              <Typography><strong>Action Required:</strong> {report.action}</Typography>
              <Box sx={{ mt: 1 }}>
                <Typography><strong>Description:</strong></Typography>
                <Typography sx={{ mt: 0.5 }}>{report.description}</Typography>
              </Box>
            </Box>
          </Grid>

          {report.images && report.images.length > 0 && (
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Images</Typography>
              <Grid container spacing={2}>
                {report.images.map((image, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Box
                      component="img"
                      src={image.url}
                      alt={`Damage report image ${index + 1}`}
                      sx={{
                        width: '100%',
                        height: 200,
                        objectFit: 'cover',
                        borderRadius: 1,
                        cursor: 'pointer'
                      }}
                      onClick={() => window.open(image.url, '_blank')}
                    />
                  </Grid>
                ))}
              </Grid>
            </Grid>
          )}

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>Reporter Information</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography><strong>Reporter:</strong> {report.reporter}</Typography>
              {report.reporterContact && (
                <Typography><strong>Contact:</strong> {report.reporterContact}</Typography>
              )}
            </Box>
          </Grid>

          {report.notes && (
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>Additional Notes</Typography>
              <Typography>{report.notes}</Typography>
            </Grid>
          )}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default ViewDamageReport;
