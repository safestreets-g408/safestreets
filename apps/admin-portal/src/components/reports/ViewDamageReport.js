import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Chip, 
  Divider,
  Paper,
  CircularProgress
} from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';
import {api} from '../../utils/api';

const ViewDamageReport = ({ report }) => {
  const theme = useTheme();
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchImage = async () => {
      if (report.beforeImage?._id) {
        try {
          const response = await api.get(
            `/damage/report/${report.reportId}/image/before`,
            { responseType: 'blob' }
          );
          setImageUrl(URL.createObjectURL(response.data));
        } catch (error) {
          console.error('Error fetching image:', error);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchImage();
    return () => {
      // Cleanup URL object on unmount
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report]);

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed': return 'success';
      case 'in progress': return 'info';
      case 'pending': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Report ID and Date */}
        <Grid item xs={12}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h5" gutterBottom>
              Report #{report.reportId}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Submitted on {new Date(report.createdAt).toLocaleString()}
            </Typography>
          </Box>
          <Divider />
        </Grid>

        {/* Status and Severity */}
        <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" gutterBottom>Status</Typography>
            <Chip 
              label={report.status}
              color={getStatusColor(report.status)}
              sx={{ fontWeight: 600 }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2" gutterBottom>Assignment</Typography>
            <Typography>{report.assignedTo || 'Unassigned'}</Typography>
          </Paper>
        </Grid>

        {/* Location Details */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Location Details</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">Region</Typography>
                <Typography variant="body1">{report.region}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">Location</Typography>
                <Typography variant="body1">{report.location}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Damage Details */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Damage Information</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">Damage Type</Typography>
                <Typography variant="body1">{report.damageType}</Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">Severity</Typography>
                <Chip 
                  label={report.severity}
                  color={getSeverityColor(report.severity)}
                  size="small"
                />
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">Priority</Typography>
                <Typography variant="body1">{report.priority}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">Description</Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>
                  {report.description}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">Action Required</Typography>
                <Typography variant="body1">{report.action}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Image */}
        {report.beforeImage && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Damage Image</Typography>
              <Box
                sx={{
                  position: 'relative',
                  width: '100%',
                  height: 300,
                  borderRadius: 1,
                  overflow: 'hidden',
                  backgroundColor: alpha(theme.palette.common.black, 0.04)
                }}
              >
                {loading ? (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%'
                    }}
                  >
                    <CircularProgress />
                  </Box>
                ) : imageUrl ? (
                  <img
                    src={imageUrl}
                    alt="Damage"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                ) : (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%'
                    }}
                  >
                    <Typography color="text.secondary">
                      No image available
                    </Typography>
                  </Box>
                )}
              </Box>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ViewDamageReport;
