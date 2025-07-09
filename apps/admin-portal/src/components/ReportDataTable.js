import React, { useState } from 'react';
import { useTheme, alpha } from '@mui/material/styles';
import { 
  Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow, TablePagination, Chip, IconButton, Tooltip, Box,
  Paper
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

function ReportDataTable({ filters }) {
  const theme = useTheme();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Sample data - in a real app, this would be fetched based on filters
  const reports = [
    {
      id: 'RPT-001',
      location: 'Central Park, Zone A',
      region: 'Central',
      damageType: 'Flooding',
      severity: 'High',
      reportDate: '2025-04-12',
      status: 'Pending',
      description: 'Flooding due to broken water main affecting park facilities.'
    },
    {
      id: 'RPT-002',
      location: 'Downtown, Main Street',
      region: 'South',
      damageType: 'Structural',
      severity: 'Medium',
      reportDate: '2025-04-13',
      status: 'In Progress',
      description: 'Cracks in building foundation after recent earthquake.'
    },
    // Add more mock data here
  ];

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const getSeverityChip = (severity) => {
    const severityConfig = {
      'Low': { backgroundColor: theme.palette.success.main, color: theme.palette.success.contrastText },
      'Medium': { backgroundColor: theme.palette.warning.main, color: theme.palette.warning.contrastText },
      'High': { backgroundColor: theme.palette.error.main, color: theme.palette.error.contrastText },
      'Critical': { backgroundColor: theme.palette.error.dark, color: theme.palette.error.contrastText },
    };
    
    const config = severityConfig[severity] || { backgroundColor: theme.palette.grey[500], color: theme.palette.getContrastText(theme.palette.grey[500]) };
    
    return (
      <Chip 
        label={severity} 
        size="small"
        sx={{
          fontWeight: 500,
          borderRadius: 1,
          px: 1,
          backgroundColor: config.backgroundColor,
          color: config.color,
        }}
      />
    );
  };

  const getStatusChip = (status) => {
    const statusConfig = {
      'Completed': { backgroundColor: theme.palette.success.main, color: theme.palette.success.contrastText },
      'In Progress': { backgroundColor: theme.palette.info.main, color: theme.palette.info.contrastText },
      'Assigned': { backgroundColor: theme.palette.primary.main, color: theme.palette.primary.contrastText },
      'Pending': { backgroundColor: theme.palette.warning.main, color: theme.palette.warning.contrastText },
    };
    
    const config = statusConfig[status] || { backgroundColor: theme.palette.grey[500], color: theme.palette.getContrastText(theme.palette.grey[500]) };
    
    return (
      <Chip 
        label={status} 
        size="small"
        sx={{
          fontWeight: 500,
          borderRadius: 1,
          px: 1,
          backgroundColor: config.backgroundColor,
          color: config.color,
        }}
      />
    );
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        borderRadius: 1,
        border: `1px solid ${theme.palette.divider}`,
        overflow: 'hidden',
      }}
    >
      <TableContainer>
        <Table aria-label="damage reports table">
          <TableHead>
            <TableRow>
              <TableCell sx={{ 
                fontWeight: 600, 
                backgroundColor: theme.palette.mode === 'dark' ? theme.palette.background.default : theme.palette.grey[50],
                borderBottom: `2px solid ${theme.palette.divider}`,
                color: theme.palette.text.primary,
              }}>Report ID</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.default, 0.6) : '#f8f9fa',
                borderBottom: `2px solid ${theme.palette.divider}`,
                color: theme.palette.text.primary,
              }}>Location</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.default, 0.6) : '#f8f9fa',
                borderBottom: `2px solid ${theme.palette.divider}`,
                color: theme.palette.text.primary,
              }}>Region</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.default, 0.6) : '#f8f9fa',
                borderBottom: `2px solid ${theme.palette.divider}`,
                color: theme.palette.text.primary,
              }}>Damage Type</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: theme.palette.mode === 'dark' ? alpha(theme.palette.background.default, 0.6) : '#f8f9fa',
                borderBottom: '2px solid #e5e7eb',
                color: '#374151',
              }}>Severity</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: '#f8f9fa',
                borderBottom: '2px solid #e5e7eb',
                color: '#374151',
              }}>Date Reported</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: '#f8f9fa',
                borderBottom: '2px solid #e5e7eb',
                color: '#374151',
              }}>Status</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                backgroundColor: '#f8f9fa',
                borderBottom: '2px solid #e5e7eb',
                color: '#374151',
              }}>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {reports
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((report, index) => (
                <TableRow 
                  key={report.id}
                  sx={{
                    transition: 'background-color 0.15s ease',
                    '&:hover': {
                      backgroundColor: theme.palette.mode === 'dark' ? theme.palette.action.hover : theme.palette.grey[50],
                    },
                    backgroundColor: index % 2 === 0 ? theme.palette.background.paper : theme.palette.mode === 'dark' ? theme.palette.background.default : theme.palette.grey[50]
                  }}
                >
                  <TableCell sx={{ fontWeight: 500, color: theme.palette.primary.main }}>
                    {report.id}
                  </TableCell>
                  <TableCell sx={{ color: '#374151' }}>{report.location}</TableCell>
                  <TableCell sx={{ color: '#374151' }}>{report.region}</TableCell>
                  <TableCell sx={{ color: '#374151' }}>{report.damageType}</TableCell>
                  <TableCell>
                    {getSeverityChip(report.severity)}
                  </TableCell>
                  <TableCell sx={{ color: '#374151' }}>{report.reportDate}</TableCell>
                  <TableCell>
                    {getStatusChip(report.status)}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="View Details" arrow>
                        <IconButton 
                          size="small"
                          sx={{ 
                            color: theme.palette.primary.main,
                            '&:hover': { 
                              backgroundColor: theme.palette.mode === 'dark' ? theme.palette.action.hover : theme.palette.primary.lighter || theme.palette.primary.light + '14',
                            }
                          }}
                        >
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit Report" arrow>
                        <IconButton 
                          size="small"
                          sx={{ 
                            color: '#f59e0b',
                            '&:hover': { 
                              backgroundColor: 'rgba(245, 158, 11, 0.04)',
                            }
                          }}
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete Report" arrow>
                        <IconButton 
                          size="small"
                          sx={{ 
                            color: '#ef4444',
                            '&:hover': { 
                              backgroundColor: 'rgba(239, 68, 68, 0.04)',
                            }
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={reports.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        sx={{
          borderTop: '1px solid #e5e7eb',
          '& .MuiTablePagination-select': {
            borderRadius: 1,
          },
        }}
      />
    </Paper>
  );
}

export default ReportDataTable;