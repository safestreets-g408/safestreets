import React, { useState } from 'react';
import { 
  Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow, TablePagination, Chip, IconButton, Tooltip, Box,
  useTheme, alpha, Paper
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

function ReportDataTable({ filters }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const theme = useTheme();

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

  const getSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case 'low':
        return 'success';
      case 'medium':
        return 'warning';
      case 'high':
        return 'error';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'success';
      case 'in progress':
        return 'info';
      case 'assigned':
        return 'primary';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        borderRadius: 2,
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        overflow: 'hidden',
      }}
    >
      <TableContainer>
        <Table aria-label="damage reports table">
          <TableHead>
            <TableRow>
              <TableCell sx={{ 
                fontWeight: 600, 
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Report ID</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Location</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Region</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Damage Type</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Severity</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Date Reported</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
              }}>Status</TableCell>
              <TableCell sx={{ 
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.04),
                borderBottom: `2px solid ${alpha(theme.palette.primary.main, 0.1)}`
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
                    transition: 'background-color 0.2s ease-in-out',
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.04),
                    },
                    bgcolor: index % 2 === 0 ? alpha(theme.palette.background.default, 0.5) : 'inherit'
                  }}
                >
                  <TableCell sx={{ fontWeight: 500, color: theme.palette.primary.main }}>
                    {report.id}
                  </TableCell>
                  <TableCell>{report.location}</TableCell>
                  <TableCell>{report.region}</TableCell>
                  <TableCell>{report.damageType}</TableCell>
                  <TableCell>
                    <Chip 
                      label={report.severity} 
                      color={getSeverityColor(report.severity)} 
                      size="small"
                      sx={{
                        fontWeight: 600,
                        borderRadius: 1,
                        px: 1,
                      }}
                    />
                  </TableCell>
                  <TableCell>{report.reportDate}</TableCell>
                  <TableCell>
                    <Chip 
                      label={report.status} 
                      color={getStatusColor(report.status)} 
                      size="small"
                      sx={{
                        fontWeight: 600,
                        borderRadius: 1,
                        px: 1,
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="View Details" arrow>
                        <IconButton 
                          size="small"
                          sx={{ 
                            color: theme.palette.primary.main,
                            '&:hover': { 
                              bgcolor: alpha(theme.palette.primary.main, 0.08),
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
                            color: theme.palette.warning.main,
                            '&:hover': { 
                              bgcolor: alpha(theme.palette.warning.main, 0.08),
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
                            color: theme.palette.error.main,
                            '&:hover': { 
                              bgcolor: alpha(theme.palette.error.main, 0.08),
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
          borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          '& .MuiTablePagination-select': {
            borderRadius: 1,
          },
        }}
      />
    </Paper>
  );
}

export default ReportDataTable;