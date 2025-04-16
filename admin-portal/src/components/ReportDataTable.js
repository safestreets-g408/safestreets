import React, { useState } from 'react';
import { 
  Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow, TablePagination, Chip, IconButton, Tooltip, Box
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

function ReportDataTable({ filters }) {
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
    <>
      <TableContainer>
        <Table aria-label="damage reports table">
          <TableHead>
            <TableRow>
              <TableCell>Report ID</TableCell>
              <TableCell>Location</TableCell>
              <TableCell>Region</TableCell>
              <TableCell>Damage Type</TableCell>
              <TableCell>Severity</TableCell>
              <TableCell>Date Reported</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {reports
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((report) => (
                <TableRow key={report.id}>
                  <TableCell>{report.id}</TableCell>
                  <TableCell>{report.location}</TableCell>
                  <TableCell>{report.region}</TableCell>
                  <TableCell>{report.damageType}</TableCell>
                  <TableCell>
                    <Chip 
                      label={report.severity} 
                      color={getSeverityColor(report.severity)} 
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{report.reportDate}</TableCell>
                  <TableCell>
                    <Chip 
                      label={report.status} 
                      color={getStatusColor(report.status)} 
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Box>
                      <Tooltip title="View Details">
                        <IconButton size="small">
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit Report">
                        <IconButton size="small">
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete Report">
                        <IconButton size="small">
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
      />
    </>
  );
}

export default ReportDataTable;