import React, { useState } from 'react';
import { 
  Box, Paper, Typography, Grid, TextField, MenuItem, 
  Button, FormControl, InputLabel, Select, Chip, Stack,
  Card, CardContent, IconButton, Tooltip,
  Pagination, Menu, ListItemIcon, ListItemText
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import SortIcon from '@mui/icons-material/Sort';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import GridViewIcon from '@mui/icons-material/GridView';
import TableRowsIcon from '@mui/icons-material/TableRows';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PendingIcon from '@mui/icons-material/Pending';
import BuildIcon from '@mui/icons-material/Build';
import CancelIcon from '@mui/icons-material/Cancel';
import AssignmentIndIcon from '@mui/icons-material/AssignmentInd';
import ReportDataTable from '../components/ReportDataTable';

function Reports() {
  const [filters, setFilters] = useState({
    dateFrom: null,
    dateTo: null,
    severity: '',
    region: '',
    damageType: '',
    repairStatus: '',
  });
  const [viewMode, setViewMode] = useState('table');
  const [page, setPage] = useState(1);
  const [sortField, setSortField] = useState('dateReported');
  const [sortDirection, setSortDirection] = useState('desc');
  const [anchorEl, setAnchorEl] = useState(null);
  const exportMenuOpen = Boolean(anchorEl);

  const handleFilterChange = (field, value) => {
    setFilters({
      ...filters,
      [field]: value
    });
    setPage(1); // Reset to first page when filters change
  };

  const clearFilters = () => {
    setFilters({
      dateFrom: null,
      dateTo: null,
      severity: '',
      region: '',
      damageType: '',
      repairStatus: '',
    });
    setPage(1);
  };

  const handleSortChange = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleExportClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleExportClose = () => {
    setAnchorEl(null);
  };

  const handleExport = (format) => {
    // Export logic would go here
    console.log(`Exporting as ${format}`);
    handleExportClose();
  };

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  const severityOptions = ['Low', 'Medium', 'High', 'Critical'];
  const regionOptions = ['North', 'South', 'East', 'West', 'Central'];
  const damageTypeOptions = ['Structural', 'Electrical', 'Plumbing', 'Flooding', 'Fire', 'Other'];
  const repairStatusOptions = ['Pending', 'Assigned', 'In-Progress', 'Resolved', 'Rejected'];

  // Mock data for card view
  const mockReports = [
    { id: 'DR-2023-001', dateReported: '2023-06-15', severity: 'High', region: 'North', damageType: 'Structural', reporter: 'John Doe', status: 'In-Progress', description: 'Major structural damage to building facade' },
    { id: 'DR-2023-002', dateReported: '2023-06-16', severity: 'Medium', region: 'South', damageType: 'Electrical', reporter: 'Jane Smith', status: 'Assigned', description: 'Power outage affecting multiple buildings' },
    { id: 'DR-2023-003', dateReported: '2023-06-17', severity: 'Critical', region: 'Central', damageType: 'Flooding', reporter: 'Mike Johnson', status: 'Pending', description: 'Severe flooding in basement level' },
    { id: 'DR-2023-004', dateReported: '2023-06-18', severity: 'Low', region: 'East', damageType: 'Plumbing', reporter: 'Sarah Williams', status: 'Resolved', description: 'Minor leak in bathroom pipes' },
  ];

  const getStatusIcon = (status) => {
    switch(status) {
      case 'Pending': return <PendingIcon color="warning" />;
      case 'Assigned': return <AssignmentIndIcon color="info" />;
      case 'In-Progress': return <BuildIcon color="primary" />;
      case 'Resolved': return <CheckCircleIcon color="success" />;
      case 'Rejected': return <CancelIcon color="error" />;
      default: return <PendingIcon color="warning" />;
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Low': return '#2196f3'; // blue
      case 'Medium': return '#ff9800'; // orange
      case 'High': return '#f44336'; // red
      case 'Critical': return '#9c27b0'; // purple
      default: return '#757575'; // grey
    }
  };

  return (
    <>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom sx={{fontWeight: 'semi-bold'}}>
          Damage Reports
        </Typography>
        <Box>
          <Tooltip title="Export Reports">
            <IconButton onClick={handleExportClick}>
              <FileDownloadIcon />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={anchorEl}
            open={exportMenuOpen}
            onClose={handleExportClose}
          >
            <MenuItem onClick={() => handleExport('csv')}>
              <ListItemIcon>
                <FileDownloadIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Export as CSV</ListItemText>
            </MenuItem>
            <MenuItem onClick={() => handleExport('excel')}>
              <ListItemIcon>
                <FileDownloadIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Export as Excel</ListItemText>
            </MenuItem>
          </Menu>
          <Tooltip title="Toggle View">
            <IconButton onClick={() => setViewMode(viewMode === 'table' ? 'card' : 'table')}>
              {viewMode === 'table' ? <GridViewIcon /> : <TableRowsIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <FilterListIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Filters</Typography>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              label="From Date"
              type="date"
              value={filters.dateFrom || ''}
              onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
              InputLabelProps={{
                shrink: true,
              }}
              fullWidth
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              label="To Date"
              type="date"
              value={filters.dateTo || ''}
              onChange={(e) => handleFilterChange('dateTo', e.target.value)}
              InputLabelProps={{
                shrink: true,
              }}
              fullWidth
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth>
              <InputLabel>Severity</InputLabel>
              <Select
                value={filters.severity}
                label="Severity"
                onChange={(e) => handleFilterChange('severity', e.target.value)}
              >
                <MenuItem value="">All</MenuItem>
                {severityOptions.map(option => (
                  <MenuItem key={option} value={option}>{option}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth>
              <InputLabel>Region</InputLabel>
              <Select
                value={filters.region}
                label="Region"
                onChange={(e) => handleFilterChange('region', e.target.value)}
              >
                <MenuItem value="">All</MenuItem>
                {regionOptions.map(option => (
                  <MenuItem key={option} value={option}>{option}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth>
              <InputLabel>Damage Type</InputLabel>
              <Select
                value={filters.damageType}
                label="Damage Type"
                onChange={(e) => handleFilterChange('damageType', e.target.value)}
              >
                <MenuItem value="">All</MenuItem>
                {damageTypeOptions.map(option => (
                  <MenuItem key={option} value={option}>{option}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth>
              <InputLabel>Repair Status</InputLabel>
              <Select
                value={filters.repairStatus}
                label="Repair Status"
                onChange={(e) => handleFilterChange('repairStatus', e.target.value)}
              >
                <MenuItem value="">All</MenuItem>
                {repairStatusOptions.map(option => (
                  <MenuItem key={option} value={option}>{option}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
            {filters.dateFrom && (
              <Chip 
                label={`From: ${filters.dateFrom}`} 
                onDelete={() => handleFilterChange('dateFrom', null)}
              />
            )}
            {filters.dateTo && (
              <Chip 
                label={`To: ${filters.dateTo}`} 
                onDelete={() => handleFilterChange('dateTo', null)}
              />
            )}
            {filters.severity && (
              <Chip 
                label={`Severity: ${filters.severity}`} 
                onDelete={() => handleFilterChange('severity', '')}
              />
            )}
            {filters.region && (
              <Chip 
                label={`Region: ${filters.region}`} 
                onDelete={() => handleFilterChange('region', '')}
              />
            )}
            {filters.damageType && (
              <Chip 
                label={`Type: ${filters.damageType}`} 
                onDelete={() => handleFilterChange('damageType', '')}
              />
            )}
            {filters.repairStatus && (
              <Chip 
                label={`Status: ${filters.repairStatus}`} 
                onDelete={() => handleFilterChange('repairStatus', '')}
              />
            )}
          </Stack>
          
          <Button variant="outlined" onClick={clearFilters}>
            Clear Filters
          </Button>
        </Box>
      </Paper>
      
      <Paper sx={{ p: 2 }}>
        {viewMode === 'table' ? (
          <>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
              <FormControl variant="outlined" size="small" sx={{ minWidth: 200, mr: 2 }}>
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={sortField}
                  label="Sort By"
                  onChange={(e) => handleSortChange(e.target.value)}
                  startAdornment={<SortIcon sx={{ mr: 1 }} />}
                >
                  <MenuItem value="dateReported">Date Reported</MenuItem>
                  <MenuItem value="severity">Severity</MenuItem>
                  <MenuItem value="region">Region</MenuItem>
                  <MenuItem value="status">Status</MenuItem>
                </Select>
              </FormControl>
              <Button 
                variant="outlined" 
                startIcon={<SortIcon />}
                onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
              >
                {sortDirection === 'asc' ? 'Ascending' : 'Descending'}
              </Button>
            </Box>
            <ReportDataTable 
              filters={filters} 
              sortField={sortField}
              sortDirection={sortDirection}
              page={page}
            />
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Pagination 
                count={10} 
                page={page} 
                onChange={handlePageChange} 
                color="primary" 
                showFirstButton 
                showLastButton
              />
            </Box>
          </>
        ) : (
          <>
            <Grid container spacing={3}>
              {mockReports.map(report => (
                <Grid item xs={12} sm={6} md={4} key={report.id}>
                  <Card sx={{ 
                    height: '100%', 
                    position: 'relative',
                    borderLeft: `4px solid ${getSeverityColor(report.severity)}`,
                    transition: 'transform 0.2s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 4
                    }
                  }}>
                    <CardContent>
                      <Box sx={{ position: 'absolute', top: 12, right: 12 }}>
                        {getStatusIcon(report.status)}
                      </Box>
                      <Typography variant="h6" component="div" gutterBottom>
                        {report.id}
                      </Typography>
                      <Typography color="text.secondary" gutterBottom>
                        Reported: {new Date(report.dateReported).toLocaleDateString()}
                      </Typography>
                      <Box sx={{ mb: 1.5 }}>
                        <Chip 
                          label={report.severity} 
                          size="small" 
                          sx={{ 
                            bgcolor: getSeverityColor(report.severity),
                            color: 'white',
                            mr: 1
                          }} 
                        />
                        <Chip label={report.region} size="small" sx={{ mr: 1 }} />
                        <Chip label={report.damageType} size="small" />
                      </Box>
                      <Typography variant="body2" sx={{ mb: 1.5 }}>
                        {report.description}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Reporter: {report.reporter}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Status: {report.status}
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                        <Button size="small" color="primary">View Details</Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Pagination 
                count={10} 
                page={page} 
                onChange={handlePageChange} 
                color="primary" 
                showFirstButton 
                showLastButton
              />
            </Box>
          </>
        )}
      </Paper>
    </>
  );
}

export default Reports;