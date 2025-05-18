import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, Grid, TextField, MenuItem, 
  Button, FormControl, InputLabel, Select, Chip, Stack,
  Card, CardContent, IconButton, Tooltip,
  Pagination, Menu, ListItemIcon, ListItemText, CircularProgress
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
// import ReportDataTable from '../components/ReportDataTable'; // Not used in this rewrite

const API_URL = 'http://localhost:5030/api/damage/reports';

function Reports() {
  const [filters, setFilters] = useState({
    dateFrom: null,
    dateTo: null,
    severity: '',
    region: '',
    damageType: '',
    priority: '',
  });
  const [viewMode, setViewMode] = useState('table');
  const [page, setPage] = useState(1);
  const [sortField, setSortField] = useState('createdAt');
  const [sortDirection, setSortDirection] = useState('desc');
  const [anchorEl, setAnchorEl] = useState(null);
  const exportMenuOpen = Boolean(anchorEl);

  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);

  // Pagination
  const pageSize = 6;
  const totalPages = Math.ceil(filteredReports().length / pageSize);

  // Fetch reports from API
  useEffect(() => {
    setLoading(true);
    setFetchError(null);
    fetch(API_URL)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch reports');
        return res.json();
      })
      .then(data => {
        setReports(data);
        setLoading(false);
      })
      .catch(err => {
        setFetchError(err.message);
        setLoading(false);
      });
  }, []);

  // Filtering logic
  function filteredReports() {
    return reports.filter(report => {
      // Date filter
      if (filters.dateFrom && new Date(report.createdAt) < new Date(filters.dateFrom)) return false;
      if (filters.dateTo && new Date(report.createdAt) > new Date(filters.dateTo)) return false;
      // Severity filter
      if (filters.severity && report.severity !== filters.severity) return false;
      // Region filter
      if (filters.region && report.region !== filters.region) return false;
      // Damage Type filter
      if (filters.damageType && report.damageType !== filters.damageType) return false;
      // Priority filter
      if (filters.priority && report.priority !== filters.priority) return false;
      return true;
    });
  }

  // Sorting logic
  function sortedReports() {
    const arr = [...filteredReports()];
    arr.sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];
      if (sortField === 'createdAt') {
        aVal = new Date(aVal);
        bVal = new Date(bVal);
      }
      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
    return arr;
  }

  // Paginated data
  function paginatedReports() {
    const sorted = sortedReports();
    const start = (page - 1) * pageSize;
    return sorted.slice(start, start + pageSize);
  }

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
      priority: '',
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
    // For now, just log
    console.log(`Exporting as ${format}`);
    handleExportClose();
  };

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  // Build options from data
  const severityOptions = Array.from(new Set(reports.map(r => r.severity))).filter(Boolean);
  const regionOptions = Array.from(new Set(reports.map(r => r.region))).filter(Boolean);
  const damageTypeOptions = Array.from(new Set(reports.map(r => r.damageType))).filter(Boolean);
  const priorityOptions = Array.from(new Set(reports.map(r => r.priority))).filter(Boolean);

  // Status mapping for icon (using action/priority as proxy)
  const getStatusIcon = (priority, action) => {
    // Map priority/action to status icon
    if (priority === 'High') return <BuildIcon color="error" />;
    if (priority === 'Medium') return <AssignmentIndIcon color="warning" />;
    if (priority === 'Low') return <PendingIcon color="info" />;
    if (action && /immediate/i.test(action)) return <CheckCircleIcon color="success" />;
    return <PendingIcon color="disabled" />;
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Mild': return '#2196f3'; // blue
      case 'Moderate': return '#ff9800'; // orange
      case 'Severe': return '#f44336'; // red
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
              <InputLabel>Priority</InputLabel>
              <Select
                value={filters.priority}
                label="Priority"
                onChange={(e) => handleFilterChange('priority', e.target.value)}
              >
                <MenuItem value="">All</MenuItem>
                {priorityOptions.map(option => (
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
            {filters.priority && (
              <Chip 
                label={`Priority: ${filters.priority}`} 
                onDelete={() => handleFilterChange('priority', '')}
              />
            )}
          </Stack>
          
          <Button variant="outlined" onClick={clearFilters}>
            Clear Filters
          </Button>
        </Box>
      </Paper>
      
      <Paper sx={{ p: 2 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
            <CircularProgress />
          </Box>
        ) : fetchError ? (
          <Box sx={{ color: 'error.main', textAlign: 'center', py: 4 }}>
            <Typography color="error">Error: {fetchError}</Typography>
          </Box>
        ) : (
          <>
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
                      <MenuItem value="createdAt">Date Reported</MenuItem>
                      <MenuItem value="severity">Severity</MenuItem>
                      <MenuItem value="region">Region</MenuItem>
                      <MenuItem value="priority">Priority</MenuItem>
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
                {/* Table View */}
                <Box sx={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Report ID</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Date</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Region</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Type</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Severity</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Priority</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Action</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Reporter</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Status</th>
                        <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedReports().map(report => (
                        <tr key={report._id}>
                          <td style={{ padding: 8 }}>{report.reportId}</td>
                          <td style={{ padding: 8 }}>{new Date(report.createdAt).toLocaleDateString()}</td>
                          <td style={{ padding: 8 }}>{report.region}</td>
                          <td style={{ padding: 8 }}>{report.damageType}</td>
                          <td style={{ padding: 8 }}>
                            <Chip 
                              label={report.severity} 
                              size="small" 
                              sx={{ bgcolor: getSeverityColor(report.severity), color: 'white' }} 
                            />
                          </td>
                          <td style={{ padding: 8 }}>{report.priority}</td>
                          <td style={{ padding: 8 }}>{report.action}</td>
                          <td style={{ padding: 8 }}>{report.reporter}</td>
                          <td style={{ padding: 8 }}>
                            {getStatusIcon(report.priority, report.action)}
                          </td>
                          <td style={{ padding: 8 }}>
                            <Button size="small" color="primary">View</Button>
                          </td>
                        </tr>
                      ))}
                      {paginatedReports().length === 0 && (
                        <tr>
                          <td colSpan={10} style={{ textAlign: 'center', padding: 24, color: '#888' }}>
                            No reports found.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                  <Pagination 
                    count={totalPages || 1} 
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
                  {paginatedReports().map(report => (
                    <Grid item xs={12} sm={6} md={4} key={report._id}>
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
                            {getStatusIcon(report.priority, report.action)}
                          </Box>
                          <Typography variant="h6" component="div" gutterBottom>
                            {report.reportId}
                          </Typography>
                          <Typography color="text.secondary" gutterBottom>
                            Reported: {new Date(report.createdAt).toLocaleDateString()}
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
                            Priority: {report.priority}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Action: {report.action}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                            <Button size="small" color="primary">View Details</Button>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                  {paginatedReports().length === 0 && (
                    <Grid item xs={12}>
                      <Box sx={{ textAlign: 'center', color: '#888', py: 4 }}>
                        No reports found.
                      </Box>
                    </Grid>
                  )}
                </Grid>
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                  <Pagination 
                    count={totalPages || 1} 
                    page={page} 
                    onChange={handlePageChange} 
                    color="primary" 
                    showFirstButton 
                    showLastButton
                  />
                </Box>
              </>
            )}
          </>
        )}
      </Paper>
    </>
  );
}

export default Reports;