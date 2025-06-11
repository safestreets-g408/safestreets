/* eslint-disable no-unused-vars */
import React, { useState, useEffect} from 'react';
import { 
  Box, Paper, Typography, Grid, TextField, MenuItem, 
  Button, FormControl, InputLabel, Select, Chip, Stack,
  Card, CardContent, IconButton, Tooltip,
  Pagination, Menu, ListItemIcon, ListItemText, CircularProgress,
  Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';
import FilterListIcon from '@mui/icons-material/FilterList';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import GridViewIcon from '@mui/icons-material/GridView';
import TableRowsIcon from '@mui/icons-material/TableRows';
import CloseIcon from '@mui/icons-material/Close';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ViewDamageReport from '../components/reports/ViewDamageReport';
import { api } from '../utils/api';

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
  const [filtersDialogOpen, setFiltersDialogOpen] = useState(false);
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [viewReportOpen, setViewReportOpen] = useState(false);

  const theme = useTheme();
  const exportMenuOpen = Boolean(anchorEl);

  // Pagination
  const pageSize = 6;
  const totalPages = Math.ceil(filteredReports().length / pageSize);

  // Fetch reports from API
  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        setFetchError(null);

        const queryParams = new URLSearchParams();
        if (filters.dateFrom) queryParams.append('startDate', filters.dateFrom);
        if (filters.dateTo) queryParams.append('endDate', filters.dateTo);
        if (filters.severity) queryParams.append('severity', filters.severity);
        if (filters.region) queryParams.append('region', filters.region);
        if (filters.priority) queryParams.append('priority', filters.priority);
        
        const query = queryParams.toString();
        const endpoint = `/damage/reports${query ? `?${query}` : ''}`;
        const data = await api.get(endpoint);
        setReports(data);
      } catch (err) {
        console.error('Error fetching reports:', err);
        setFetchError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, [filters]); 

  // Utility functions
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return 'success';
      case 'in progress':
        return 'info';
      case 'pending':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
        return 'error';
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  function filteredReports() {
    return reports.filter(report => {
      if (filters.dateFrom && new Date(report.createdAt) < new Date(filters.dateFrom)) return false;
      if (filters.dateTo && new Date(report.createdAt) > new Date(filters.dateTo)) return false;
      if (filters.severity && report.severity !== filters.severity) return false;
      if (filters.region && report.region !== filters.region) return false;
      if (filters.damageType && report.damageType !== filters.damageType) return false;
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

  const handleExportMenuClose = () => {
    setAnchorEl(null);
  };

  const handleExport = (format) => {
    // Export logic implementation
    console.log(`Exporting as ${format}`);
    handleExportMenuClose();
  };

  const handlePageChange = (event, newPage) => {
    setPage(newPage);
  };

  const handleViewReport = (report) => {
    setSelectedReport(report);
    setViewReportOpen(true);
  };

  const handleCloseReport = () => {
    setViewReportOpen(false);
    setSelectedReport(null);
  };

  const handleFilterDelete = (key) => {
    setFilters({
      ...filters,
      [key]: '',
    });
  };

  // Build options from data
  const severityOptions = Array.from(new Set(reports.map(r => r.severity))).filter(Boolean);
  const regionOptions = Array.from(new Set(reports.map(r => r.region))).filter(Boolean);
  const damageTypeOptions = Array.from(new Set(reports.map(r => r.damageType))).filter(Boolean);
  const priorityOptions = Array.from(new Set(reports.map(r => r.priority))).filter(Boolean);

  // Filter Dialog
  const FilterDialog = () => (
    <Dialog 
      open={filtersDialogOpen} 
      onClose={() => setFiltersDialogOpen(false)}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle sx={{ 
        pb: 2, 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Typography variant="h6">Filter Reports</Typography>
        <IconButton 
          onClick={() => setFiltersDialogOpen(false)}
          size="small"
          sx={{
            color: theme.palette.text.secondary,
            '&:hover': { color: theme.palette.text.primary }
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Date From"
              type="date"
              value={filters.dateFrom || ''}
              onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Date To"
              type="date"
              value={filters.dateTo || ''}
              onChange={(e) => handleFilterChange('dateTo', e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
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
          <Grid item xs={12} sm={6}>
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
          <Grid item xs={12} sm={6}>
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
          <Grid item xs={12} sm={6}>
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
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button 
          onClick={() => {
            clearFilters();
            setFiltersDialogOpen(false);
          }}
          color="inherit"
          sx={{ mr: 1 }}
        >
          Clear All
        </Button>
        <Button 
          variant="contained" 
          onClick={() => setFiltersDialogOpen(false)}
        >
          Apply Filters
        </Button>
      </DialogActions>
    </Dialog>
  );

  const renderTable = () => (
    <Paper 
      elevation={0}
      sx={{ 
        borderRadius: 2,
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        overflow: 'hidden',
      }}
    >
      <Box sx={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Report ID</th>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Date</th>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Region</th>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Type</th>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Severity</th>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Status</th>
              <th style={{ padding: 8, borderBottom: '1px solid #eee' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {paginatedReports().map((report, index) => (
              <tr 
                key={report.id}
                style={{
                  transition: 'background-color 0.2s ease-in-out',
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.04),
                  },
                  backgroundColor: index % 2 === 0 ? alpha(theme.palette.background.default, 0.5) : 'inherit'
                }}
              >
                <td style={{ padding: 8, fontWeight: 500, color: theme.palette.primary.main }}>
                  {report.reportId}
                </td>
                <td style={{ padding: 8 }}>
                  {new Date(report.createdAt).toLocaleDateString()}
                </td>
                <td style={{ padding: 8 }}>{report.region}</td>
                <td style={{ padding: 8 }}>{report.damageType}</td>
                <td style={{ padding: 8 }}>
                  <Chip 
                    label={report.severity}
                    color={getSeverityColor(report.severity)}
                    size="small"
                    sx={{
                      fontWeight: 600,
                      borderRadius: 1,
                    }}
                  />
                </td>
                <td style={{ padding: 8 }}>
                  <Chip 
                    label={report.status}
                    color={getStatusColor(report.status)}
                    size="small"
                    sx={{
                      fontWeight: 600,
                      borderRadius: 1,
                    }}
                  />
                </td>
                <td style={{ padding: 8 }}>
                  <Stack direction="row" spacing={1}>
                    <Tooltip title="View Details" arrow>
                      <IconButton 
                        size="small"
                        onClick={() => handleViewReport(report)}
                        sx={{ 
                          color: theme.palette.primary.main,
                          '&:hover': { 
                            backgroundColor: alpha(theme.palette.primary.main, 0.08),
                          }
                        }}
                      >
                        <VisibilityIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    {/* Add more action buttons as needed */}
                  </Stack>
                </td>
              </tr>
            ))}
            {paginatedReports().length === 0 && (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: 24, color: '#888' }}>
                  No reports found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Box>
    </Paper>
  );

  const renderGrid = () => (
    <Grid container spacing={3}>
      {paginatedReports().map(report => (
        <Grid item xs={12} sm={6} md={4} key={report.id}>
          <Card 
            sx={{ 
              height: '100%',
              borderRadius: 2,
              transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[8],
              },
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              overflow: 'visible',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '4px',
                background: `linear-gradient(90deg, ${theme.palette[getSeverityColor(report.severity)].main}, ${theme.palette[getSeverityColor(report.severity)].light})`,
                borderRadius: '8px 8px 0 0',
              }
            }}
          >
            <CardContent sx={{ flex: 1 }}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" component="div" gutterBottom>
                  {report.id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {new Date(report.createdAt).toLocaleDateString()}
                </Typography>
              </Box>
              
              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <Chip 
                  label={report.severity}
                  color={getSeverityColor(report.severity)}
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
                <Chip 
                  label={report.status}
                  color={getStatusColor(report.status)}
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
              </Stack>

              <Typography variant="body2" paragraph>
                {report.description}
              </Typography>

              <Box sx={{ mt: 'auto' }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {report.region} • {report.damageType}
                </Typography>
              </Box>
            </CardContent>

            <Box 
              sx={{ 
                p: 2, 
                pt: 0,
                borderTop: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
                display: 'flex',
                justifyContent: 'flex-end'
              }}
            >
              <Button
                size="small"
                variant="contained"
                onClick={() => handleViewReport(report)}
                sx={{
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: theme.palette.primary.main,
                  '&:hover': {
                    bgcolor: alpha(theme.palette.primary.main, 0.2),
                  },
                }}
              >
                View Details
              </Button>
            </Box>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <>
      <Box sx={{ py: 3 }}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            justifyContent: 'space-between',
            alignItems: { xs: 'stretch', md: 'center' },
            mb: 4,
            gap: 2,
          }}
        >
          <Typography
            variant="h4"
            sx={{
              fontSize: { xs: '1.5rem', sm: '1.875rem' },
              fontWeight: 700,
              color: theme.palette.text.primary,
              letterSpacing: '-0.5px',
            }}
          >
            Damage Reports
          </Typography>

          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={2}
            sx={{ minWidth: { sm: '400px' } }}
          >
            <Button
              variant="contained"
              startIcon={<FilterListIcon />}
              onClick={() => setFiltersDialogOpen(true)}
              sx={{
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: theme.palette.primary.main,
                '&:hover': {
                  bgcolor: alpha(theme.palette.primary.main, 0.2),
                },
                px: 3,
              }}
            >
              Filters
            </Button>

            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant={viewMode === 'grid' ? 'contained' : 'outlined'}
                onClick={() => setViewMode('grid')}
                sx={{
                  minWidth: 'auto',
                  px: 2,
                  ...(viewMode === 'grid' ? {
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    color: theme.palette.primary.main,
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.2),
                    },
                  } : {
                    borderColor: alpha(theme.palette.divider, 0.2),
                  })
                }}
              >
                <GridViewIcon />
              </Button>
              <Button
                variant={viewMode === 'table' ? 'contained' : 'outlined'}
                onClick={() => setViewMode('table')}
                sx={{
                  minWidth: 'auto',
                  px: 2,
                  ...(viewMode === 'table' ? {
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    color: theme.palette.primary.main,
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.2),
                    },
                  } : {
                    borderColor: alpha(theme.palette.divider, 0.2),
                  })
                }}
              >
                <TableRowsIcon />
              </Button>
            </Box>

            <Button
              variant="contained"
              startIcon={<FileDownloadIcon />}
              onClick={(e) => setAnchorEl(e.currentTarget)}
              sx={{
                bgcolor: theme.palette.success.main,
                '&:hover': {
                  bgcolor: theme.palette.success.dark,
                },
              }}
            >
              Export
            </Button>
          </Stack>
        </Box>

        {/* Active Filters */}
        <Box sx={{ mb: 3 }}>
          <Stack
            direction="row"
            spacing={1}
            flexWrap="wrap"
            sx={{ gap: 1 }}
          >
            {Object.entries(filters).map(([key, value]) => {
              if (!value) return null;
              return (
                <Chip
                  key={key}
                  label={`${key}: ${value}`}
                  onDelete={() => handleFilterDelete(key)}
                  sx={{
                    bgcolor: alpha(theme.palette.primary.main, 0.08),
                    color: theme.palette.primary.main,
                    '& .MuiChip-deleteIcon': {
                      color: theme.palette.primary.main,
                    },
                    fontWeight: 500,
                  }}
                />
              );
            })}
          </Stack>
        </Box>

        {/* Main Content */}
        <Box sx={{ position: 'relative' }}>
          {loading ? (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '400px',
              }}
            >
              <CircularProgress />
            </Box>
          ) : fetchError ? (
            <Paper
              sx={{
                p: 3,
                textAlign: 'center',
                bgcolor: alpha(theme.palette.error.main, 0.05),
                border: `1px solid ${alpha(theme.palette.error.main, 0.1)}`,
                borderRadius: 2,
              }}
            >
              <Typography color="error.main">{fetchError}</Typography>
            </Paper>
          ) : (
            <>
              <Box sx={{ mb: 3 }}>
                {viewMode === 'table' ? renderTable() : renderGrid()}
              </Box>
              
              <Box 
                sx={{ 
                  display: 'flex', 
                  justifyContent: 'center',
                  mt: 4,
                }}
              >
                <Pagination
                  count={totalPages}
                  page={page}
                  onChange={handlePageChange}
                  color="primary"
                  size="large"
                  showFirstButton
                  showLastButton
                  siblingCount={1}
                  boundaryCount={1}
                  sx={{
                    '& .MuiPaginationItem-root': {
                      borderRadius: 1,
                    },
                  }}
                />
              </Box>
            </>
          )}
        </Box>

        {/* Filters Dialog */}
        <FilterDialog />

        {/* Render Export Menu */}
        <Menu
          anchorEl={anchorEl}
          open={exportMenuOpen}
          onClose={handleExportMenuClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
        >
          <MenuItem onClick={() => handleExport('csv')}>
            <ListItemIcon>
              <FileDownloadIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Export as CSV</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => handleExport('pdf')}>
            <ListItemIcon>
              <FileDownloadIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Export as PDF</ListItemText>
          </MenuItem>
        </Menu>

        {/* Render View Report Dialog */}
        <Dialog
          open={viewReportOpen}
          onClose={handleCloseReport}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle sx={{ 
            pb: 2, 
            display: 'flex', 
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <Typography variant="h6">Report Details</Typography>
            <IconButton 
              onClick={handleCloseReport}
              size="small"
              sx={{
                color: theme.palette.text.secondary,
                '&:hover': { color: theme.palette.text.primary }
              }}
            >
              <CloseIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent dividers>
            {selectedReport && <ViewDamageReport report={selectedReport} />}
          </DialogContent>
          <DialogActions sx={{ px: 3, py: 2 }}>
            <Button onClick={handleCloseReport}>Close</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </>
  );
}

export default Reports;