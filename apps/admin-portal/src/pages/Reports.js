import React, { useState, useEffect} from 'react';
import { 
  Box, Paper, Typography, Grid, TextField, MenuItem, 
  Button, FormControl, InputLabel, Select, Chip, Stack,
  Card, CardContent, IconButton,
  Pagination, Menu, ListItemIcon, ListItemText, CircularProgress,
  Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import GridViewIcon from '@mui/icons-material/GridView';
import TableRowsIcon from '@mui/icons-material/TableRows';
import CloseIcon from '@mui/icons-material/Close';
import AssignmentIcon from '@mui/icons-material/Assignment';
import ViewDamageReport from '../components/reports/ViewDamageReport';
import ReportActions from '../components/reports/ReportActions';
import AiReportsDialog from '../components/dashboard/AiReportsDialog';
import CreateDamageReportDialog from '../components/dashboard/CreateDamageReportDialog';
import { api } from '../utils/api';
import { API_BASE_URL, TOKEN_KEY, API_ENDPOINTS } from '../config/constants';

// Professional color palette
const colors = {
  primary: '#2563eb',
  primaryDark: '#1d4ed8',
  secondary: '#64748b',
  success: '#059669',
  warning: '#d97706',
  error: '#dc2626',
  surface: '#ffffff',
  border: '#e2e8f0',
  text: {
    primary: '#1e293b',
    secondary: '#64748b'
  }
};

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
  const [sortField] = useState('createdAt');
  const [sortDirection] = useState('desc');
  const [anchorEl, setAnchorEl] = useState(null);
  const [filtersDialogOpen, setFiltersDialogOpen] = useState(false);
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [viewReportOpen, setViewReportOpen] = useState(false);
  const [editReportOpen, setEditReportOpen] = useState(false);
  const [editReportData, setEditReportData] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [reportToDelete, setReportToDelete] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  // AI Reports state
  const [aiReports, setAiReports] = useState([]);
  const [aiReportsLoading, setAiReportsLoading] = useState(false);
  const [aiReportsError, setAiReportsError] = useState(null);
  const [aiReportsOpen, setAiReportsOpen] = useState(false);
  const [selectedAiReport, setSelectedAiReport] = useState(null);
  
  // Create Report state
  const [fieldWorkers, setFieldWorkers] = useState([]);
  const [selectedFieldWorker, setSelectedFieldWorker] = useState('');
  const [createReportOpen, setCreateReportOpen] = useState(false);
  const [reportFormData, setReportFormData] = useState({
    region: '',
    location: '',
    damageType: '',
    severity: '',
    priority: '',
    description: '',
    reporter: 'admin@example.com', 
    aiReportId: null, 
    assignToWorker: false
  });
  const [createReportLoading, setCreateReportLoading] = useState(false);
  const [createReportError, setCreateReportError] = useState(null);

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

  // AI Reports functions
  const fetchAiReports = async () => {
    try {
      setAiReportsLoading(true);
      setAiReportsError(null);
      const response = await api.get(`${API_ENDPOINTS.IMAGES}/reports`);
      
      let reportsData = [];
      if (Array.isArray(response)) {
        reportsData = response;
      } else if (response?.reports && Array.isArray(response.reports)) {
        reportsData = response.reports;
      } else if (response && typeof response === 'object') {
        const possibleReports = Object.values(response).find(val => Array.isArray(val));
        reportsData = Array.isArray(possibleReports) ? possibleReports : [];
      }
      
      setAiReports(reportsData || []);
    } catch (error) {
      console.error('Error fetching AI reports:', error);
      setAiReportsError(error.message || 'Failed to fetch AI reports');
      setAiReports([]);
    } finally {
      setAiReportsLoading(false);
    }
  };

  const fetchFieldWorkers = async () => {
    try {
      const workers = await api.get(`${API_ENDPOINTS.FIELD_WORKERS}`);
      const workersData = Array.isArray(workers) ? workers : 
                         (workers.fieldWorkers || workers.workers || []);
      
      setFieldWorkers(workersData);
    } catch (error) {
      console.error('Error fetching field workers:', error);
      setFieldWorkers([]);
    }
  };

  const handleViewAiReports = async () => {
    try {
      setAiReportsLoading(true);
      await Promise.all([
        fetchAiReports(),
        fetchFieldWorkers()
      ]);
      setAiReportsOpen(true);
    } catch (error) {
      console.error('Error in View AI Reports handler:', error);
      setAiReportsOpen(true);
    }
  };

  const handleSelectAiReport = (report) => {
    if (report.damageReportGenerated) {
      alert('A damage report has already been generated from this AI report. Each AI report can only generate one damage report.');
      return;
    }
    setSelectedAiReport(report);
    setReportFormData({
      ...reportFormData,
      region: reportFormData.region || '',
      location: reportFormData.location || '',
      damageType: report.damageType || '',
      severity: report.severity || '',
      priority: report.priority || '',
      description: report.description || `AI-detected ${report.damageType} damage`,
      aiReportId: report._id || report.id,
    });
    setAiReportsOpen(false);
    setCreateReportOpen(true);
  };

  const handleAiReportsClose = () => {
    setAiReportsOpen(false);
    setSelectedAiReport(null);
  };

  const handleDialogClose = () => {
    setCreateReportOpen(false);
    setSelectedAiReport(null);
    setCreateReportError(null);
    setReportFormData({
      region: '',
      location: '',
      damageType: '',
      severity: '',
      priority: '',
      description: '',
      reporter: 'admin@example.com',
      aiReportId: null,
      assignToWorker: false
    });
  };

  const handleFormInputChange = (field, value) => {
    setReportFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleFieldWorkerChange = (workerId) => {
    setSelectedFieldWorker(workerId);
  };

  const handleCreateReport = async (formData) => {
    try {
      setCreateReportLoading(true);
      setCreateReportError(null);

      const reportData = {
        ...formData,
        assignedWorker: formData.assignToWorker ? selectedFieldWorker : null,
        status: 'Pending',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      console.log('Creating damage report with data:', reportData);

      const response = await api.post('/damage/reports', reportData);
      
      if (selectedAiReport && selectedAiReport._id) {
        try {
          await api.put(`${API_ENDPOINTS.IMAGES}/reports/${selectedAiReport._id}`, {
            damageReportGenerated: true,
            damageReportId: response._id || response.id
          });
          console.log('AI report updated with damage report reference');
        } catch (updateError) {
          console.warn('Failed to update AI report reference:', updateError);
        }
      }

      // Refresh reports list
      const updatedReports = await api.get('/damage/reports');
      setReports(updatedReports);

      handleDialogClose();
      alert('Damage report created successfully!');
      
    } catch (error) {
      console.error('Error creating damage report:', error);
      setCreateReportError(error.message || 'Failed to create damage report');
    } finally {
      setCreateReportLoading(false);
    }
  };

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

  const getSeverityBarColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
        return colors.error;
      case 'high':
        return colors.error;
      case 'medium':
        return colors.warning;
      case 'low':
        return colors.success;
      default:
        return colors.secondary;
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
    const reportsToExport = filteredReports();
    
    if (reportsToExport.length === 0) {
      // TODO: Show notification that there are no reports to export
      handleExportMenuClose();
      return;
    }
    
    if (format === 'csv') {
      exportToCSV(reportsToExport);
    } else if (format === 'pdf') {
      exportToPDF(reportsToExport);
    }
    
    handleExportMenuClose();
  };
  
  const exportToCSV = (data) => {
    // Format the data for CSV
    const headers = [
      'ReportID', 
      'Date', 
      'Region', 
      'Location', 
      'Damage Type', 
      'Severity', 
      'Priority', 
      'Status',
      'Description'
    ];
    
    const csvRows = [
      headers.join(','),
      ...data.map(report => [
        report.reportId,
        new Date(report.createdAt).toLocaleDateString(),
        report.region || '',
        report.location || '',
        report.damageType || '',
        report.severity || '',
        report.priority || '',
        report.status || '',
        `"${(report.description || '').replace(/"/g, '""')}"`, // Escape quotes in CSV
      ].join(','))
    ];
    
    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `damage-reports-${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const exportToPDF = (data) => {
    // For PDF export, we'll use client-side HTML to PDF conversion
    // Create a new window with formatted content
    const newWindow = window.open('', '_blank');
    
    // Base styles for the PDF
    const styles = `
      body { font-family: Arial, sans-serif; margin: 20px; }
      h1 { color: #2563eb; margin-bottom: 20px; }
      .report-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
      .report-table th, .report-table td { border: 1px solid #e2e8f0; padding: 10px; text-align: left; }
      .report-table th { background-color: #f8fafc; color: #1e293b; }
      .low { color: #059669; }
      .medium { color: #d97706; }
      .high, .critical { color: #dc2626; }
      .pending { color: #d97706; }
      .completed { color: #059669; }
      .in-progress { color: #2563eb; }
    `;
    
    // Create HTML content
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Damage Reports - ${new Date().toLocaleDateString()}</title>
        <style>${styles}</style>
      </head>
      <body>
        <h1>Damage Reports - ${new Date().toLocaleDateString()}</h1>
        <table class="report-table">
          <thead>
            <tr>
              <th>Report ID</th>
              <th>Date</th>
              <th>Region</th>
              <th>Damage Type</th>
              <th>Severity</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${data.map(report => `
              <tr>
                <td>${report.reportId}</td>
                <td>${new Date(report.createdAt).toLocaleDateString()}</td>
                <td>${report.region || ''}</td>
                <td>${report.damageType || ''}</td>
                <td class="${report.severity?.toLowerCase() || ''}">${report.severity || ''}</td>
                <td class="${report.status?.toLowerCase().replace(' ', '-') || ''}">${report.status || ''}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
        <p><em>Generated on ${new Date().toLocaleString()}</em></p>
        <script>
          window.onload = function() { window.print(); }
        </script>
      </body>
      </html>
    `;
    
    newWindow.document.write(htmlContent);
    newWindow.document.close();
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

  const handleEditReport = (report) => {
    setEditReportData({...report});
    setEditReportOpen(true);
  };

  const handleSaveEdit = async () => {
    try {
      setActionLoading(true);
      const response = await api.put(`/damage/report/${editReportData.reportId}`, editReportData);
      
      // Update the reports state with the updated report
      setReports(reports.map(r => 
        r.reportId === response.report.reportId ? {...r, ...response.report} : r
      ));
      
      setEditReportOpen(false);
      setEditReportData(null);
    } catch (err) {
      console.error('Error updating report:', err);
      // TODO: Add error notification
    } finally {
      setActionLoading(false);
    }
  };

  const handleDeleteReport = (report) => {
    setReportToDelete(report);
    setDeleteDialogOpen(true);
  };

  const confirmDeleteReport = async () => {
    try {
      setActionLoading(true);
      await api.delete(`/damage/report/${reportToDelete.reportId}`);
      
      // Remove the deleted report from the state
      setReports(reports.filter(r => r.reportId !== reportToDelete.reportId));
      
      setDeleteDialogOpen(false);
      setReportToDelete(null);
    } catch (err) {
      console.error('Error deleting report:', err);
      // TODO: Add error notification
    } finally {
      setActionLoading(false);
    }
  };

  const handleDownloadReport = async (report) => {
    try {
      // Fetch the report data with image
      const reportData = await api.get(`/damage/report/${report.reportId}`);
      
      // Fetch the image separately
      const imageBlob = await fetch(`${API_BASE_URL}/damage/report/${report.reportId}/image/before`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem(TOKEN_KEY)}`
        }
      }).then(res => res.blob());
      
      // Create a URL for the image
      const imageUrl = URL.createObjectURL(imageBlob);
      
      // Create a formatted report for downloading
      const formattedReport = {
        reportId: reportData.reportId,
        createdAt: new Date(reportData.createdAt).toLocaleString(),
        damageType: reportData.damageType,
        severity: reportData.severity,
        priority: reportData.priority,
        region: reportData.region,
        location: reportData.location,
        description: reportData.description,
        status: reportData.status,
        action: reportData.action,
        imageUrl,
      };
      
      // Download as JSON
      const jsonString = JSON.stringify(formattedReport, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const downloadUrl = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = `report-${report.reportId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      
      // Clean up URLs
      URL.revokeObjectURL(imageUrl);
      URL.revokeObjectURL(downloadUrl);
      
    } catch (err) {
      console.error('Error downloading report:', err);
      // TODO: Add error notification
    }
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
            color: colors.text.secondary,
            '&:hover': { color: colors.text.primary }
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
        border: `1px solid ${colors.border}`,
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
                  backgroundColor: index % 2 === 0 ? '#f8fafc' : 'inherit'
                }}
              >
                <td style={{ padding: 8, fontWeight: 500, color: colors.primary }}>
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
                  <ReportActions 
                    report={report}
                    onView={handleViewReport}
                    onEdit={handleEditReport}
                    onDelete={handleDeleteReport}
                    onDownload={handleDownloadReport}
                    colors={colors}
                  />
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
              transition: 'box-shadow 0.2s ease-in-out',
              '&:hover': {
                boxShadow: '0 4px 8px rgba(0, 0, 0, 0.15)',
              },
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              overflow: 'visible',
              border: `1px solid ${colors.border}`,
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '4px',
                background: getSeverityBarColor(report.severity),
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
                borderTop: `1px solid ${colors.border}`,
                display: 'flex',
                justifyContent: 'flex-end'
              }}
            >
              <Button
                size="small"
                variant="contained"
                onClick={() => handleViewReport(report)}
                sx={{
                  bgcolor: colors.primary,
                  color: 'white',
                  '&:hover': {
                    bgcolor: colors.primaryDark,
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

  // Edit Report Dialog
  const EditReportDialog = () => (
    <Dialog 
      open={editReportOpen} 
      onClose={() => {
        if (!actionLoading) {
          setEditReportOpen(false);
          setEditReportData(null);
        }
      }}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle sx={{ 
        pb: 2, 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Typography variant="h6">Edit Report</Typography>
        <IconButton 
          onClick={() => {
            if (!actionLoading) {
              setEditReportOpen(false);
              setEditReportData(null);
            }
          }}
          size="small"
          disabled={actionLoading}
          sx={{
            color: colors.text.secondary,
            '&:hover': { color: colors.text.primary }
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {editReportData && (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Report ID"
                value={editReportData.reportId || ''}
                disabled
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Date Created"
                value={new Date(editReportData.createdAt).toLocaleString() || ''}
                disabled
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Damage Type</InputLabel>
                <Select
                  value={editReportData.damageType || ''}
                  label="Damage Type"
                  onChange={(e) => setEditReportData({...editReportData, damageType: e.target.value})}
                  disabled={actionLoading}
                >
                  {damageTypeOptions.map(option => (
                    <MenuItem key={option} value={option}>{option}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Severity</InputLabel>
                <Select
                  value={editReportData.severity || ''}
                  label="Severity"
                  onChange={(e) => setEditReportData({...editReportData, severity: e.target.value})}
                  disabled={actionLoading}
                >
                  <MenuItem value="Low">Low</MenuItem>
                  <MenuItem value="Medium">Medium</MenuItem>
                  <MenuItem value="High">High</MenuItem>
                  <MenuItem value="Critical">Critical</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={editReportData.priority || ''}
                  label="Priority"
                  onChange={(e) => setEditReportData({...editReportData, priority: e.target.value})}
                  disabled={actionLoading}
                >
                  <MenuItem value="Low">Low</MenuItem>
                  <MenuItem value="Medium">Medium</MenuItem>
                  <MenuItem value="High">High</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={editReportData.status || ''}
                  label="Status"
                  onChange={(e) => setEditReportData({...editReportData, status: e.target.value})}
                  disabled={actionLoading}
                >
                  <MenuItem value="Pending">Pending</MenuItem>
                  <MenuItem value="Assigned">Assigned</MenuItem>
                  <MenuItem value="In Progress">In Progress</MenuItem>
                  <MenuItem value="Completed">Completed</MenuItem>
                  <MenuItem value="Rejected">Rejected</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Region"
                value={editReportData.region || ''}
                onChange={(e) => setEditReportData({...editReportData, region: e.target.value})}
                disabled={actionLoading}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Location"
                value={editReportData.location || ''}
                onChange={(e) => setEditReportData({...editReportData, location: e.target.value})}
                disabled={actionLoading}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={4}
                value={editReportData.description || ''}
                onChange={(e) => setEditReportData({...editReportData, description: e.target.value})}
                disabled={actionLoading}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Action Required"
                multiline
                rows={2}
                value={editReportData.action || ''}
                onChange={(e) => setEditReportData({...editReportData, action: e.target.value})}
                disabled={actionLoading}
              />
            </Grid>
          </Grid>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button 
          onClick={() => {
            if (!actionLoading) {
              setEditReportOpen(false);
              setEditReportData(null);
            }
          }}
          color="inherit"
          disabled={actionLoading}
        >
          Cancel
        </Button>
        <Button 
          variant="contained" 
          onClick={handleSaveEdit}
          disabled={actionLoading}
          startIcon={actionLoading ? <CircularProgress size={20} color="inherit" /> : null}
        >
          {actionLoading ? 'Saving...' : 'Save Changes'}
        </Button>
      </DialogActions>
    </Dialog>
  );
  
  // Delete Confirmation Dialog
  const DeleteConfirmationDialog = () => (
    <Dialog 
      open={deleteDialogOpen} 
      onClose={() => {
        if (!actionLoading) {
          setDeleteDialogOpen(false);
          setReportToDelete(null);
        }
      }}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle sx={{ pb: 2 }}>
        <Typography variant="h6">Confirm Delete</Typography>
      </DialogTitle>
      <DialogContent>
        <Typography>
          Are you sure you want to delete the report {reportToDelete?.reportId}? This action cannot be undone.
        </Typography>
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button 
          onClick={() => {
            if (!actionLoading) {
              setDeleteDialogOpen(false);
              setReportToDelete(null);
            }
          }}
          color="inherit"
          disabled={actionLoading}
        >
          Cancel
        </Button>
        <Button 
          variant="contained" 
          color="error"
          onClick={confirmDeleteReport}
          disabled={actionLoading}
          startIcon={actionLoading ? <CircularProgress size={20} color="inherit" /> : null}
        >
          {actionLoading ? 'Deleting...' : 'Delete'}
        </Button>
      </DialogActions>
    </Dialog>
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
                bgcolor: colors.primary,
                color: 'white',
                '&:hover': {
                  bgcolor: colors.primaryDark,
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
                    bgcolor: colors.primary,
                    color: 'white',
                    '&:hover': {
                      bgcolor: colors.primaryDark,
                    },
                  } : {
                    borderColor: colors.border,
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
                    bgcolor: colors.primary,
                    color: 'white',
                    '&:hover': {
                      bgcolor: colors.primaryDark,
                    },
                  } : {
                    borderColor: colors.border,
                  })
                }}
              >
                <TableRowsIcon />
              </Button>
            </Box>

            <Button
              variant="contained"
              startIcon={<AssignmentIcon />}
              onClick={handleViewAiReports}
              sx={{
                bgcolor: colors.warning,
                color: 'white',
                '&:hover': {
                  bgcolor: '#b45309',
                },
              }}
            >
              AI Reports
            </Button>

            <Button
              variant="contained"
              startIcon={<FileDownloadIcon />}
              onClick={(e) => setAnchorEl(e.currentTarget)}
              sx={{
                bgcolor: colors.success,
                '&:hover': {
                  bgcolor: '#047857',
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
                    bgcolor: colors.border,
                    color: colors.text.primary,
                    '& .MuiChip-deleteIcon': {
                      color: colors.text.secondary,
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
                bgcolor: '#fef2f2',
                border: `1px solid ${colors.error}`,
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

        {/* Edit Report Dialog */}
        <EditReportDialog />
        
        {/* Delete Confirmation Dialog */}
        <DeleteConfirmationDialog />

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
                color: colors.text.secondary,
                '&:hover': { color: colors.text.primary }
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

        {/* AI Reports Dialog */}
        <AiReportsDialog 
          open={aiReportsOpen}
          onClose={handleAiReportsClose}
          reports={aiReports}
          loading={aiReportsLoading}
          error={aiReportsError}
          onSelectReport={handleSelectAiReport}
        />

        {/* Create Damage Report Dialog */}
        <CreateDamageReportDialog 
          open={createReportOpen}
          onClose={handleDialogClose}
          onSubmit={handleCreateReport}
          selectedAiReport={selectedAiReport}
          formData={reportFormData}
          onFormChange={handleFormInputChange}
          fieldWorkers={fieldWorkers}
          onFieldWorkerChange={handleFieldWorkerChange}
          selectedFieldWorker={selectedFieldWorker}
          loading={createReportLoading}
          error={createReportError}
        />
      </Box>
    </>
  );
}

export default Reports;