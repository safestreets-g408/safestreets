import React, { useState, useEffect} from 'react';
import { useTheme, alpha } from '@mui/material/styles';
import { 
  Box, Paper, Typography, Grid, TextField, MenuItem, 
  Button, FormControl, InputLabel, Select, Chip, Stack,
  Card, CardContent, IconButton,
  Pagination, Menu, ListItemIcon, ListItemText, CircularProgress,
  Dialog, DialogTitle, DialogContent, DialogActions,
  Alert, Snackbar
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import GridViewIcon from '@mui/icons-material/GridView';
import TableRowsIcon from '@mui/icons-material/TableRows';
import CloseIcon from '@mui/icons-material/Close';
import AssignmentIcon from '@mui/icons-material/Assignment';
import BusinessIcon from '@mui/icons-material/Business';
import ViewDamageReport from '../components/reports/ViewDamageReport';
import ReportActions from '../components/reports/ReportActions';
import AiReportsDialog from '../components/dashboard/AiReportsDialog';
import CreateDamageReportDialog from '../components/dashboard/CreateDamageReportDialog';
import { useAuth } from '../hooks/useAuth';
import { useTenant } from '../context/TenantContext';
import { api } from '../utils/api';
import { API_BASE_URL, TOKEN_KEY, API_ENDPOINTS } from '../config/constants';

// Professional color palette - using theme instead for dark mode compatibility

function Reports() {
  const theme = useTheme();
  const { user } = useAuth();
  const { tenants } = useTenant();
  const isSuperAdmin = user?.role === 'super-admin';
  
  // Define colors based on the theme
  const colors = {
    primary: theme.palette.primary.main,
    primaryDark: theme.palette.primary.dark,
    secondary: theme.palette.secondary.main,
    success: theme.palette.success.main,
    warning: theme.palette.warning.main,
    error: theme.palette.error.main,
    surface: theme.palette.background.paper,
    border: theme.palette.divider,
    text: {
      primary: theme.palette.text.primary,
      secondary: theme.palette.text.secondary
    }
  };
  
  // Tenant selection state for super admin
  const [selectedTenant, setSelectedTenant] = useState('');
  const [allTenantsReports, setAllTenantsReports] = useState(false);
  
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

  // Snackbar state
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success'); // success, error, warning, info

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

        let endpoint = '/damage/reports';
        let data = [];

        // For super admin, fetch reports based on tenant selection
        if (isSuperAdmin) {
          if (selectedTenant) {
            // Fetch reports for specific tenant
            endpoint = `/admin/tenants/${selectedTenant}/reports`;
            data = await api.get(endpoint);
          } else if (allTenantsReports) {
            // Fetch reports from all tenants
            const tenantsData = tenants || [];
            const reportPromises = tenantsData.map(async (tenant) => {
              try {
                const tenantReports = await api.get(`/admin/tenants/${tenant._id}/reports`);
                return tenantReports.map(report => ({
                  ...report,
                  tenantInfo: {
                    id: tenant._id,
                    name: tenant.name,
                    organizationName: tenant.organizationName
                  }
                }));
              } catch (error) {
                console.error(`Error fetching reports for tenant ${tenant.name}:`, error);
                return [];
              }
            });
            
            const allReports = await Promise.all(reportPromises);
            data = allReports.flat();
          } else {
            // No tenant selected, show empty or prompt to select
            data = [];
          }
        } else {
          // Regular admin - use existing endpoint with filters
          const queryParams = new URLSearchParams();
          if (filters.dateFrom) queryParams.append('startDate', filters.dateFrom);
          if (filters.dateTo) queryParams.append('endDate', filters.dateTo);
          if (filters.severity) queryParams.append('severity', filters.severity);
          if (filters.region) queryParams.append('region', filters.region);
          if (filters.priority) queryParams.append('priority', filters.priority);
          
          const query = queryParams.toString();
          endpoint = `/damage/reports${query ? `?${query}` : ''}`;
          data = await api.get(endpoint);
        }

        setReports(data);
      } catch (err) {
        console.error('Error fetching reports:', err);
        setFetchError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, [filters, isSuperAdmin, selectedTenant, allTenantsReports, tenants]); 

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
      // For non-super-admin users, the tenant ID should be automatically applied by the backend middleware
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
    
    // Log detailed information about the AI report
    console.log('Selected AI report:', {
      id: report._id || report.id,
      damageType: report.damageType,
      hasAnnotatedImage: !!report.annotatedImageBase64,
      imageLength: report.annotatedImageBase64?.length || 0,
      imageId: report.imageId,
      fullReport: report
    });
    
    setSelectedAiReport(report);
    setReportFormData({
      region: '',
      location: report.location?.address || `${report.location?.coordinates?.[1]}, ${report.location?.coordinates?.[0]}` || '',
      damageType: report.damageType || '',
      severity: report.severity || '',
      priority: report.priority || '',
      description: report.description || `AI-detected ${report.damageType} damage`,
      reporter: 'admin@example.com',
      aiReportId: report._id || report.id,
      assignToWorker: false
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
    setSelectedFieldWorker('');
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

  const handleFormInputChange = (e) => {
    const { name, value } = e.target;
    setReportFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFieldWorkerChange = (e) => {
    setSelectedFieldWorker(e.target.value);
    setReportFormData(prev => ({
      ...prev,
      assignToWorker: !!e.target.value
    }));
  };

  const handleCreateReport = async (data) => {
    try {
      setCreateReportLoading(true);
      setCreateReportError(null);
      
      // If passed an event object accidentally, use our formData from state
      // Otherwise use the data parameter directly (which should be the formData)
      const dataToUse = data && data.preventDefault ? reportFormData : data;
      
      // Validate that we have correct data before proceeding
      if (!dataToUse || typeof dataToUse !== 'object') {
        throw new Error('Invalid report data format');
      }

      const reportData = {
        ...dataToUse,
        assignedWorker: dataToUse.assignToWorker ? selectedFieldWorker : null,
        status: 'Pending',
        reporter: 'Admin', // Add reporter field which is required by the backend
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      // Remove any event objects or circular references before sending to API
      let dataToSend = reportData;
      if (reportData.target || reportData.currentTarget) {
        console.warn('Removing event properties from report data');
        const { target, currentTarget, ...cleanData } = reportData;
        dataToSend = cleanData;
      }

      console.log('Creating damage report with data:', dataToSend);

      let response;
      
      // Use createFromAiReport endpoint if we have a selected AI report
      // This will properly handle the image transfer from AI report to damage report
      if (selectedAiReport && (selectedAiReport._id || selectedAiReport.id)) {
        const aiReportId = selectedAiReport._id || selectedAiReport.id;
        console.log('Creating damage report from AI report:', selectedAiReport);
        console.log('AI report ID:', aiReportId);
        
        // Create the payload with all required fields including the AI report ID
        const aiPayload = {
          ...dataToSend,
          aiReportId: aiReportId
        };
        
        // Make sure the aiReportId is properly formatted
        if (typeof aiPayload.aiReportId === 'object' && aiPayload.aiReportId !== null) {
          aiPayload.aiReportId = aiPayload.aiReportId.toString();
        }
        
        console.log('Payload for create-from-ai:', aiPayload);
        
        // Use the correct endpoint path as defined in damageRoutes.js
        response = await api.post(`${API_ENDPOINTS.DAMAGE_REPORTS}/reports/create-from-ai`, aiPayload);
        
        console.log('Response from create-from-ai:', response);
        
        // Update the AI report to mark it as used
        try {
          const reportId = response?._id || response?.id || response?.report?.id;
          await api.put(`${API_ENDPOINTS.IMAGES}/reports/${selectedAiReport._id}`, {
            damageReportGenerated: true,
            damageReportId: reportId
          });
          console.log('AI report updated with damage report reference:', reportId);
        } catch (updateError) {
          console.warn('Failed to update AI report reference:', updateError);
        }
      } else {
        // Standard creation without AI report
        response = await api.post(`${API_ENDPOINTS.DAMAGE_REPORTS}/reports`, dataToSend);
      }

      // Check response for success flag
      if (response && response.success) {
        // Refresh reports list
        const updatedReports = await api.get(`${API_ENDPOINTS.DAMAGE_REPORTS}/reports`);
        setReports(updatedReports);

        handleDialogClose();
        setSnackbarOpen(true);
        setSnackbarMessage(selectedAiReport ? 'Damage report with AI image created successfully!' : 'Damage report created successfully!');
        setSnackbarSeverity('success');
        
        console.log('Damage report created:', response.report || response);
      } else {
        throw new Error(response?.message || 'Failed to create report');
      }
    } catch (error) {
      console.error('Error creating damage report:', error);
      setCreateReportError(error.message || 'Failed to create damage report');
    } finally {
      setCreateReportLoading(false);
    }
  };

  // Snackbar handler
  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') return;
    setSnackbarOpen(false);
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

  // Tenant selection handlers for super admin
  const handleTenantChange = (tenantId) => {
    setSelectedTenant(tenantId);
    setAllTenantsReports(false);
    setPage(1);
  };

  const handleAllTenantsToggle = () => {
    setAllTenantsReports(!allTenantsReports);
    setSelectedTenant('');
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
    
    // Use theme colors for the PDF (but keep it readable - always use light mode for PDFs)
    const styles = `
      body { font-family: Arial, sans-serif; margin: 20px; background-color: white; }
      h1 { color: ${colors.primary}; margin-bottom: 20px; }
      .report-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
      .report-table th { background-color: ${alpha(colors.primary, 0.1)}; color: ${colors.text.primary}; }
      .report-table td, .report-table th { border: 1px solid ${colors.border}; padding: 8px; }
      .low { color: ${colors.success}; }
      .medium { color: ${colors.warning}; }
      .high, .critical { color: ${colors.error}; }
      .pending { color: ${colors.warning}; }
      .completed { color: ${colors.success}; }
      .in-progress { color: ${colors.primary}; }
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
      setActionLoading(true);
      
      // Show loading notification
      setSnackbarOpen(true);
      setSnackbarMessage('Generating PDF report...');
      setSnackbarSeverity('info');
      
      // Fetch the report data with complete details
      const reportData = await api.get(`/damage/report/${report.reportId}`);
      
      // Get the authenticated image URL
      const imageUrl = `${API_BASE_URL}/damage/report/${report.reportId}/image/before?token=${localStorage.getItem(TOKEN_KEY)}`;
      
      // Import the PDF utility dynamically to reduce initial load time
      const { generateReportPDF } = await import('../utils/pdfUtils');
      
      // Generate and download the PDF
      await generateReportPDF(reportData, imageUrl);
      
      // Show success notification
      setSnackbarOpen(true);
      setSnackbarMessage('PDF report downloaded successfully!');
      setSnackbarSeverity('success');
    } catch (error) {
      console.error('Error downloading report:', error);
      setSnackbarOpen(true);
      setSnackbarMessage('Error generating PDF report. Please try again.');
      setSnackbarSeverity('error');
    } finally {
      setActionLoading(false);
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
      PaperProps={{
        sx: {
          borderRadius: 1,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
        }
      }}
    >
      <DialogTitle sx={{ 
        py: 1.5,
        px: 2,
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid',
        borderColor: 'divider'
      }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>Filter Reports</Typography>
        <IconButton 
          onClick={() => setFiltersDialogOpen(false)}
          size="small"
          edge="end"
          sx={{ color: 'text.secondary' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 1.5, px: 2 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              size="small"
              label="Date From"
              type="date"
              value={filters.dateFrom || ''}
              onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              size="small"
              label="Date To"
              type="date"
              value={filters.dateTo || ''}
              onChange={(e) => handleFilterChange('dateTo', e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
            <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
            <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
            <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
      <DialogActions sx={{ px: 2, py: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
        <Button 
          size="small"
          onClick={() => {
            clearFilters();
            setFiltersDialogOpen(false);
          }}
          sx={{ 
            color: 'text.secondary',
            textTransform: 'none',
            fontWeight: 500,
          }}
        >
          Clear All
        </Button>
        <Button 
          variant="outlined" 
          size="small"
          onClick={() => setFiltersDialogOpen(false)}
          sx={{ 
            borderColor: 'divider', 
            color: 'text.primary',
            textTransform: 'none',
            fontWeight: 500,
          }}
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
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
      }}
    >
      <Box sx={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ backgroundColor: 'background.default' }}>
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Report ID
              </th>
              {isSuperAdmin && allTenantsReports && (
                <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                  Tenant
                </th>
              )}
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Date
              </th>
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Region
              </th>
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Type
              </th>
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Severity
              </th>
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'left', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Status
              </th>
              <th style={{ padding: '8px 12px', fontWeight: 500, color: 'text.secondary', fontSize: '0.875rem', textAlign: 'center', borderBottom: '1px solid', borderBottomColor: 'divider' }}>
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {paginatedReports().map((report, index) => (
              <tr 
                key={report._id} 
                style={{ 
                  cursor: 'pointer',
                  borderBottom: '1px solid',
                  borderBottomColor: 'divider',
                  backgroundColor: index % 2 === 0 ? 'rgba(0, 0, 0, 0.01)' : 'inherit',
                  transition: 'background-color 0.2s',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 0, 0, 0.04)'
                  }
                }} 
                onClick={(e) => {
                  if (!e.target.closest('td:last-child')) {
                    handleViewReport(report);
                  }
                }}
              >
                <td style={{ padding: '8px 12px', fontSize: '0.875rem', color: 'text.primary' }}>
                  <Typography variant="body2" fontWeight="500" color="primary">
                    {report.reportId}
                  </Typography>
                </td>
                {isSuperAdmin && allTenantsReports && (
                  <td style={{ padding: '8px 12px', fontSize: '0.875rem' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <BusinessIcon sx={{ fontSize: '0.875rem', color: 'text.secondary' }} />
                      <Typography variant="body2">
                        {report.tenantInfo?.organizationName || report.tenantInfo?.name || 'Unknown'}
                      </Typography>
                    </Box>
                  </td>
                )}
                <td style={{ padding: '8px 12px', fontSize: '0.875rem', color: 'text.secondary' }}>
                  {new Date(report.createdAt).toLocaleDateString()}
                </td>
                <td style={{ padding: '8px 12px', fontSize: '0.875rem', color: 'text.secondary' }}>{report.region}</td>
                <td style={{ padding: '8px 12px', fontSize: '0.875rem', color: 'text.secondary' }}>{report.damageType}</td>
                <td style={{ padding: '8px 12px', fontSize: '0.875rem' }}>
                  <Chip 
                    label={report.severity}
                    color={getSeverityColor(report.severity)}
                    size="small"
                    variant="outlined"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      height: '22px',
                      borderRadius: '4px',
                      '.MuiChip-label': { px: 1 }
                    }}
                  />
                </td>
                <td style={{ padding: '8px 12px', fontSize: '0.875rem' }}>
                  <Chip 
                    label={report.status}
                    color={getStatusColor(report.status)}
                    size="small"
                    variant="outlined"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      height: '22px',
                      borderRadius: '4px',
                      '.MuiChip-label': { px: 1 }
                    }}
                  />
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'center' }}>
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
                <td colSpan={isSuperAdmin && allTenantsReports ? 8 : 7} style={{ textAlign: 'center', padding: '24px', color: 'text.secondary' }}>
                  <Typography variant="body2">
                    {isSuperAdmin && !selectedTenant && !allTenantsReports ? 
                      'Please select a tenant or choose "Show All Tenants" to view reports.' :
                      'No reports found.'
                    }
                  </Typography>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Box>
    </Paper>
  );

  const renderGrid = () => (
    <Grid container spacing={2}>
      {paginatedReports().length === 0 ? (
        <Grid item xs={12}>
          <Paper 
            elevation={0} 
            sx={{ 
              p: 3, 
              textAlign: 'center', 
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
            }}
          >
            <Typography variant="subtitle1" color="text.secondary" gutterBottom fontWeight={500}>
              {isSuperAdmin && !selectedTenant && !allTenantsReports ? 
                'Select a tenant to view reports' :
                'No reports found'
              }
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {isSuperAdmin && !selectedTenant && !allTenantsReports ? 
                'Please select a tenant from the dropdown above or choose "Show All Tenants".' :
                'Try adjusting your filters or check back later.'
              }
            </Typography>
          </Paper>
        </Grid>
      ) : (
        paginatedReports().map(report => (
          <Grid item xs={12} sm={6} md={4} key={report._id || report.id}>
            <Card 
              elevation={0}
              sx={{ 
                height: '100%',
                borderRadius: 1,
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                overflow: 'hidden',
                border: '1px solid',
                borderColor: 'divider',
                transition: 'border-color 0.2s',
                '&:hover': {
                  borderColor: 'primary.main',
                  cursor: 'pointer',
                },
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '3px',
                  background: getSeverityBarColor(report.severity),
                }
              }}
              onClick={() => handleViewReport(report)}
            >
              <CardContent sx={{ flex: 1, p: 2 }}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" fontWeight={500} color="primary" gutterBottom>
                    {report.reportId}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" fontSize="0.75rem">
                    {new Date(report.createdAt).toLocaleDateString()}
                  </Typography>
                  
                  {/* Show tenant info for super admin viewing all tenants */}
                  {isSuperAdmin && allTenantsReports && report.tenantInfo && (
                    <Box sx={{ 
                      mt: 1, 
                      py: 0.5, 
                      px: 1, 
                      bgcolor: 'background.default',
                      borderRadius: 0.5, 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 0.5,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}>
                      <BusinessIcon sx={{ fontSize: '0.875rem', color: 'text.secondary' }} />
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>
                        {report.tenantInfo.organizationName || report.tenantInfo.name}
                      </Typography>
                    </Box>
                  )}
                </Box>
                
                <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                  <Chip 
                    label={report.severity}
                    color={getSeverityColor(report.severity)}
                    size="small"
                    variant="outlined"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      height: '22px',
                      borderRadius: '4px',
                      '.MuiChip-label': { px: 1 }
                    }}
                  />
                  <Chip 
                    label={report.status}
                    color={getStatusColor(report.status)}
                    size="small"
                    variant="outlined"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      height: '22px',
                      borderRadius: '4px',
                      '.MuiChip-label': { px: 1 }
                    }}
                  />
                </Stack>

                <Typography variant="body2" sx={{ 
                  mb: 2, 
                  overflow: 'hidden', 
                  textOverflow: 'ellipsis', 
                  display: '-webkit-box',
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: 'vertical',
                }}>
                  {report.description}
                </Typography>

                <Box sx={{ mt: 'auto' }}>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {report.region} • {report.damageType}
                  </Typography>
                </Box>
              </CardContent>

              <Box 
                sx={{ 
                  px: 2, 
                  py: 1.5,
                  borderTop: '1px solid',
                  borderColor: 'divider',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  Priority: {report.priority || 'N/A'}
                </Typography>
                <Button
                  size="small"
                  variant="text"
                  color="primary"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleViewReport(report);
                  }}
                  sx={{
                    fontWeight: 500,
                    textTransform: 'none',
                    p: 0.5,
                  }}
                >
                  View Details
                </Button>
              </Box>
            </Card>
          </Grid>
        ))
      )}
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
      PaperProps={{
        sx: {
          borderRadius: 1,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
        }
      }}
    >
      <DialogTitle sx={{ 
        py: 1.5,
        px: 2,
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid',
        borderColor: 'divider'
      }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>Edit Report</Typography>
        <IconButton 
          onClick={() => {
            if (!actionLoading) {
              setEditReportOpen(false);
              setEditReportData(null);
            }
          }}
          size="small"
          edge="end"
          disabled={actionLoading}
          sx={{ color: 'text.secondary' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 2, px: 2 }}>
        {editReportData && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                size="small"
                label="Report ID"
                value={editReportData.reportId || ''}
                disabled
                sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                size="small"
                label="Date Created"
                value={new Date(editReportData.createdAt).toLocaleString() || ''}
                disabled
                sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
              <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
              <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
              <FormControl fullWidth size="small" sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
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
                size="small"
                label="Region"
                value={editReportData.region || ''}
                onChange={(e) => setEditReportData({...editReportData, region: e.target.value})}
                disabled={actionLoading}
                sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                size="small"
                label="Location"
                value={editReportData.location || ''}
                onChange={(e) => setEditReportData({...editReportData, location: e.target.value})}
                disabled={actionLoading}
                sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                size="small"
                label="Description"
                multiline
                rows={3}
                value={editReportData.description || ''}
                onChange={(e) => setEditReportData({...editReportData, description: e.target.value})}
                disabled={actionLoading}
                sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                size="small"
                label="Action Required"
                multiline
                rows={2}
                value={editReportData.action || ''}
                onChange={(e) => setEditReportData({...editReportData, action: e.target.value})}
                disabled={actionLoading}
                sx={{ '& .MuiOutlinedInput-root': { borderRadius: 1 } }}
              />
            </Grid>
          </Grid>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 2, py: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
        <Button 
          size="small"
          onClick={() => {
            if (!actionLoading) {
              setEditReportOpen(false);
              setEditReportData(null);
            }
          }}
          sx={{ 
            color: 'text.secondary',
            textTransform: 'none',
            fontWeight: 500,
          }}
          disabled={actionLoading}
        >
          Cancel
        </Button>
        <Button 
          variant="outlined"
          size="small"
          onClick={handleSaveEdit}
          disabled={actionLoading}
          startIcon={actionLoading ? <CircularProgress size={16} color="inherit" /> : null}
          sx={{ 
            borderColor: 'divider',
            color: 'text.primary',
            textTransform: 'none',
            fontWeight: 500,
          }}
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
      maxWidth="xs"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 1,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
        }
      }}
    >
      <DialogTitle sx={{ 
        py: 1.5,
        px: 2,
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid',
        borderColor: 'divider'
      }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>Confirm Delete</Typography>
        <IconButton 
          onClick={() => {
            if (!actionLoading) {
              setDeleteDialogOpen(false);
              setReportToDelete(null);
            }
          }}
          edge="end"
          size="small"
          disabled={actionLoading}
          sx={{ color: 'text.secondary' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ py: 2, px: 2 }}>
        <Typography variant="body2">
          Are you sure you want to delete the report <strong>{reportToDelete?.reportId}</strong>? This action cannot be undone.
        </Typography>
      </DialogContent>
      <DialogActions sx={{ px: 2, py: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
        <Button 
          size="small"
          onClick={() => {
            if (!actionLoading) {
              setDeleteDialogOpen(false);
              setReportToDelete(null);
            }
          }}
          sx={{ 
            color: 'text.secondary',
            textTransform: 'none',
            fontWeight: 500,
          }}
          disabled={actionLoading}
        >
          Cancel
        </Button>
        <Button 
          variant="outlined"
          size="small"
          color="error"
          onClick={confirmDeleteReport}
          disabled={actionLoading}
          startIcon={actionLoading ? <CircularProgress size={16} color="inherit" /> : null}
          sx={{ 
            borderColor: 'error.main',
            textTransform: 'none',
            fontWeight: 500,
          }}
        >
          {actionLoading ? 'Deleting...' : 'Delete'}
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <>
      <Box sx={{ py: 2 }}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            justifyContent: 'space-between',
            alignItems: { xs: 'stretch', md: 'center' },
            mb: 3,
            gap: 1.5,
          }}
        >
          {/* Super Admin Tenant Selection */}
          {isSuperAdmin && (
            <Box sx={{ mb: { xs: 2, md: 0 } }}>
              <Typography variant="subtitle1" sx={{ mb: 1.5, fontWeight: 500 }}>
                Tenant Reports
              </Typography>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems="center">
                <FormControl size="small" sx={{ minWidth: 200, '& .MuiOutlinedInput-root': { borderRadius: 1 } }}>
                  <InputLabel>Select Tenant</InputLabel>
                  <Select
                    value={selectedTenant}
                    label="Select Tenant"
                    onChange={(e) => handleTenantChange(e.target.value)}
                    disabled={allTenantsReports}
                  >
                    <MenuItem value="">
                      <em>Choose a tenant...</em>
                    </MenuItem>
                    {(tenants || []).map((tenant) => (
                      <MenuItem key={tenant._id} value={tenant._id}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <BusinessIcon sx={{ fontSize: '0.875rem', color: 'text.secondary' }} />
                          <Typography variant="body2">
                            {tenant.organizationName || tenant.name}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <Button
                  variant={allTenantsReports ? 'outlined' : 'text'}
                  size="small"
                  onClick={handleAllTenantsToggle}
                  sx={{
                    borderColor: allTenantsReports ? 'primary.main' : 'transparent',
                    color: allTenantsReports ? 'primary.main' : 'text.secondary',
                    textTransform: 'none',
                    fontWeight: 500,
                    px: 2,
                  }}
                >
                  {allTenantsReports ? 'Showing All Tenants' : 'Show All Tenants'}
                </Button>
              </Stack>
              
              {/* Current Selection Info */}
              {(selectedTenant || allTenantsReports) && (
                <Box sx={{ 
                  mt: 2, 
                  p: 2, 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(37, 99, 235, 0.1)' : '#f0f9ff',
                  borderRadius: 2, 
                  border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(37, 99, 235, 0.3)' : '#bfdbfe'}`
                }}>
                  <Typography variant="body2" color="primary">
                    {allTenantsReports 
                      ? `📊 Showing reports from all ${(tenants || []).length} tenants`
                      : `📋 Showing reports for: ${(tenants || []).find(t => t._id === selectedTenant)?.organizationName || 'Selected Tenant'}`
                    }
                  </Typography>
                </Box>
              )}
            </Box>
          )}

          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1.5}
            sx={{ minWidth: { sm: '400px' } }}
          >
            <Button
              variant="outlined"
              size="small"
              startIcon={<FilterListIcon fontSize="small" />}
              onClick={() => setFiltersDialogOpen(true)}
              sx={{
                borderColor: 'divider',
                color: 'text.primary',
                '&:hover': {
                  bgcolor: 'background.default',
                },
                px: 2,
                textTransform: 'none',
                fontWeight: 500,
              }}
            >
              Filters
            </Button>

            <Box sx={{ display: 'flex', gap: 0.5, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Button
                variant="text"
                onClick={() => setViewMode('grid')}
                sx={{
                  minWidth: 'auto',
                  px: 1.5,
                  borderRadius: 0,
                  color: viewMode === 'grid' ? 'primary.main' : 'text.secondary',
                  bgcolor: viewMode === 'grid' ? 'action.selected' : 'transparent',
                }}
              >
                <GridViewIcon fontSize="small" />
              </Button>
              <Button
                variant="text"
                onClick={() => setViewMode('table')}
                sx={{
                  minWidth: 'auto',
                  px: 1.5,
                  borderRadius: 0,
                  color: viewMode === 'table' ? 'primary.main' : 'text.secondary',
                  bgcolor: viewMode === 'table' ? 'action.selected' : 'transparent',
                }}
              >
                <TableRowsIcon fontSize="small" />
              </Button>
            </Box>

            <Button
              variant="outlined"
              size="small"
              startIcon={<AssignmentIcon fontSize="small" />}
              onClick={handleViewAiReports}
              sx={{
                borderColor: 'divider',
                color: 'text.primary',
                '&:hover': {
                  bgcolor: 'background.default',
                },
                px: 2,
                textTransform: 'none',
                fontWeight: 500,
              }}
            >
              AI Reports
            </Button>

            <Button
              variant="outlined"
              size="small"
              startIcon={<FileDownloadIcon fontSize="small" />}
              onClick={(e) => setAnchorEl(e.currentTarget)}
              sx={{
                borderColor: 'divider',
                color: 'text.primary',
                '&:hover': {
                  bgcolor: 'background.default',
                },
                px: 2,
                textTransform: 'none',
                fontWeight: 500,
              }}
            >
              Export
            </Button>
          </Stack>
        </Box>

        {/* Tenant Summary Stats for Super Admin */}
        {isSuperAdmin && allTenantsReports && reports.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Paper 
              elevation={0}
              sx={{ 
                p: 2, 
                borderRadius: 1, 
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 500 }}>
                Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5 }}>
                      Total Tenants
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 500 }}>
                      {(tenants || []).length}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5 }}>
                      Total Reports
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 500 }}>
                      {reports.length}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5 }}>
                      High Priority
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 500, color: 'error.main' }}>
                      {reports.filter(r => r.severity === 'HIGH' || r.severity === 'CRITICAL').length}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5 }}>
                      Pending
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 500 }}>
                      {reports.filter(r => r.status === 'Pending').length}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        )}

        {/* Active Filters */}
        <Box sx={{ mb: 2 }}>
          <Stack
            direction="row"
            spacing={0.5}
            flexWrap="wrap"
            sx={{ gap: 0.5 }}
          >
            {Object.entries(filters).map(([key, value]) => {
              if (!value) return null;
              return (
                <Chip
                  key={key}
                  size="small"
                  label={`${key}: ${value}`}
                  onDelete={() => handleFilterDelete(key)}
                  variant="outlined"
                  sx={{
                    borderColor: 'divider',
                    color: 'text.primary',
                    '& .MuiChip-deleteIcon': {
                      color: 'text.secondary',
                      fontSize: '0.75rem',
                    },
                    fontSize: '0.75rem',
                    height: '24px',
                    fontWeight: 500,
                    borderRadius: 0.5,
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
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '300px',
                gap: 2
              }}
            >
              <CircularProgress size={40} thickness={4} sx={{ color: 'primary.main' }} />
              <Typography variant="body2" color="text.secondary">
                {isSuperAdmin && allTenantsReports 
                  ? 'Loading reports from all tenants...' 
                  : isSuperAdmin && selectedTenant 
                    ? `Loading reports for ${(tenants || []).find(t => t._id === selectedTenant)?.organizationName || 'selected tenant'}...`
                    : 'Loading reports...'
                }
              </Typography>
            </Box>
          ) : fetchError ? (
            <Paper
              elevation={0}
              sx={{
                p: 2,
                textAlign: 'center',
                border: '1px solid',
                borderColor: 'error.light',
                borderRadius: 1,
                backgroundColor: 'rgba(244, 67, 54, 0.05)',
                my: 2,
              }}
            >
              <Typography variant="body2" color="error.main">{fetchError}</Typography>
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
                  mt: 3,
                }}
              >
                <Pagination
                  count={totalPages}
                  page={page}
                  onChange={handlePageChange}
                  color="primary"
                  size="small"
                  showFirstButton
                  showLastButton
                  siblingCount={1}
                  boundaryCount={1}
                  sx={{
                    '& .MuiPaginationItem-root': {
                      borderRadius: 1,
                      minWidth: '30px',
                      height: '30px',
                    },
                    '& .MuiPaginationItem-page.Mui-selected': {
                      fontWeight: 600,
                    }
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
          PaperProps={{
            elevation: 1,
            sx: { 
              minWidth: 150,
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider',
              boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.1)'
            }
          }}
        >
          <MenuItem 
            onClick={() => handleExport('csv')}
            sx={{ 
              py: 1, 
              fontSize: '0.875rem',
              '&:hover': { backgroundColor: 'background.default' } 
            }}
          >
            <ListItemIcon sx={{ minWidth: 36 }}>
              <FileDownloadIcon fontSize="small" sx={{ fontSize: '1rem', color: 'text.secondary' }} />
            </ListItemIcon>
            <ListItemText primary="Export as CSV" primaryTypographyProps={{ fontSize: '0.875rem' }} />
          </MenuItem>
          <MenuItem 
            onClick={() => handleExport('pdf')}
            sx={{ 
              py: 1, 
              fontSize: '0.875rem',
              '&:hover': { backgroundColor: 'background.default' } 
            }}
          >
            <ListItemIcon sx={{ minWidth: 36 }}>
              <FileDownloadIcon fontSize="small" sx={{ fontSize: '1rem', color: 'text.secondary' }} />
            </ListItemIcon>
            <ListItemText primary="Export as PDF" primaryTypographyProps={{ fontSize: '0.875rem' }} />
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
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSnackbarOpen(false)} 
          severity={snackbarSeverity} 
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </>
  );
}

export default Reports;