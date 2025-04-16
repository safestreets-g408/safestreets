import React, { useState } from 'react';
import { 
  Box, Paper, Typography, Grid, TextField, MenuItem, 
  Button, FormControl, InputLabel, Select, Chip, Stack
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import ReportDataTable from '../components/ReportDataTable';

function Reports() {
  const [filters, setFilters] = useState({
    dateFrom: null,
    dateTo: null,
    severity: '',
    region: '',
    damageType: '',
  });

  const handleFilterChange = (field, value) => {
    setFilters({
      ...filters,
      [field]: value
    });
  };

  const clearFilters = () => {
    setFilters({
      dateFrom: null,
      dateTo: null,
      severity: '',
      region: '',
      damageType: '',
    });
  };

  const severityOptions = ['Low', 'Medium', 'High', 'Critical'];
  const regionOptions = ['North', 'South', 'East', 'West', 'Central'];
  const damageTypeOptions = ['Structural', 'Electrical', 'Plumbing', 'Flooding', 'Fire', 'Other'];

  return (
    <>
      <Typography variant="h4" gutterBottom>
        Damage Reports
      </Typography>
      
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <FilterListIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Filters</Typography>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
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
          <Grid item xs={12} sm={6} md={3}>
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
        </Grid>
        
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Stack direction="row" spacing={1}>
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
          </Stack>
          
          <Button variant="outlined" onClick={clearFilters}>
            Clear Filters
          </Button>
        </Box>
      </Paper>
      
      <Paper sx={{ p: 2 }}>
        <ReportDataTable filters={filters} />
      </Paper>
    </>
  );
}

export default Reports;