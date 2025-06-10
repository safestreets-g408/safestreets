import React from 'react';
import { Box, Typography, Breadcrumbs, Link, useTheme } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';

const PageHeader = ({
  title,
  subtitle,
  breadcrumbs = [],
  action,
  sx = {}
}) => {
  const theme = useTheme();

  return (
    <Box 
      sx={{ 
        mb: 4,
        position: 'relative',
        ...sx 
      }}
    >
      {breadcrumbs.length > 0 && (
        <Breadcrumbs
          separator={
            <NavigateNextIcon 
              fontSize="small" 
              sx={{ 
                color: theme.palette.text.disabled,
                fontSize: '1.2rem' 
              }} 
            />
          }
          sx={{ 
            mb: 2.5,
            '& .MuiBreadcrumbs-li': {
              display: 'flex',
              alignItems: 'center',
            }
          }}
        >
          {breadcrumbs.map((crumb, index) => {
            const isLast = index === breadcrumbs.length - 1;
            
            if (isLast || !crumb.path) {
              return (
                <Typography
                  key={crumb.label}
                  color="text.secondary"
                  variant="body2"
                  sx={{ 
                    fontSize: '0.875rem',
                    fontWeight: isLast ? 500 : 400 
                  }}
                >
                  {crumb.label}
                </Typography>
              );
            }

            return (
              <Link
                key={crumb.label}
                component={RouterLink}
                to={crumb.path}
                color="inherit"
                variant="body2"
                sx={{
                  textDecoration: 'none',
                  '&:hover': {
                    textDecoration: 'underline',
                  },
                }}
              >
                {crumb.label}
              </Link>
            );
          })}
        </Breadcrumbs>
      )}

      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box>
          <Typography
            variant="h4"
            component="h1"
            gutterBottom={!!subtitle}
            sx={{
              color: theme.palette.text.primary,
              fontWeight: 600,
            }}
          >
            {title}
          </Typography>
          
          {subtitle && (
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ mt: -1 }}
            >
              {subtitle}
            </Typography>
          )}
        </Box>

        {action && (
          <Box sx={{ ml: 2 }}>
            {action}
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default PageHeader; 