import React from 'react';
import { Chip, useTheme, alpha } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import PendingIcon from '@mui/icons-material/Pending';
import BuildIcon from '@mui/icons-material/Build';

const statusConfig = {
  completed: {
    color: 'success',
    icon: CheckCircleIcon,
    label: 'Completed'
  },
  pending: {
    color: 'warning',
    icon: PendingIcon,
    label: 'Pending'
  },
  inProgress: {
    color: 'info',
    icon: BuildIcon,
    label: 'In Progress'
  },
  failed: {
    color: 'error',
    icon: ErrorIcon,
    label: 'Failed'
  },
  critical: {
    color: 'error',
    icon: ErrorIcon,
    label: 'Critical'
  },
  warning: {
    color: 'warning',
    icon: WarningIcon,
    label: 'Warning'
  }
};

const StatusChip = ({
  status,
  customLabel,
  size = 'small',
  sx = {}
}) => {
  const theme = useTheme();
  const config = statusConfig[status] || statusConfig.pending;
  const Icon = config.icon;

  return (
    <Chip
      icon={<Icon fontSize="small" />}
      label={customLabel || config.label}
      color={config.color}
      size={size}
      sx={{
        fontWeight: 600,
        borderRadius: 1,
        '& .MuiChip-icon': {
          fontSize: '1rem',
        },
        boxShadow: `0 2px 8px ${alpha(theme.palette[config.color].main, 0.2)}`,
        ...sx
      }}
    />
  );
};

export default StatusChip; 