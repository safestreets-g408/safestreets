import React from 'react';
import { useTheme } from '@mui/material/styles';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  alpha,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Zoom,
} from '@mui/material';
import { CheckCircle } from '@mui/icons-material';

const ComponentCard = ({ component, index }) => {
  const theme = useTheme();
  
  return (
    <Zoom in={true} style={{ transitionDelay: `${index * 100}ms` }}>
      <Card
        elevation={0}
        sx={{
          height: '100%',
          borderRadius: '16px',
          transition: 'all 0.3s ease',
          bgcolor: alpha(theme.palette.background.paper, 0.8),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          '&:hover': {
            boxShadow: `0 10px 30px ${alpha(theme.palette.common.black, 0.1)}`,
            transform: 'translateY(-6px)',
            borderColor: alpha(theme.palette.primary.main, 0.2),
          },
        }}
      >
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box sx={{ 
              color: 'primary.main',
              mr: 1.5
            }}>
              {component.icon}
            </Box>
            
            <Typography
              variant="h6"
              fontWeight={700}
            >
              {component.title}
            </Typography>
          </Box>
          
          <Typography
            variant="body2"
            color="text.secondary"
            gutterBottom
            sx={{ mb: 2 }}
          >
            {component.description}
          </Typography>
          
          <List disablePadding>
            {component.features.map((feature, i) => (
              <ListItem
                key={i}
                disableGutters
                disablePadding
                sx={{ py: 0.5 }}
              >
                <ListItemIcon sx={{ minWidth: 30 }}>
                  <CheckCircle sx={{ fontSize: 18, color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText
                  primary={feature}
                  primaryTypographyProps={{
                    variant: 'body2',
                    fontWeight: 500,
                  }}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>
    </Zoom>
  );
};

const SystemComponents = ({ systemComponents }) => {
  return (
    <Box
      sx={{
        py: { xs: 8, md: 12 },
        position: 'relative',
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ mb: { xs: 6, md: 8 }, textAlign: 'center' }}>
          <Typography
            variant="h2"
            fontWeight={800}
            gutterBottom
            sx={{
              mb: 2,
              fontSize: { xs: '2rem', md: '2.5rem' },
            }}
          >
            System Architecture
          </Typography>
          
          <Typography
            variant="h6"
            color="text.secondary"
            sx={{
              fontWeight: 500,
              maxWidth: '800px',
              mx: 'auto',
              px: { xs: 2, md: 0 },
              lineHeight: 1.6,
            }}
          >
            SafeStreets is built with a modular, scalable architecture using modern technologies
            designed for reliability and performance.
          </Typography>
        </Box>
        
        <Grid container spacing={4}>
          {systemComponents.map((component, index) => (
            <Grid item xs={12} sm={6} key={index}>
              <ComponentCard component={component} index={index} />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default SystemComponents;
