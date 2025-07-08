import { getAuthToken } from './auth';
import { getBaseUrl } from '../config';

// Fetch user reports
export const fetchReports = async (params = {}) => {
  try {
    const token = await getAuthToken();
    const baseUrl = await getBaseUrl();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    // Convert params object to query string
    const queryParams = new URLSearchParams();
    Object.keys(params).forEach(key => {
      if (params[key] !== null && params[key] !== undefined) {
        queryParams.append(key, params[key]);
      }
    });
    
    const queryString = queryParams.toString();
    const url = `${baseUrl}/fieldworker/damage/reports${queryString ? `?${queryString}` : ''}`;
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch reports');
    }
    
    return data.reports;
  } catch (error) {
    console.error('Error fetching reports:', error);
    throw error;
  }
};

// Fetch single report details
export const fetchReportDetails = async (reportId) => {
  try {
    const token = await getAuthToken();
    const baseUrl = await getBaseUrl();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${baseUrl}/fieldworker/damage/reports/${reportId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch report details');
    }
    
    return data.report;
  } catch (error) {
    console.error('Error fetching report details:', error);
    throw error;
  }
};

// Update report status
export const updateReportStatus = async (reportId, status, notes = '') => {
  try {
    const token = await getAuthToken();
    const baseUrl = await getBaseUrl();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${baseUrl}/fieldworker/damage/reports/${reportId}/status`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ status, notes }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to update report status');
    }
    
    return data.report;
  } catch (error) {
    console.error('Error updating report status:', error);
    throw error;
  }
};

// Submit new report
export const submitNewReport = async (reportData, imageUri) => {
  try {
    const token = await getAuthToken();
    const baseUrl = await getBaseUrl();

    if (!token) {
      throw new Error('Authentication required');
    }

    // Convert image to base64 if provided
    let imageBase64 = null;
    if (imageUri) {
      try {
        const response = await fetch(imageUri);
        const blob = await response.blob();

        imageBase64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
          };
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      } catch (imageError) {
        console.error('Error converting image to base64:', imageError);
        // Continue without image rather than failing completely
      }
    }

    // Prepare data for the field worker upload endpoint
    // Map severity and priority to match backend schema
    const severity = (reportData.severity || 'low').toUpperCase();
    
    // Priority mapping based on the error, AiReport model expects a number between 1-10
    // Numeric priority value for AiReport model
    const priorityNumeric = typeof reportData.priority === 'number'
      ? reportData.priority
      : typeof reportData.priority === 'string'
        ? { low: 3, medium: 5, high: 8 }[reportData.priority.toLowerCase()] || 5
        : 5;
    
    // String priority value for DamageReport model
    const priorityMap = { low: 'Low', medium: 'Medium', high: 'High' };
    const priorityString = typeof reportData.priority === 'string'
      ? priorityMap[reportData.priority.toLowerCase()] || 'Medium'
      : (reportData.priority <= 3 ? 'Low' : reportData.priority >= 8 ? 'High' : 'Medium');
    
    // Parse coordinates from string to array of numbers if needed
    let coordinates = [];
    if (reportData.location) {
      if (typeof reportData.location === 'string') {
        // Handle string format like "lat,lng"
        const [lat, lng] = reportData.location.split(',').map(Number);
        if (!isNaN(lat) && !isNaN(lng)) {
          coordinates = [lng, lat]; // MongoDB uses [longitude, latitude] order
        }
      } else if (Array.isArray(reportData.location)) {
        // Handle array format
        coordinates = reportData.location.map(coord => 
          typeof coord === 'string' ? Number(coord) : coord
        );
      }
    }
    
    // Debug the coordinates format
    console.log('Coordinates format:', coordinates);

    // Create a local report as fallback in case server submission fails
    const tempReportId = `TEMP-${Date.now()}`;
    const localReport = {
      id: tempReportId,
      reportId: tempReportId,
      damageType: reportData.damageType || 'POTHOLE',
      severity: severity,
      priority: priorityString, // Use string priority for local reports
      location: reportData.address || 'Unknown Location',
      description: reportData.description || '',
      coordinates: coordinates,
      image: imageBase64 ? '[Image data available]' : '[No image]',
      status: 'Pending',
      createdAt: new Date().toISOString(),
      reporter: 'Mobile App User',
      _isLocalOnly: true
    };
    
    // Attempt to send to server with proper formatting
    let maxAttempts = 3;
    let attempt = 0;
    let lastError = null;

    while (attempt < maxAttempts) {
      attempt++;
      try {
        console.log(`Attempt ${attempt} to submit report to server...`);
        
        // Format request body to match exactly what AiReport model expects
        const requestBody = {
          // Required core fields for AiReport model
          damageType: reportData.damageType || 'POTHOLE',
          severity: severity, // Already uppercase from earlier processing
          priority: priorityNumeric,  // Send numerical priority (1-10)
          predictionClass: reportData.damageType || 'POTHOLE', // Same as damageType
          annotatedImageBase64: imageBase64 || '', // Required field, provide empty string if no image
          
          // Location data structured as expected by the model
          location: {
            coordinates: coordinates.length === 2 ? coordinates : undefined,
            address: (() => {
              let addr = reportData.location?.address || reportData.address || 'Location not specified';
              // Safety check: ensure address doesn't contain AI analysis text
              if (addr && (addr.includes('Road damage report') || addr.includes('observed on') || addr.length > 200)) {
                console.warn('Detected AI analysis text in address field, using fallback');
                addr = 'Address data corrupted';
              }
              return addr;
            })()
          },
          
          // Additional fields for compatibility
          description: reportData.description || '',
          region: reportData.region || 'Default Region'
        };
        
        console.log('Sending request body with structure:', {
          ...requestBody,
          annotatedImageBase64: imageBase64 ? `[${imageBase64.length} chars]` : '[empty]'
        });
        
        console.log(`Sending request with all required fields to AI reports endpoint...`);
        
        // Use the AI-specific endpoint with the proper AiReport fields
        const response = await fetch(`${baseUrl}/fieldworker/damage/ai-reports/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
        
        console.log('Server response status:', response.status);
        
        // Process response
        if (response.headers.get('content-type')?.includes('application/json')) {
          const data = await response.json();
          console.log('Server response data:', data);
          
          if (response.ok) {
            console.log('Server accepted the report!');
            return data.report;
          } else {
            lastError = new Error(data.message || 'Server error');
            console.error(`Error on attempt ${attempt}:`, data);
            
            // If we're on the last attempt and still failing with the AI endpoint,
            // try the standard damage report endpoint as fallback
            if (attempt === maxAttempts) {
              try {
                console.log('Trying standard damage report endpoint as fallback');
                
                // Create simpler request for standard endpoint
                const fallbackRequestBody = {
                  reportId: `MOBILE-${Date.now()}`,
                  damageType: reportData.damageType || 'POTHOLE',
                  severity: severity,
                  priority: priorityString,  // Use string format for DamageReport
                  location: reportData.address || 'Location not specified',
                  description: reportData.description || '',
                  region: reportData.region || 'Default Region',
                  action: 'Pending Review',
                  reporter: 'Mobile App User',
                  coordinates: coordinates.length === 2 ? coordinates : undefined,
                  imageData: imageBase64
                };
                
                // Try the standard endpoint
                const fallbackResponse = await fetch(`${baseUrl}/fieldworker/damage/reports/upload`, {
                  method: 'POST',
                  headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify(fallbackRequestBody),
                });
                
                if (fallbackResponse.ok) {
                  const fallbackData = await fallbackResponse.json();
                  console.log('Fallback endpoint accepted the report!');
                  return fallbackData.report;
                }
              } catch (fallbackError) {
                console.error('Fallback attempt also failed:', fallbackError);
              }
            }
          }
        } else {
          lastError = new Error('Server returned non-JSON response');
          console.error(`Error on attempt ${attempt}: Non-JSON response`);
        }
      } catch (networkError) {
        lastError = networkError;
        console.error(`Network error on attempt ${attempt}:`, networkError.message);
      }
      
      // Wait a short time before retrying (exponential backoff)
      if (attempt < maxAttempts) {
        const delay = 1000 * Math.pow(2, attempt - 1); // 1s, 2s, 4s
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    if (lastError) {
      console.log('All server submission attempts failed, using local report as fallback');
      console.error(`Failed after ${maxAttempts} attempts:`, lastError);
    }
    
    // Return the local report as fallback
    return localReport;
  } catch (error) {
    console.error('Error submitting report:', error);
    
    // Return the local report as a fallback
    console.log('Using local report due to exception');
    return localReport;
  }
};

// Submit AI report with YOLOv8 detection results
export const submitAiReport = async (aiReportData) => {
  try {
    const token = await getAuthToken();
    const baseUrl = await getBaseUrl();

    if (!token) {
      throw new Error('Authentication required');
    }

    // Process YOLOv8 results if available
    let yoloDetections = [];
    let yoloDetectionCount = 0;
    
    if (aiReportData.yoloResults) {
      // Extract YOLOv8 detection results
      yoloDetections = aiReportData.yoloResults.detections || [];
      yoloDetectionCount = aiReportData.yoloResults.detectionCount || yoloDetections.length;
      
      console.log(`Including ${yoloDetectionCount} YOLOv8 detections in report`);
      
      // Remove the temporary yoloResults field and add proper schema fields
      delete aiReportData.yoloResults;
    }
    
    // Add YOLOv8 results to the request body
    const requestBody = {
      ...aiReportData,
      yoloDetections,
      yoloDetectionCount
    };

    console.log('Submitting AI report with structure:', {
      damageType: requestBody.damageType,
      severity: requestBody.severity,
      predictionClass: requestBody.predictionClass,
      hasAnnotatedImage: !!requestBody.annotatedImageBase64,
      yoloDetectionCount: requestBody.yoloDetectionCount
    });

    const response = await fetch(`${baseUrl}/fieldworker/damage/ai-reports`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Server error: ${response.status}`);
    }

    const data = await response.json();
    return data.report;
  } catch (error) {
    console.error('Error submitting AI report:', error);
    throw error;
  }
};
