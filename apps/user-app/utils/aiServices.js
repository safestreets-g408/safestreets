import { getBaseUrl } from '../config';
import { getAuthToken } from './auth';


const AI_SERVER_BASE_URL = 'http://192.168.199.27:5000'; 

class AIServices {
  constructor() {
    this.backendBaseUrl = null;
    this.initializeBaseUrl();
  }

  async initializeBaseUrl() {
    try {
      this.backendBaseUrl = await getBaseUrl();
    } catch (error) {
      console.error('Error getting base URL:', error);
      this.backendBaseUrl = 'http://localhost:5030/api';
    }
  }


  async imageToBase64(imageUri) {
    try {
      const response = await fetch(imageUri);
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = reader.result.split(',')[1]; 
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('Error converting image to base64:', error);
      throw new Error('Failed to process image for AI analysis');
    }
  }

  async classifyDamage(imageUri) {
    try {
      console.log('Starting damage classification...');
      
      // Convert image to base64
      const base64Image = await this.imageToBase64(imageUri);
      
      const response = await fetch(`${AI_SERVER_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image
        })
      });

      if (!response.ok) {
        throw new Error(`AI server error: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'AI analysis failed');
      }

      console.log('Damage classification successful:', result.prediction);
      
      // Validate annotated image if it exists
      let validatedAnnotatedImage = null;
      if (result.annotated_image) {
        if (typeof result.annotated_image === 'string' && 
            result.annotated_image.length > 500 &&
            !/[^A-Za-z0-9+/=]/.test(result.annotated_image)) {
          validatedAnnotatedImage = result.annotated_image;
          console.log('Valid annotated image received, length:', validatedAnnotatedImage.length);
        } else {
          console.warn('Invalid annotated image format received from server');
        }
      } else {
        console.log('No annotated image received from server');
      }
      
      return {
        damageClass: result.prediction,
        annotatedImage: validatedAnnotatedImage,
        success: true
      };
    } catch (error) {
      console.error('Error in damage classification:', error);
      throw error;
    }
  }


  async generateDamageSummary(damageDetails) {
    try {
      console.log('Generating damage summary...');
      
      const { location, damageType, severity, priority } = damageDetails;
      
      // Validate required fields
      if (!location || !damageType || !severity || !priority) {
        throw new Error('Missing required details for summary generation');
      }

      // Try backend first (includes tenant context)
      if (this.backendBaseUrl) {
        try {
          const token = await getAuthToken();
          const response = await fetch(`${this.backendBaseUrl}/ai/generate-summary`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              ...(token && { 'Authorization': `Bearer ${token}` })
            },
            body: JSON.stringify({
              location,
              damageType,
              severity,
              priority
            })
          });

          if (response.ok) {
            const result = await response.json();
            if (result.success && result.summary) {
              return {
                summary: result.summary,
                success: true
              };
            }
          }
        } catch (backendError) {
          console.log('Backend AI service failed, trying direct AI server:', backendError.message);
        }
      }

      // Fallback to direct AI server call
      const response = await fetch(`${AI_SERVER_BASE_URL}/generate-summary`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          location,
          damageType,
          severity,
          priority
        })
      });

      if (!response.ok) {
        throw new Error(`AI server error: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Summary generation failed');
      }

      console.log('Damage summary generated successfully');
      
      return {
        summary: result.summary,
        success: true
      };
    } catch (error) {
      console.error('Error generating damage summary:', error);
      throw error;
    }
  }

  async analyzeRoadDamage(imageUri, locationDetails) {
    try {
      console.log('Starting complete AI analysis...');
      
      // Step 1: Classify the damage
      const classificationResult = await this.classifyDamage(imageUri);
      
      // Step 2: Map damage class to readable format
      const damageMapping = {
        'D00': { type: 'Longitudinal Crack', severity: 'Low' },
        'D10': { type: 'Transverse Crack', severity: 'Low' },
        'D20': { type: 'Alligator Crack', severity: 'Medium' },
        'D30': { type: 'Pothole', severity: 'High' },
        'D40': { type: 'White Line Blur', severity: 'Low' },
        'D43': { type: 'Cross Walk Blur', severity: 'Medium' },
        'D44': { type: 'White Line Blur', severity: 'Medium' },
        'D50': { type: 'Manhole Cover', severity: 'Medium' }
      };

      const damageInfo = damageMapping[classificationResult.damageClass] || {
        type: 'Road Damage',
        severity: 'Medium'
      };

      // Ensure we have a valid location string
      const locationString = locationDetails?.formattedAddress || 'Road location';
      console.log(`Using location for AI summary: "${locationString}"`);
      
      // Create default summary in case the AI summary generation fails
      const defaultSummary = `${damageInfo.type} detected at ${locationString}. This damage has a ${damageInfo.severity.toLowerCase()} severity level that requires attention.`;

      let summaryResult;
      try {
        // Step 3: Generate summary with classification results (with timeout)
        const summaryPromise = this.generateDamageSummary({
          location: locationString,
          damageType: damageInfo.type,
          severity: damageInfo.severity,
          priority: damageInfo.severity === 'High' ? 'High' : 
                   damageInfo.severity === 'Medium' ? 'Medium' : 'Low'
        });
        
        // 10 second timeout for summary generation
        summaryResult = await Promise.race([
          summaryPromise,
          new Promise((resolve) => setTimeout(() => {
            console.log('Summary generation timed out, using default summary');
            resolve({ summary: defaultSummary, success: true });
          }, 10000))
        ]);
      } catch (summaryError) {
        console.error('Error generating summary, using default:', summaryError);
        summaryResult = { summary: defaultSummary, success: true };
      }

      return {
        classification: {
          damageClass: classificationResult.damageClass,
          damageType: damageInfo.type,
          severity: damageInfo.severity,
          annotatedImage: classificationResult.annotatedImage
        },
        summary: summaryResult.summary || defaultSummary,
        success: true
      };
    } catch (error) {
      console.error('Error in complete AI analysis:', error);
      throw error;
    }
  }
}

export const aiServices = new AIServices();
