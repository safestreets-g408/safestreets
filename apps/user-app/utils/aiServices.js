import { getBaseUrl } from '../config';
import { getAuthToken } from './auth';
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';

const MANUAL_OVERRIDE_IP = '192.168.23.177'; 
const AI_SERVER_PORT = '5000';

class AIServices {
  constructor() {
    this.backendBaseUrl = null;
    this.aiServerBaseUrl = null;
    this.initializeUrls();
  }

  async getAiServerUrl() {
    try {
      const storedUrl = await AsyncStorage.getItem('custom_ai_server_url');
      if (storedUrl) {
        console.log('Using custom AI server URL from storage:', storedUrl);
        return storedUrl;
      }
    } catch (error) {
      console.log('Error reading custom AI server URL from storage:', error);
    }

    // Get the Expo host URL when running in Expo Go
    const debuggerHost = Constants.expoConfig?.hostUri || Constants.manifest?.debuggerHost;
    const expoHost = debuggerHost ? debuggerHost.split(':')[0] : null;
    
    let aiServerUrl;

    if (__DEV__) {
      if (Platform.OS === 'android') {
        // For Android Emulator (10.0.2.2 maps to host's localhost)
        aiServerUrl = `http://10.0.2.2:${AI_SERVER_PORT}`;
      } else if (expoHost) {
        // For Expo Go - use the same IP that Expo server is running on
        aiServerUrl = `http://${expoHost}:${AI_SERVER_PORT}`;
      } else if (MANUAL_OVERRIDE_IP) {
        // Use manually specified IP if available
        aiServerUrl = `http://${MANUAL_OVERRIDE_IP}:${AI_SERVER_PORT}`;
      } else {
        // Fallback to localhost
        aiServerUrl = `http://localhost:${AI_SERVER_PORT}`;
      }
    } else {
      // Production URL for AI server
      aiServerUrl = 'https://ai.safestreets-prod.com';
    }

    console.log('Using AI server URL:', aiServerUrl);
    return aiServerUrl;
  }

  async initializeUrls() {
    try {
      this.backendBaseUrl = await getBaseUrl();
      this.aiServerBaseUrl = await this.getAiServerUrl();
    } catch (error) {
      console.error('Error getting base URLs:', error);
      this.backendBaseUrl = 'http://localhost:5030/api';
      this.aiServerBaseUrl = 'http://localhost:5000';
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

  // The classifyDamage method has been removed as we now use YOLOv8 for all detection


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
      if (!this.aiServerBaseUrl) {
        this.aiServerBaseUrl = await this.getAiServerUrl();
      }
      
      const response = await fetch(`${this.aiServerBaseUrl}/generate-summary`, {
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
      console.log('Starting complete AI analysis using YOLOv8...');
      
      // Step 1: Use YOLOv8 for damage detection instead of the VIT model
      const yoloResult = await this.detectWithYolov8(imageUri);
      
      // Step 2: Map detected objects to damage classes
      // Use the first detection with highest confidence or a fallback
      let damageClass = 'D00'; // Default to Longitudinal Crack
      let highestConfidence = 0;
      
      if (yoloResult && yoloResult.detections && yoloResult.detections.length > 0) {
        // Find detection with highest confidence
        for (const detection of yoloResult.detections) {
          if (detection.confidence > highestConfidence) {
            highestConfidence = detection.confidence;
            // Use the class as damage class if it starts with D (damage class format)
            if (detection.class.startsWith('D')) {
              damageClass = detection.class;
            }
          }
        }
      }
      
      const classificationResult = {
        damageClass: damageClass,
        annotatedImage: yoloResult.annotatedImage,
        success: true
      };
      
      // Step 3: Map damage class to readable format
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

  // New method to use YOLOv8 model for detection
  async detectWithYolov8(imageUri) {
    try {
      console.log('Starting YOLOv8 object detection...');
      
      // Ensure AI server URL is initialized
      if (!this.aiServerBaseUrl) {
        this.aiServerBaseUrl = await this.getAiServerUrl();
      }
      
      // Convert image to base64
      const base64Image = await this.imageToBase64(imageUri);
      
      const response = await fetch(`${this.aiServerBaseUrl}/detect-yolo`, {
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
        throw new Error(result.error || 'YOLOv8 detection failed');
      }

      console.log('YOLOv8 detection successful:', {
        detectionCount: result.detections?.length || 0,
        hasAnnotatedImage: !!result.annotated_image
      });
      
      return {
        detections: result.detections || [],
        annotatedImage: result.annotated_image,
        count: result.detections?.length || 0,
        success: true
      };
    } catch (error) {
      console.error('Error in YOLOv8 detection:', error);
      throw error;
    }
  }

  // Road classification is now handled directly by the AI server
}

export const aiServices = new AIServices();
