const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);
const multer = require('multer');

// Constants
const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:5000';

/**
 * Get model information from the AI server
 */
exports.getModelInfo = async (req, res) => {
  try {
    const { type } = req.query;
    
    if (type === 'yolo') {
      // Get YOLO model information
      const response = await axios.get(`${AI_SERVER_URL}/yolo-info`);
      
      return res.status(200).json({
        success: true,
        modelInfo: response.data
      });
    } else {
      return res.status(400).json({
        success: false,
        message: 'Invalid model type specified'
      });
    }
  } catch (error) {
    console.error('Error getting model info:', error);
    return res.status(500).json({
      success: false,
      message: 'Error retrieving model information',
      error: error.message
    });
  }
};

/**
 * Analyze image using YOLO model
 */
exports.analyzeWithYolo = async (req, res) => {
  try {
    // Check if there's an image in the request
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No image file provided'
      });
    }

    // Read the image file as base64
    const imageBuffer = fs.readFileSync(req.file.path);
    const base64Image = imageBuffer.toString('base64');

    console.log(`Calling AI server at ${AI_SERVER_URL}/detect-yolo with image size: ${base64Image.length}`);
    
    try {
      // Call the AI server with timeout and detailed error handling
      const response = await axios.post(`${AI_SERVER_URL}/detect-yolo`, {
        image: base64Image
      }, {
        timeout: 30000, // 30 second timeout
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      console.log('AI server response received:', response.status);
      
      // Clean up the temporary file
      await unlinkAsync(req.file.path);
      
      if (!response.data.success) {
        throw new Error(`AI server returned error: ${response.data.message || 'Unknown error'}`);
      }

      // Return the AI server response
      return res.status(200).json({
        success: true,
        ...response.data
      });
    } catch (axiosError) {
      console.error('Axios error details:', {
        message: axiosError.message,
        code: axiosError.code,
        response: axiosError.response?.data,
        status: axiosError.response?.status
      });
      throw axiosError; // Re-throw for outer catch block
    }
  } catch (error) {
    console.error('Error analyzing with YOLO:', error);
    
    // Try to clean up temporary file if it exists
    if (req.file && req.file.path) {
      try {
        await unlinkAsync(req.file.path);
      } catch (cleanupError) {
        console.error('Error cleaning up temporary file:', cleanupError);
      }
    }
    
    // Provide more detailed error messages based on the error type
    let errorMessage = 'Error analyzing image with YOLO model';
    let statusCode = 500;
    
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      errorMessage = `AI server responded with error: ${error.response.status} - ${error.response.data?.message || 'Unknown error'}`;
      statusCode = error.response.status === 404 ? 404 : 500;
      console.error('Response error data:', error.response.data);
    } else if (error.request) {
      // The request was made but no response was received
      errorMessage = `No response received from AI server: ${error.message}`;
      console.error('No response received. Request:', error.request);
      statusCode = 503; // Service Unavailable
    } else if (error.code === 'ECONNREFUSED') {
      errorMessage = 'Could not connect to AI server. Is it running?';
      statusCode = 503; // Service Unavailable
    }
    
    return res.status(statusCode).json({
      success: false,
      message: errorMessage,
      error: error.message,
      details: error.response?.data || error.code || 'No additional details'
    });
  }
};
