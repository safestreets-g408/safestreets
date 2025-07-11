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
    console.log('YOLO Analysis request received:', { 
      hasFile: !!req.file,
      fileName: req.file?.originalname || 'No file name',
      fileSize: req.file?.size ? `${(req.file.size / 1024).toFixed(2)} KB` : 'Unknown size',
      mimeType: req.file?.mimetype || 'Unknown type'
    });
    
    // Check if there's an image in the request
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No image file provided'
      });
    }

    // With memoryStorage, the file is already available as a buffer in req.file.buffer
    // No need to read from disk
    const imageBuffer = req.file.buffer;
    if (!imageBuffer) {
      return res.status(400).json({
        success: false,
        message: 'Image buffer is empty or undefined'
      });
    }
    
    const base64Image = imageBuffer.toString('base64');

    console.log(`Calling AI server at ${AI_SERVER_URL}/detect-yolo with image size: ${base64Image.length}`);
    
    try {
      console.log(`Calling AI server for YOLO detection, image size: ${Math.round(base64Image.length/1024)}KB`);
      const startTime = Date.now();
      
      // Call the AI server with timeout and detailed error handling
      const response = await axios.post(`${AI_SERVER_URL}/detect-yolo`, {
        image: base64Image
      }, {
        timeout: 120000, // 120 second timeout for very large images
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
      
      console.log(`YOLO detection completed in ${(Date.now() - startTime)/1000}s`);
      
      console.log('AI server response received:', response.status);
      
      // No cleanup needed for memory storage
      
      // Check if we have a valid response
      if (!response.data) {
        throw new Error('AI server returned empty response');
      }

      // Even if the AI server returned an error message, send back the response
      // This allows the frontend to handle fallback detections
      const isSuccess = response.data.success !== false;
      
      // If there's a fallback message, use HTTP 200 but with success: false
      if (!isSuccess && response.data.detections) {
        console.log('AI server used fallback detection');
        return res.status(200).json({
          success: true, // Changed to true since we have usable results
          fallback: true,
          ...response.data
        });
      }

      // Return the AI server response
      return res.status(200).json({
        success: isSuccess,
        ...response.data
      });
    } catch (axiosError) {
      console.error('Axios error details:', {
        message: axiosError.message,
        code: axiosError.code,
        response: axiosError.response?.data,
        status: axiosError.response?.status
      });
      
      // If the AI server returned any usable data, send it back
      if (axiosError.response && axiosError.response.data && axiosError.response.data.detections) {
        return res.status(200).json({
          success: true,
          fallback: true,
          ...axiosError.response.data,
          message: 'Using fallback detection due to partial AI server error'
        });
      }
      
      throw axiosError; // Re-throw for outer catch block
    }
  } catch (error) {
    console.error('Error analyzing with YOLO:', error);
    
    // No temporary file cleanup needed with memory storage
    
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
