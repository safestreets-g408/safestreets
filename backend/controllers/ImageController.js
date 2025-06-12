const fs = require('fs');
const multer = require('multer');
const axios = require('axios');
const Image = require('../models/Image');
const AiReport = require('../models/AiReport');

// Configuration
const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:5000';
const AI_REQUEST_TIMEOUT = 60000; // 60 seconds

// Helper function to determine damage type from prediction class 
const getDamageType = (predClass) => {
    const damageTypes = {
        'D00': 'Linear Crack',
        'D10': 'Linear Crack',
        'D20': 'Alligator Crack',
        'D30': 'Potholes',
        'D40': 'Patches',
        'D43': 'White Line Blur',
        'D44': 'Cross Walk Blur',
        'D50': 'Utility Hole'
    };
    return damageTypes[predClass] || 'Unknown';
};

// Helper function to calculate severity and priority
const calculateSeverityAndPriority = (predClass) => {
    const highSeverityDamages = ['D30', 'D20']; // Potholes and Alligator cracks are high severity
    const mediumSeverityDamages = ['D00', 'D10', 'D40']; // Linear cracks and patches are medium
    const lowSeverityDamages = ['D43', 'D44', 'D50']; // Line blur and utility holes are low

    let severity = 'MEDIUM';
    let priority = 5;

    if (highSeverityDamages.includes(predClass)) {
        severity = 'HIGH';
        priority = 8;
    } else if (lowSeverityDamages.includes(predClass)) {
        severity = 'LOW';
        priority = 3;
    }

    return { severity, priority };
};

const uploadImage = async (req, res) => {
    let newImageId = null;
    
    try {
        console.log('=== Upload Request Started ===');
        console.log('Request headers:', req.headers);
        console.log('Request body keys:', Object.keys(req.body));
        console.log('Image data length:', req.body.image?.length);
        
        const { name, email, image, contentType } = req.body;

        // Validate required fields
        if (!image) {
            console.error('No image data received');
            return res.status(400).json({ 
                message: 'Image data is required',
                success: false 
            });
        }

        if (!name || !email) {
            console.error('Missing required fields:', { name: !!name, email: !!email });
            return res.status(400).json({ 
                message: 'Name and email are required',
                success: false 
            });
        }

        // Validate base64 image data
        if (typeof image !== 'string' || image.length === 0) {
            console.error('Invalid image data format');
            return res.status(400).json({ 
                message: 'Invalid image data format',
                success: false 
            });
        }

        try {
            console.log('Converting base64 to buffer...');
            const imageBuffer = Buffer.from(image, 'base64');
            console.log('Buffer created, size:', imageBuffer.length);

            // Create new image document
            console.log('Creating new image document...');
            const newImage = new Image({
                name,
                email,
                image: {
                    data: imageBuffer,
                    contentType: contentType || 'image/jpeg'
                },
                result: 'Processing'
            });

            const savedImage = await newImage.save();
            newImageId = savedImage._id;
            console.log('Image saved to database with ID:', newImageId);

            // Call AI server for prediction with enhanced error handling
            console.log(`Calling AI server at ${AI_SERVER_URL}/predict...`);
            
            const requestConfig = {
                timeout: AI_REQUEST_TIMEOUT,
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                validateStatus: function (status) {
                    return status < 500; // Resolve only if the status code is less than 500
                }
            };

            const requestData = { image: image };
            console.log('Request config:', JSON.stringify(requestConfig, null, 2));

            let aiResponse;
            try {
                aiResponse = await axios.post(`${AI_SERVER_URL}/predict`, requestData, requestConfig);
            } catch (networkError) {
                console.error('Network error calling AI server:', networkError.message);
                if (networkError.code === 'ECONNREFUSED') {
                    throw new Error('AI server is not available. Please ensure the AI service is running on port 5000.');
                } else if (networkError.code === 'ETIMEDOUT') {
                    throw new Error('AI server request timed out. Please try again.');
                } else {
                    throw new Error(`Network error: ${networkError.message}`);
                }
            }

            console.log('AI server response status:', aiResponse.status);
            console.log('AI server response headers:', aiResponse.headers);
            console.log('AI server response data:', aiResponse.data);

            // Handle non-200 responses
            if (aiResponse.status !== 200) {
                console.error('AI server returned non-200 status:', aiResponse.status);
                const errorMsg = aiResponse.data?.error || aiResponse.data?.message || 'Unknown error';
                throw new Error(`AI server error (${aiResponse.status}): ${errorMsg}`);
            }

            if (!aiResponse.data) {
                throw new Error('No response data from AI server');
            }

            if (!aiResponse.data.success) {
                throw new Error('AI prediction failed: ' + (aiResponse.data.error || 'Unknown error'));
            }

            const { prediction, annotated_image } = aiResponse.data;
            
            if (!prediction) {
                throw new Error('No prediction received from AI server');
            }

            console.log('Prediction received:', prediction);
            
            // Calculate severity and priority
            const damageType = getDamageType(prediction);
            const { severity, priority } = calculateSeverityAndPriority(prediction);

            console.log('Damage analysis:', { damageType, severity, priority });

            // Create AI Report
            console.log('Creating AI report...');
            const aiReport = new AiReport({
                imageId: savedImage._id,
                predictionClass: prediction,
                damageType,
                severity,
                priority,
                annotatedImageBase64: annotated_image || null
            });

            const savedReport = await aiReport.save();
            console.log('AI report saved with ID:', savedReport._id);

            // Update image status
            savedImage.result = 'Completed';
            await savedImage.save();
            
            console.log('=== Processing completed successfully ===');
            res.status(201).json({ 
                message: 'Road damage image processed successfully',
                success: true,
                imageId: savedImage._id,
                reportId: savedReport._id,
                prediction: {
                    damageType,
                    severity,
                    priority,
                    predictionClass: prediction
                }
            });

        } catch (processError) {
            console.error('Error processing image:', processError.message);
            console.error('Error stack:', processError.stack);
            
            // Update image status to failed if we created one
            if (newImageId) {
                try {
                    await Image.findByIdAndUpdate(newImageId, { result: 'Failed' });
                    console.log('Updated image status to Failed');
                } catch (updateError) {
                    console.error('Error updating image status:', updateError);
                }
            }
            
            // Re-throw with the original error message
            throw processError;
        }
    } catch (error) {
        console.error('=== Upload Error ===');
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
        
        // Provide user-friendly error messages
        let userMessage = 'Error processing road damage image. Please try again.';
        
        if (error.message.includes('AI server is not available')) {
            userMessage = 'AI processing service is currently unavailable. Please try again later.';
        } else if (error.message.includes('timeout')) {
            userMessage = 'Request timed out. Please try again with a smaller image.';
        } else if (error.message.includes('Invalid image')) {
            userMessage = 'Invalid image format. Please use JPEG or PNG images.';
        } else if (error.message.includes('AI server error')) {
            userMessage = 'AI processing failed. Please try again.';
        }
        
        res.status(500).json({ 
            message: userMessage,
            success: false,
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};

// Test AI server connectivity
const testAiServer = async (req, res) => {
    try {
        console.log(`Testing AI server connectivity at ${AI_SERVER_URL}...`);
        
        const response = await axios.get(`${AI_SERVER_URL}/health`, {
            timeout: 5000,
            validateStatus: function (status) {
                return status < 500;
            }
        });
        
        console.log('AI server test response:', response.status, response.data);
        
        if (response.status === 200) {
            res.status(200).json({
                message: 'AI server is accessible',
                success: true,
                serverUrl: AI_SERVER_URL,
                serverResponse: response.data
            });
        } else {
            res.status(response.status).json({
                message: 'AI server responded with error',
                success: false,
                serverUrl: AI_SERVER_URL,
                statusCode: response.status,
                serverResponse: response.data
            });
        }
    } catch (error) {
        console.error('AI server test failed:', error.message);
        
        let errorMessage = 'AI server is not accessible';
        if (error.code === 'ECONNREFUSED') {
            errorMessage = 'AI server is not running or not accessible';
        } else if (error.code === 'ETIMEDOUT') {
            errorMessage = 'AI server connection timed out';
        }
        
        res.status(503).json({
            message: errorMessage,
            success: false,
            serverUrl: AI_SERVER_URL,
            error: error.code || error.message
        });
    }
};

const getImage = async (req, res) => {
    try {
        const { email } = req.params;
        const image = await Image.findOne({ email });

        if (!image) {
            return res.status(404).json({ 
                message: 'Road damage image not found',
                success: false 
            });
        }

        // Send binary data directly
        res.set({
            'Content-Type': image.image.contentType,
            'Content-Disposition': `inline; filename="${image.name}"`,
            'Cache-Control': 'no-cache'
        });
        res.send(image.image.data);
    } catch (error) {
        console.error('Error retrieving image:', error);
        res.status(500).json({ 
            message: 'Error retrieving road damage image. Please try again.',
            success: false 
        });
    }
};

const getImageById = async (req, res) => {
    try {
        const { imageId } = req.params;
        const image = await Image.findById(imageId);

        if (!image) {
            return res.status(404).json({ 
                message: 'Road damage image not found',
                success: false 
            });
        }

        // Send binary data directly
        res.set({
            'Content-Type': image.image.contentType,
            'Content-Disposition': `inline; filename="${image.name}"`,
            'Cache-Control': 'no-cache'
        });
        res.send(image.image.data);
    } catch (error) {
        console.error('Error retrieving image:', error);
        res.status(500).json({ 
            message: 'Error retrieving road damage image. Please try again.',
            success: false 
        });
    }
};

const getReports = async (req, res) => {
    try {
        // Fetch all reports and populate the imageId reference
        const reports = await AiReport.find({})
            .populate('imageId', 'name email')
            .sort({ createdAt: -1 }); // Sort by newest first

        if (!reports || reports.length === 0) {
            return res.status(200).json({ 
                message: 'No reports found',
                success: true,
                reports: []
            });
        }

        res.status(200).json({
            message: 'Reports retrieved successfully',
            success: true,
            reports: reports.map(report => ({
                id: report._id,
                imageId: report.imageId._id,
                imageName: report.imageId.name,
                userEmail: report.imageId.email,
                damageType: report.damageType,
                severity: report.severity,
                priority: report.priority,
                predictionClass: report.predictionClass,
                annotatedImage: report.annotatedImageBase64 ? `data:image/jpeg;base64,${report.annotatedImageBase64}` : null,
                createdAt: report.createdAt
            }))
        });
    } catch (error) {
        console.error('Error retrieving reports:', error);
        res.status(500).json({ 
            message: 'Error retrieving reports. Please try again.',
            success: false 
        });
    }
};

const getReportById = async (req, res) => {
    try {
        const { reportId } = req.params;
        const report = await AiReport.findById(reportId)
            .populate('imageId', 'name email');

        if (!report) {
            return res.status(404).json({ 
                message: 'Report not found',
                success: false 
            });
        }

        res.status(200).json({
            message: 'Report retrieved successfully',
            success: true,
            report: {
                id: report._id,
                imageId: report.imageId._id,
                imageName: report.imageId.name,
                userEmail: report.imageId.email,
                damageType: report.damageType,
                severity: report.severity,
                priority: report.priority,
                predictionClass: report.predictionClass,
                annotatedImage: report.annotatedImageBase64 ? `data:image/jpeg;base64,${report.annotatedImageBase64}` : null,
                createdAt: report.createdAt
            }
        });
    } catch (error) {
        console.error('Error retrieving report:', error);
        res.status(500).json({ 
            message: 'Error retrieving report. Please try again.',
            success: false 
        });
    }
};

module.exports = { 
    uploadImage, 
    getImage, 
    getImageById, 
    getReports, 
    getReportById,
    testAiServer 
};