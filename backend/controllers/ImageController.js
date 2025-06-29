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

// Mapping function that combines damage type and severity/priority calculations
const mapDamageClass = (predClass) => {
    const damageType = getDamageType(predClass);
    const { severity, priority } = calculateSeverityAndPriority(predClass);
    return { damageType, severity, priority };
};

const uploadImage = async (req, res) => {
    try {
        console.log('Upload image request received');
        console.log('Request headers:', req.headers);
        console.log('Files:', req.file ? 'File present' : 'No file');
        console.log('Body keys:', Object.keys(req.body));
        console.log('Admin:', req.admin ? `ID: ${req.admin._id}, Role: ${req.admin.role}` : 'Not set');
        console.log('TenantId:', req.tenantId || 'Not set');

        if (!req.file) {
            console.error('No image file uploaded');
            return res.status(400).json({ 
                message: 'No image file uploaded', 
                success: false 
            });
        }

        // Extract location data from request if available
        const { latitude, longitude, address } = req.body;
        console.log('Location data received:', { latitude, longitude, address });

        // Get tenant ID either from req.tenantId or from req.admin.tenant
        const tenantId = req.tenantId || (req.admin && req.admin.tenant && req.admin.tenant._id) || 
                          (req.admin && req.admin.tenant);
                          
        if (!tenantId) {
            console.error('No tenant ID available');
            return res.status(400).json({
                message: 'No tenant association found. Please contact support.',
                success: false
            });
        }
        
        console.log('Using tenant ID:', tenantId);
        
        // Create a new image document
        const newImage = new Image({
            tenant: tenantId,
            data: req.file.buffer,
            contentType: req.file.mimetype,
            result: 'Processing'
        });

        const savedImage = await newImage.save();
        const newImageId = savedImage._id;
        console.log('New image saved with ID:', newImageId);

        try {
            // Convert image buffer to base64
            const base64Image = req.file.buffer.toString('base64');
            console.log(`Base64 image created, length: ${base64Image.length}`);
            
            // Send image to AI server for processing
            const aiServerUrl = process.env.AI_SERVER_URL || 'http://localhost:5000';
            console.log(`Sending request to AI server: ${aiServerUrl}/predict`);
            
            const response = await axios.post(`${aiServerUrl}/predict`, {
                image: base64Image
            }, {
                timeout: AI_REQUEST_TIMEOUT
            });

            console.log('AI server response received');
            
            if (!response.data || !response.data.success) {
                console.error('Invalid response from AI server:', response.data);
                throw new Error('Invalid response from AI server');
            }

            // Extract prediction results
            const { prediction, annotated_image, confidence } = response.data;
            console.log('Prediction:', prediction, 'Confidence:', confidence);
            
            // Map damage class to damage type, severity and priority
            const { damageType, severity, priority } = mapDamageClass(prediction);

            // Create AI report
            const aiReport = new AiReport({
                imageId: newImageId,
                tenant: req.tenantId,
                predictionClass: prediction,
                damageType,
                severity,
                priority,
                annotatedImageBase64: annotated_image,
                location: {
                    coordinates: longitude && latitude ? [parseFloat(longitude), parseFloat(latitude)] : undefined,
                    address: address || undefined
                }
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

            res.status(500).json({
                message: 'Error processing image. Please try again.',
                error: processError.message,
                success: false
            });
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        res.status(500).json({ 
            message: 'Error uploading image. Please try again.',
            error: error.message,
            success: false 
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
                annotatedImageBase64: report.annotatedImageBase64,
                createdAt: report.createdAt,
                location: report.location // Include location information from the report
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
                createdAt: report.createdAt,
                location: report.location // Include location information
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