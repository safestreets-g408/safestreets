const fs = require('fs');
const multer = require('multer');
const Image = require('../models/Image');

const uploadImage = async (req, res) => {
    try {
        const { name, email } = req.body;
        const { path, mimetype } = req.file;

        if (!path) {
            return res.status(400).json({ message: 'Image data is required' });
        }

        const imageData = fs.readFileSync(path);

        const newImage = new Image({
            name,
            email,
            image: {
                data: imageData,
                contentType: mimetype
            },
            result: 'Pending'
        });

        await newImage.save();
        
        // Delete the file from uploads folder after saving to DB
        fs.unlinkSync(path);
        
        res.status(201).json({ 
            message: 'Road damage image uploaded successfully',
            imageId: newImage._id 
        });
    } catch (error) {
        if (error instanceof multer.MulterError) {
            console.error('MulterError:', error);
            res.status(400).json({ message: error.message });
        } else {
            console.error('Error uploading Image:', error);
            res.status(500).json({ message: 'Error uploading road damage image. Please try again.' });
        }
    }
}

const getImage = async (req, res) => {
    try {
        const { email } = req.params;
        const image = await Image.findOne({ email });

        if (!image) {
            return res.status(404).json({ message: 'Road damage image not found' });
        }

        res.set('Content-Type', image.image.contentType);
        res.send({
            name: image.name,
            imageData: image.image.data,
            result: image.result,
            createdAt: image.createdAt
        });
    } catch (error) {
        console.error('Error retrieving image:', error);
        res.status(500).json({ message: 'Error retrieving road damage image. Please try again.' });
    }
}

module.exports = { uploadImage, getImage };