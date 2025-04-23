const express = require('express');
const router = express.Router();
const { uploadImage, getImage, getImageById } = require('../controllers/ImageController');
const upload = require('../middleware/multerConfig');
const protect = require('../middleware/authMiddleware');

// Protected routes
router.post('/upload', protect, upload.single('image'), uploadImage);
router.get('/email/:email', protect, getImage);  // Fixed route path
router.get('/id/:imageId', protect, getImageById);

module.exports = router;