const express = require('express');
const router = express.Router();
const { uploadImage, getImage } = require('../controllers/ImageController');
const upload = require('../middleware/multerConfig');
const protect = require('../middleware/authMiddleware');

// Protected routes
router.post('/upload', protect, upload.single('image'), uploadImage);
router.get('/:email', protect, getImage);

module.exports = router;