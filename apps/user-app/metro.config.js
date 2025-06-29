// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add support for ML model files
config.resolver.assetExts.push('pt', 'tflite', 'ptl', 'onnx', 'txt');

module.exports = config;
