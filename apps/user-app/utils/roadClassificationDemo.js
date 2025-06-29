/**
 * Road Classification Demo
 * This script demonstrates how to integrate a real PyTorch model for road classification
 * 
 * To use your actual .pt model, you would need to:
 * 1. Convert the PyTorch model to TensorFlow Lite (.tflite) format
 * 2. Use react-native-tflite or similar library for inference
 * 3. Replace the simulation code with actual model loading and inference
 */

import { advancedRoadClassifier } from './advancedRoadClassifier';

/**
 * Example usage of the road classifier
 */
export const demonstrateRoadClassification = async () => {
  console.log('=== Road Classification Demo ===');
  
  // Example image URIs (in a real app, these would come from camera/gallery)
  const testImages = [
    'file:///path/to/road_image_1.jpg',
    'file:///path/to/road_image_2.jpg',
    'file:///path/to/non_road_image.jpg'
  ];
  
  for (const imageUri of testImages) {
    try {
      console.log(`\nAnalyzing: ${imageUri}`);
      
      // Classify the image
      const result = await advancedRoadClassifier.classifyImage(imageUri);
      
      console.log('Classification Result:', {
        isRoad: result.isRoad,
        confidence: `${(result.confidence * 100).toFixed(1)}%`,
        modelUsed: result.details.modelUsed
      });
      
      // Validate for damage analysis
      const validation = await advancedRoadClassifier.validateForRoadDamageAnalysis(imageUri, 0.7);
      
      console.log('Validation Result:', {
        isValid: validation.isValid,
        message: validation.message
      });
      
    } catch (error) {
      console.error('Error processing image:', error.message);
    }
  }
};

/**
 * Instructions for integrating a real PyTorch model
 */
export const integrationInstructions = {
  step1: {
    title: "Convert PyTorch Model to TensorFlow Lite",
    description: "Use the following Python script to convert your .pt model",
    pythonScript: `
# Convert PyTorch model to TensorFlow Lite
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.mobile_optimizer import optimize_for_mobile

# Load your PyTorch model
model = torch.jit.load('cnn_road_classifier_scripted.pt')
model.eval()

# Create example input tensor (adjust dimensions to match your model)
example_input = torch.randn(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Optimize for mobile
optimized_model = optimize_for_mobile(traced_model)

# Save optimized model
optimized_model._save_for_lite_interpreter("road_classifier_optimized.ptl")

# Alternative: Convert to ONNX then to TensorFlow Lite
torch.onnx.export(model, example_input, "road_classifier.onnx", 
                  export_params=True, opset_version=11, 
                  do_constant_folding=True,
                  input_names=['input'], output_names=['output'])
    `
  },
  
  step2: {
    title: "Install TensorFlow Lite React Native Package",
    commands: [
      "npm install react-native-tflite",
      "cd ios && pod install", // For iOS
      "npx react-native link react-native-tflite" // For older RN versions
    ]
  },
  
  step3: {
    title: "Update the Road Classifier Implementation",
    codeExample: `
// In advancedRoadClassifier.js
import TfLite from 'react-native-tflite';

class AdvancedRoadClassifier {
  async loadModel() {
    try {
      // Load the converted model
      const modelPath = 'road_classifier.tflite'; // Place in assets folder
      await TfLite.loadModel({
        model: modelPath,
        labels: 'road_labels.txt' // Create labels file: road\\nnon_road
      });
      
      this.modelLoaded = true;
      console.log('TensorFlow Lite model loaded successfully');
    } catch (error) {
      console.error('Failed to load TensorFlow Lite model:', error);
      this.modelLoaded = false;
    }
  }
  
  async runInference(imageUri) {
    if (!this.modelLoaded) {
      return this.fallbackClassification(imageUri);
    }
    
    try {
      // Run inference with TensorFlow Lite
      const result = await TfLite.runModelOnImage({
        path: imageUri,
        imageMean: 128.0, // Adjust based on your model's preprocessing
        imageStd: 128.0,
        numResults: 2, // road, non_road
        threshold: 0.1
      });
      
      // Process the results
      const roadConfidence = result.find(r => r.label === 'road')?.confidence || 0;
      const isRoad = roadConfidence > 0.5;
      
      return {
        isRoad,
        confidence: roadConfidence,
        probabilities: {
          road: roadConfidence,
          nonRoad: 1 - roadConfidence
        },
        modelUsed: 'TensorFlow Lite',
        rawResults: result
      };
    } catch (error) {
      console.error('TensorFlow Lite inference failed:', error);
      return this.fallbackClassification(imageUri);
    }
  }
}
    `
  },
  
  step4: {
    title: "Asset Organization",
    instructions: [
      "Place the converted .tflite model in: apps/user-app/assets/models/road_classifier.tflite",
      "Create a labels file: apps/user-app/assets/models/road_labels.txt",
      "Update metro.config.js to include .tflite files in asset extensions"
    ]
  },
  
  step5: {
    title: "Metro Config Update",
    metroConfig: `
// metro.config.js
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add .tflite and .ptl extensions
config.resolver.assetExts.push('tflite', 'ptl', 'txt');

module.exports = config;
    `
  }
};

export default {
  demonstrateRoadClassification,
  integrationInstructions
};
