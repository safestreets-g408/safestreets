#!/usr/bin/env python3
"""
Test script for CNN road classifier to confirm fix for the
'index 1 is out of bounds for dimension 1 with size 1' error
"""
import os
import sys
import time
import base64
import json
import io
from PIL import Image

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils.road_classifier import validate_road_image, load_road_classifier
from src.utils.image_utils import save_image_from_base64

def encode_image_to_base64(image_path):
    """Encode an image to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_road_classifier():
    """Test the CNN road classifier with various images"""
    print("Testing CNN road classifier...")
    
    # Ensure model is loaded
    load_result = load_road_classifier()
    if not load_result:
        print("❌ Failed to load road classifier model")
        return False
    
    print("✅ Road classifier model loaded successfully")
    
    # Test with sample image paths
    test_images = [
        {'path': 'static/uploads/road1.jpg', 'expected': True, 'name': 'Road image 1'},
        {'path': 'static/uploads/road2.jpg', 'expected': True, 'name': 'Road image 2'}
    ]
    
    # Create test directory if it doesn't exist
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    # If test images don't exist, create sample images
    if not os.path.exists(test_images[0]['path']):
        print("Creating sample test images...")
        # Create a gray road-like image
        road_img = Image.new('RGB', (640, 480), (100, 100, 100))
        road_img.save(test_images[0]['path'])
        
        # Create another sample road image
        road_img2 = Image.new('RGB', (640, 480), (120, 120, 120))
        road_img2.save(test_images[1]['path'])
    
    success = True
    
    # Test each image
    for test_img in test_images:
        try:
            print(f"\nTesting {test_img['name']} ({test_img['path']})...")
            
            # Read and encode image
            base64_img = encode_image_to_base64(test_img['path'])
            
            start_time = time.time()
            result = validate_road_image(base64_img)
            elapsed = time.time() - start_time
            
            # Print results
            print(f"Classification time: {elapsed:.2f} seconds")
            print(f"Is road: {result['is_road']} (expected: {test_img['expected']})")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Model prediction: {result['model_prediction']} (confidence: {result['model_confidence']:.4f})")
            print(f"Heuristic features: {json.dumps(result['heuristic_features'], indent=2)}")
            
            if result['is_road'] == test_img['expected']:
                print("✅ Classification matches expected result")
            else:
                print("❌ Classification doesn't match expected result")
                success = False
            
        except Exception as e:
            print(f"❌ Error testing {test_img['name']}: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    return success

if __name__ == "__main__":
    if test_road_classifier():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
