#!/usr/bin/env python3
"""
Test script for ViT model to confirm the fix for the
'y1 must be greater than or equal to y0' error
"""
import os
import sys
import time
import base64
import json
import io
import traceback

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from PIL import Image
except ImportError:
    print("WARNING: PIL not installed. Please install with 'pip install pillow'")
    sys.exit(1)

from src.utils.predict import load_vit_model, predict_from_base64

def encode_image_to_base64(image_path):
    """Encode an image to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_vit_model():
    """Test the ViT model with various images"""
    print("Testing ViT model...")
    
    # Ensure model is loaded
    load_result = load_vit_model()
    if not load_result:
        print("❌ Failed to load ViT model")
        return False
    
    print("✅ ViT model loaded successfully")
    
    # Test with sample image paths or create them
    test_images = [
        {'path': 'static/uploads/damage1.jpg', 'name': 'Damage image 1'},
        {'path': 'static/uploads/damage2.jpg', 'name': 'Damage image 2'}
    ]
    
    # Create test directory if it doesn't exist
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    # If test images don't exist, create sample images
    for test_img in test_images:
        if not os.path.exists(test_img['path']):
            print(f"Creating sample test image {test_img['name']}...")
            # Create a simple image with some "damage"
            width, height = 640, 480
            image = Image.new('RGB', (width, height), (200, 200, 200))
            
            # Add a dark spot to simulate damage
            for x in range(width//3, width//2):
                for y in range(height//3, height//2):
                    image.putpixel((x, y), (50, 50, 50))
                    
            image.save(test_img['path'])
    
    success = True
    
    # Test each image
    for test_img in test_images:
        try:
            print(f"\nTesting {test_img['name']} ({test_img['path']})...")
            
            # Read and encode image
            base64_img = encode_image_to_base64(test_img['path'])
            
            start_time = time.time()
            result = predict_from_base64(base64_img)
            elapsed = time.time() - start_time
            
            # Print results
            print(f"Prediction time: {elapsed:.2f} seconds")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Bounding box: {result['bbox']}")
            
            # Validate bounding box coordinates
            bbox = result['bbox']
            if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                print("✅ Bounding box coordinates are valid")
            else:
                print("❌ Invalid bounding box coordinates:", bbox)
                success = False
            
            # Test if annotated image was generated
            if 'annotated_image' in result and result['annotated_image']:
                print("✅ Annotated image generated successfully")
                
                # Save the annotated image for inspection
                annotated_file = f"{test_img['path'].replace('.jpg', '_annotated.jpg')}"
                img_data = base64.b64decode(result['annotated_image'])
                with open(annotated_file, 'wb') as f:
                    f.write(img_data)
                print(f"Saved annotated image to {annotated_file}")
            else:
                print("❌ Failed to generate annotated image")
                success = False
            
        except Exception as e:
            print(f"❌ Error testing {test_img['name']}: {e}")
            traceback.print_exc()
            success = False
    
    return success

if __name__ == "__main__":
    if test_vit_model():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
