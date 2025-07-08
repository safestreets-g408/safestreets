#!/usr/bin/env python3
"""
Test script for YOLOv8 road damage detection
"""
import os
import sys
import time
import base64
import json
import argparse
from PIL import Image

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def encode_image_to_base64(image_path):
    """Encode an image to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_yolo_detector(image_path=None, save_output=True, confidence=0.25):
    """Test the YOLOv8 road damage detector"""
    print("Testing YOLOv8 road damage detector...")
    
    # Import the detector
    try:
        from src.utils.yolov8_utils import detect_road_damage, load_yolo_model
        print("Using YOLOv8 detector")
    except ImportError as e:
        print(f"Could not import YOLOv8 detector: {e}")
        print("Trying fallback YOLOv5 detector...")
        try:
            from src.utils.yolo_utils import detect_road_damage, load_yolo_model
            print("Using YOLOv5 detector")
        except ImportError:
            print("No YOLO detector available")
            return False
    
    # Ensure model is loaded
    load_result = load_yolo_model()
    if not load_result:
        print("❌ Failed to load YOLO model")
        return False
    
    print("✅ YOLO model loaded successfully")
    
    # Use default test image if none provided
    if not image_path:
        # Look for sample images in static/uploads directory
        uploads_dir = os.path.join(parent_dir, "static", "uploads")
        if os.path.exists(uploads_dir):
            image_files = [f for f in os.listdir(uploads_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                         and os.path.isfile(os.path.join(uploads_dir, f))]
            if image_files:
                image_path = os.path.join(uploads_dir, image_files[0])
                print(f"Using sample image: {image_path}")
            else:
                print("No sample images found in static/uploads")
                return False
        else:
            print(f"Uploads directory not found at {uploads_dir}")
            return False
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return False
    
    # Prepare output path if saving results
    output_path = None
    if save_output:
        output_dir = os.path.join(parent_dir, "static", "results")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"yolo_result_{filename}")
    
    try:
        print(f"Processing image: {image_path}")
        
        # Read and encode image
        base64_img = encode_image_to_base64(image_path)
        
        # Time the detection
        start_time = time.time()
        
        # Detect road damage
        result = detect_road_damage(
            base64_img, 
            save_path=output_path,
            conf_threshold=confidence
        )
        
        elapsed = time.time() - start_time
        
        # Print results
        print(f"Detection time: {elapsed:.2f} seconds")
        print(f"Detections found: {result['count']}")
        
        if 'error' in result:
            print(f"⚠️ Error in detection: {result['error']}")
        
        if result['count'] > 0:
            print("\nDetected damages:")
            for i, detection in enumerate(result['detections']):
                print(f"  {i+1}. Class: {detection['class']}, " 
                     f"Confidence: {detection['confidence']:.4f}, "
                     f"Box: {[round(x, 2) for x in detection['bbox']]}")
        else:
            print("No damages detected")
        
        if output_path and os.path.exists(output_path):
            print(f"\n✅ Annotated image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Error testing YOLO detector: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test YOLO road damage detector')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--no-save', action='store_true', help='Do not save output image')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection confidence threshold')
    args = parser.parse_args()
    
    if test_yolo_detector(args.image, not args.no_save, args.confidence):
        print("\n✅ Test completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Test failed")
        sys.exit(1)
