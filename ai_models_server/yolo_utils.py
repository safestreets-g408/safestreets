import torch
import cv2
import numpy as np
import base64
from PIL import Image
import io

# Load YOLO model globally for reuse
print("Loading YOLO model...")
try:
    YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo_model.pt')
    # Set confidence threshold
    YOLO_MODEL.conf = 0.5
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    YOLO_MODEL = None

def process_image_from_base64(base64_string):
    """
    Convert base64 string to image for YOLO processing
    """
    try:
        # Decode base64 string
        img_bytes = base64.b64decode(base64_string)
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except Exception as e:
        print(f"Error processing base64 image: {e}")
        raise

def detect_road_damage(image, save_path=None):
    if YOLO_MODEL is None:
        raise RuntimeError("YOLO model not loaded successfully")
        
    try:
        # Run inference
        results = YOLO_MODEL(image)
        
        # Convert results to JSON serializable format
        detections = []
        for pred in results.xyxy[0].cpu().numpy():  # Process first image only
            x1, y1, x2, y2, conf, cls = pred
            class_id = int(cls)
            class_name = results.names[class_id]
            
            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Get annotated image in base64 format
        annotated_img = results.render()[0]  # Get BGR image with annotations
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            
        return {
            'detections': detections,
            'annotated_image': base64_image,
            'count': len(detections)
        }
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        raise
