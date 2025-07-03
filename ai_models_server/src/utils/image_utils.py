"""
Image processing utilities for AI Models Server
"""
import base64
import io
import os
import time
import random
from typing import Optional, Tuple, Dict, Any

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available")


def create_mock_annotation(image_data: str, damage_class: str) -> Optional[str]:
    """Create a mock annotated image with bounding box for fallback mode"""
    if not PIL_AVAILABLE:
        return None
    
    try:
        # Convert base64 to image
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Create random but reasonably placed bounding box coordinates
        width, height = img.size
        x1 = random.randint(width // 4, width // 2)
        y1 = random.randint(height // 4, height // 2)
        x2 = random.randint(x1 + 50, min(x1 + 200, width - 10))
        y2 = random.randint(y1 + 50, min(y1 + 200, height - 10))
        
        # Draw on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, y1 - 20), damage_class, fill=(255, 0, 0))
        
        # Convert back to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error creating mock annotation: {e}")
        return None


def create_mock_yolo_detection(image_data: str) -> Dict[str, Any]:
    """Create mock YOLO detection for fallback mode"""
    if not PIL_AVAILABLE:
        return {
            "detections": [],
            "annotated_image": None,
            "success": False,
            "error": "PIL not available"
        }
    
    try:
        # Convert base64 to image
        pil_img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        width, height = pil_img.size
        
        # Generate reasonable mock detection
        mock_bbox = [
            width // 4,             # x1
            height // 3,            # y1
            3 * width // 4,         # x2
            2 * height // 3         # y2
        ]
        
        # Create a random but reasonable damage class
        damage_classes = ["pothole", "crack", "alligator_crack", "longitudinal_crack"]
        mock_class = random.choice(damage_classes)
        
        # Draw the detection
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle(mock_bbox, outline=(0, 255, 0), width=3)
        draw.text((mock_bbox[0], mock_bbox[1] - 20), mock_class, fill=(255, 0, 0))
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        annotated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "detections": [{
                "class": mock_class,
                "confidence": 0.7,
                "bbox": mock_bbox
            }],
            "annotated_image": annotated_image,
            "success": True
        }
    except Exception as e:
        print(f"Error creating mock YOLO detection: {e}")
        return {
            "detections": [],
            "annotated_image": None,
            "success": False,
            "error": str(e)
        }


def save_image_from_base64(image_data: str, save_dir: str, filename_prefix: str = "upload") -> Tuple[bool, str, str]:
    """Save base64 image to disk"""
    try:
        # Create filename
        filename = f"{filename_prefix}_{int(time.time())}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # Decode and save
        img_data = base64.b64decode(image_data)
        with open(save_path, "wb") as f:
            f.write(img_data)
        
        # Generate URL
        image_url = f"/static/uploads/{filename}"
        
        return True, save_path, image_url
    except Exception as e:
        print(f"Error saving image: {e}")
        return False, "", ""


def get_image_dimensions(image_data: str) -> Tuple[int, int]:
    """Get image dimensions from base64 data"""
    if not PIL_AVAILABLE:
        return 640, 480  # Default dimensions
    
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return img.size
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return 640, 480  # Default dimensions
