"""
Validation utilities for AI Models Server
"""
import base64
import io
from PIL import Image
from typing import Tuple, Optional


def validate_request_json(request) -> Tuple[bool, Optional[str]]:
    """Validate if request is JSON"""
    if not request.is_json:
        return False, "Request must be JSON"
    
    data = request.get_json()
    if not data:
        return False, "No data in request body"
    
    return True, None


def validate_image_data(data: dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate image data in request"""
    if 'image' not in data:
        return False, "No image data in request body", None
    
    image_string = data['image']
    
    if not image_string or not isinstance(image_string, str):
        return False, "Invalid image data format", None
    
    # Try to decode base64 to validate
    try:
        image_data = base64.b64decode(image_string)
        # Try to open as image to validate format
        Image.open(io.BytesIO(image_data))
        return True, None, image_string
    except Exception as e:
        return False, f"Invalid image format: {str(e)}", None


def validate_summary_request(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate summary generation request"""
    required_fields = ['location', 'damageType', 'severity', 'priority']
    missing_fields = []
    
    for field in required_fields:
        if field not in data or not data[field]:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required parameters: {', '.join(missing_fields)}"
    
    return True, None


def validate_confidence_threshold(data: dict, default: float = 0.6) -> float:
    """Validate and return confidence threshold"""
    try:
        threshold = float(data.get('confidenceThreshold', default))
        # Ensure threshold is within valid range
        return max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        return default


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal"""
    # Remove any directory separators and dangerous characters
    import re
    filename = re.sub(r'[^\w\-_\.]', '', filename)
    return filename[:100]  # Limit length
