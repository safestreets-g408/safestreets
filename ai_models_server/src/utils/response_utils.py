"""
Response utilities for AI Models Server
"""
import json
import numpy as np

try:
    from flask import jsonify as flask_jsonify
except ImportError:
    print("Warning: Flask not available")
    def flask_jsonify(data):
        return data


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def jsonify(data):
    """Custom jsonify that handles numpy types"""
    try:
        # Convert numpy types to Python native types
        json_str = json.dumps(data, cls=CustomJSONEncoder)
        json_data = json.loads(json_str)
        return flask_jsonify(json_data)
    except Exception as e:
        print(f"JSON serialization error: {e}")
        # Fallback: try to convert problematic types
        return flask_jsonify(_sanitize_for_json(data))


def _sanitize_for_json(obj):
    """Recursively sanitize object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def create_success_response(data, message="Success"):
    """Create a standardized success response"""
    response_data = {
        "success": True,
        "message": message,
        **data
    }
    return jsonify(response_data)


def create_error_response(error_message, status_code=400, details=None):
    """Create a standardized error response"""
    response_data = {
        "success": False,
        "error": error_message,
        "message": error_message
    }
    
    if details:
        response_data["details"] = details
    
    return jsonify(response_data), status_code


def create_fallback_response(data, message="Using fallback mode"):
    """Create a response for fallback mode"""
    response_data = {
        "success": True,
        "fallback": True,
        "message": message,
        **data
    }
    return jsonify(response_data)


def add_cors_headers(response):
    """Add CORS headers to response"""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def handle_preflight_request():
    """Handle OPTIONS preflight request"""
    response = jsonify({'message': 'OK'})
    return add_cors_headers(response)
