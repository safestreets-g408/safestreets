from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import io
import random
import time
from PIL import Image, ImageDraw, ImageFont
import tempfile
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import gemini utils for AI summary generation
from gemini_utils import generate_road_damage_summary
# Import YOLO detection utilities
from yolo_utils import process_image_from_base64, detect_road_damage

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True
)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add route to serve static files
@app.route('/static/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "AI server is running",
        "version": "1.0.0"
    }), 200

# Mock function to create an annotated image with bounding box
def annotate_mock_image(image_data, damage_class):
    try:
        # Convert base64 to image
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Create random bounding box coordinates
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
        print(f"Error annotating image: {e}")
        return None

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight request
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    print("Prediction endpoint called")
    
    try:
        # Validate request content type
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({
                "error": "Request must be JSON",
                "success": False
            }), 400

        data = request.get_json()
        if not data:
            print("No data in request body")
            return jsonify({
                "error": "No data in request body",
                "success": False
            }), 400
            
        if 'image' not in data:
            print("No image data in request")
            return jsonify({
                "error": "No image data in request body",
                "success": False
            }), 400

        image_string = data['image']
        
        # Validate base64 string
        if not image_string or not isinstance(image_string, str):
            print("Invalid image data format")
            return jsonify({
                "error": "Invalid image data format",
                "success": False
            }), 400

        print("Image data received, length:", len(image_string))
            
        # Simulate processing time
        time.sleep(1)
        
        # Generate a random prediction
        damage_classes = ['D00', 'D10', 'D20', 'D30', 'D40', 'D43', 'D44', 'D50']
        prediction = random.choice(damage_classes)
        
        print(f"Generating mock prediction: {prediction}")
        
        # Create annotated image
        annotated_image = annotate_mock_image(image_string, prediction)
        
        print("Returning response")
        
        response = jsonify({
            "prediction": prediction,
            "annotated_image": annotated_image,
            "success": True,
            "message": "Mock prediction successful"
        })
        
        # Add CORS headers to response
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "message": "Server error processing image",
            "success": False
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "success": False
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500

# Endpoint for generating road damage summaries
@app.route("/generate-summary", methods=["POST", "OPTIONS"])
def generate_summary():
    # Handle preflight request
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    print("Summary generation endpoint called")
    
    try:
        # Validate request content type
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({
                "error": "Request must be JSON",
                "success": False
            }), 400

        data = request.get_json()
        if not data:
            print("No data in request body")
            return jsonify({
                "error": "No data in request body",
                "success": False
            }), 400
            
        # Extract required parameters
        location = data.get('location')
        damage_type = data.get('damageType')
        severity = data.get('severity')
        priority = data.get('priority')
        
        # Validate required parameters
        if not all([location, damage_type, severity, priority]):
            missing = []
            if not location: missing.append('location')
            if not damage_type: missing.append('damageType')
            if not severity: missing.append('severity')
            if not priority: missing.append('priority')
            
            print(f"Missing required parameters: {missing}")
            return jsonify({
                "error": f"Missing required parameters: {', '.join(missing)}",
                "success": False
            }), 400
        
        # Generate the summary
        print(f"Generating summary for {location}, {damage_type}, {severity}, priority {priority}")
        summary = generate_road_damage_summary(location, damage_type, severity, priority)
        
        # Return the generated summary
        response = jsonify({
            "summary": summary,
            "success": True,
            "message": "Summary generated successfully"
        })
        
        # Add CORS headers to response
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print(f"Error in summary generation endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "message": "Server error generating summary",
            "success": False
        }), 500

# Endpoint for YOLO detection
@app.route("/detect-yolo", methods=["POST", "OPTIONS"])
def detect_yolo():
    # Handle preflight request
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
    
    print("YOLO detection endpoint called")
    
    try:
        # Validate request content type
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({
                "error": "Request must be JSON",
                "success": False
            }), 400

        data = request.get_json()
        if not data:
            print("No data in request body")
            return jsonify({
                "error": "No data in request body",
                "success": False
            }), 400
            
        if 'image' not in data:
            print("No image data in request")
            return jsonify({
                "error": "No image data in request body",
                "success": False
            }), 400

        image_string = data['image']
        
        # Validate base64 string
        if not image_string or not isinstance(image_string, str):
            print("Invalid image data format")
            return jsonify({
                "error": "Invalid image data format",
                "success": False
            }), 400

        print("Image data received, length:", len(image_string))
        
        # Save a copy of the uploaded image
        filename = f"yolo_upload_{int(time.time())}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Process image with YOLO
        try:
            # Convert base64 to image
            image = process_image_from_base64(image_string)
            
            # Perform detection
            detection_results = detect_road_damage(image, save_path)
            
            # Generate a URL for the saved image
            image_url = f"/static/uploads/{filename}"
            
            print(f"YOLO detection successful, found {detection_results['count']} objects")
            
            response = jsonify({
                "detections": detection_results['detections'],
                "annotated_image": detection_results['annotated_image'],
                "image_url": image_url,
                "success": True,
                "message": "YOLO detection successful"
            })
            
        except Exception as e:
            print(f"Error in YOLO processing: {e}")
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "success": False
            }), 500

        # Add CORS headers to response
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print(f"Error in YOLO detection endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Endpoint for YOLO model information
@app.route("/yolo-info", methods=["GET"])
def yolo_info():
    try:
        # Import here to avoid circular imports
        from yolo_utils import YOLO_MODEL
        
        if YOLO_MODEL is None:
            return jsonify({
                "status": "error",
                "message": "YOLO model not loaded",
                "success": False
            }), 500
            
        # Get model information
        model_info = {
            "model_type": "YOLOv5",
            "classes": list(YOLO_MODEL.names.values()),
            "confidence_threshold": float(YOLO_MODEL.conf),
            "status": "loaded",
            "success": True
        }
        
        return jsonify(model_info), 200
        
    except Exception as e:
        print(f"Error getting YOLO model info: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == "__main__":
    print("Starting AI server on http://127.0.0.1:5000")
    print("Health check available at: http://127.0.0.1:5000/health")
    print("ViT Prediction endpoint available at: http://127.0.0.1:5000/predict")
    print("YOLO Detection endpoint available at: http://127.0.0.1:5000/detect-yolo")
    print("YOLO Model Info available at: http://127.0.0.1:5000/yolo-info")
    app.run(host='0.0.0.0', port=5000, debug=True)