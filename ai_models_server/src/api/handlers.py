"""
Route handlers for AI Models Server
"""
import traceback
import random
from flask import request

from ..models.model_manager import model_manager
from ..utils.response_utils import (
    create_success_response, 
    create_error_response, 
    create_fallback_response,
    handle_preflight_request,
    add_cors_headers
)
from ..utils.validation_utils import (
    validate_request_json,
    validate_image_data,
    validate_summary_request,
    validate_confidence_threshold
)
from ..utils.image_utils import (
    create_mock_annotation,
    create_mock_yolo_detection,
    save_image_from_base64
)
from ..core.config import UPLOAD_FOLDER, API_VERSION


def handle_health_check():
    """Handle health check endpoint"""
    return create_success_response({
        "status": "healthy",
        "message": "AI server is running",
        "version": API_VERSION
    })


def handle_predict_endpoint():
    """Handle ViT prediction endpoint"""
    if request.method == "OPTIONS":
        return handle_preflight_request()
    
    print("📸 Prediction endpoint called")
    
    try:
        # Validate request
        is_valid, error_msg = validate_request_json(request)
        if not is_valid:
            return create_error_response(error_msg)
        
        data = request.get_json()
        is_valid, error_msg, image_string = validate_image_data(data)
        if not is_valid:
            return create_error_response(error_msg)
        
        print(f"📊 Image data received, length: {len(image_string)}")
        
        # Always try to use the actual model
        predict_func = model_manager.get_model('vit')
        if predict_func is None:
            print("⚠️  ViT model function not available, generating mock prediction")
            
            # Generate a random prediction
            damage_classes = ['D00', 'D10', 'D20', 'D30', 'D40', 'D43', 'D44', 'D50']
            prediction = random.choice(damage_classes)
            
            # Create annotated image
            annotated_image = create_mock_annotation(image_string, prediction)
            
            response_data = {
                "prediction": prediction,
                "annotated_image": annotated_image,
                "message": "Mock prediction generated (ViT model not available)"
            }
            return add_cors_headers(create_fallback_response(response_data))
        
        # Use actual model for prediction
        print("🔍 Using ViT model for prediction")
        predict_func = model_manager.get_model('vit')
        result = predict_func(image_string)
        
        if result.get("fallback", False) or "error" in result:
            return create_error_response(result.get("error", "Prediction failed"), 500)
        
        response_data = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "annotated_image": result.get("annotated_image"),
            "bbox": result["bbox"],
            "all_predictions": result.get("all_predictions", {}),
            "message": "Prediction successful"
        }
        
        return add_cors_headers(create_success_response(response_data))
        
    except Exception as e:
        print(f"❌ Error in prediction endpoint: {e}")
        traceback.print_exc()
        return create_error_response(f"Server error processing image: {str(e)}", 500)


def handle_summary_endpoint():
    """Handle summary generation endpoint"""
    if request.method == "OPTIONS":
        return handle_preflight_request()
    
    print("📝 Summary generation endpoint called")
    
    try:
        # Validate request
        is_valid, error_msg = validate_request_json(request)
        if not is_valid:
            return create_error_response(error_msg)
        
        data = request.get_json()
        is_valid, error_msg = validate_summary_request(data)
        if not is_valid:
            return create_error_response(error_msg)
        
        # Extract parameters
        location = data.get('location')
        damage_type = data.get('damageType')
        severity = data.get('severity')
        priority = data.get('priority')
        
        print(f"🏗️  Generating summary for {location}, {damage_type}, {severity}, priority {priority}")
        
        # Check if Gemini is available
        if not model_manager.is_model_available('gemini'):
            # Generate fallback summary
            fallback_summary = f"Road damage report for {location}: {damage_type} with {severity} severity. Priority level: {priority}. This is a fallback summary as the AI summary generator is not available."
            
            response_data = {
                "summary": fallback_summary,
                "message": "Fallback summary generated (Gemini not available)"
            }
            return add_cors_headers(create_fallback_response(response_data))
        
        # Generate actual summary
        generate_summary_func = model_manager.get_model('gemini')
        summary = generate_summary_func(location, damage_type, severity, priority)
        
        response_data = {
            "summary": summary,
            "message": "Summary generated successfully"
        }
        
        return add_cors_headers(create_success_response(response_data))
        
    except Exception as e:
        print(f"❌ Error in summary generation endpoint: {e}")
        traceback.print_exc()
        return create_error_response(f"Server error generating summary: {str(e)}", 500)


def handle_yolo_detection_endpoint():
    """Handle YOLO detection endpoint"""
    if request.method == "OPTIONS":
        return handle_preflight_request()
    
    print("🎯 YOLO detection endpoint called")
    
    try:
        # Validate request
        is_valid, error_msg = validate_request_json(request)
        if not is_valid:
            return create_error_response(error_msg)
        
        data = request.get_json()
        is_valid, error_msg, image_string = validate_image_data(data)
        if not is_valid:
            return create_error_response(error_msg)
        
        print(f"📊 Image data received, length: {len(image_string)}")
        
        # Save image
        success, save_path, image_url = save_image_from_base64(
            image_string, UPLOAD_FOLDER, "yolo_upload"
        )
        
        if not success:
            return create_error_response("Failed to save image")
        
        # Always try to use the YOLO model
        detect_func = model_manager.get_model('yolo')
        if detect_func is None:
            print("⚠️  YOLO detection function not available, using fallback detection")
            
            # Create mock detection
            mock_result = create_mock_yolo_detection(image_string)
            
            if not mock_result["success"]:
                return create_error_response(mock_result["error"])
            
            response_data = {
                "detections": mock_result["detections"],
                "annotated_image": mock_result["annotated_image"],
                "image_url": image_url,
                "message": "Fallback YOLO detection (model not available)"
            }
            return add_cors_headers(create_fallback_response(response_data))
        
        # Use actual YOLO model
        try:
            from ..utils.yolo_utils import process_image_from_base64
            
            image = process_image_from_base64(image_string)
            
            detect_func = model_manager.get_model('yolo')
            detection_results = detect_func(image, save_path)
            
            print(f"🎯 YOLO detection successful, found {detection_results['count']} objects")
            
            response_data = {
                "detections": detection_results['detections'],
                "annotated_image": detection_results['annotated_image'],
                "image_url": image_url,
                "message": "YOLO detection successful"
            }
            
            return add_cors_headers(create_success_response(response_data))
            
        except Exception as e:
            print(f"❌ Error in YOLO processing: {e}")
            traceback.print_exc()
            
            # Fall back to mock detection
            mock_result = create_mock_yolo_detection(image_string)
            
            response_data = {
                "error": str(e),
                "detections": mock_result.get("detections", []),
                "annotated_image": mock_result.get("annotated_image"),
                "image_url": image_url,
                "message": "Error processing image with YOLO, using fallback"
            }
            
            return add_cors_headers(create_fallback_response(response_data))
    
    except Exception as e:
        print(f"❌ Error in YOLO detection endpoint: {e}")
        traceback.print_exc()
        return create_error_response(str(e), 500)


def handle_road_classification_endpoint():
    """Handle road classification endpoint"""
    if request.method == "OPTIONS":
        return handle_preflight_request()
    
    print("🛣️  Road classification endpoint called")
    
    try:
        # Always try to use the road classifier
        validate_func = model_manager.get_model('road_classifier')
        if validate_func is None:
            print("⚠️  Road classifier function not available, using fallback")
            response_data = {
                "isRoad": True,  # Default to True to allow processing
                "confidence": 0.7,
                "message": "Road surface detected (using fallback classifier)",
                "features": {"fallback": True}
            }
            return add_cors_headers(create_fallback_response(response_data))
        
        # Validate request
        is_valid, error_msg = validate_request_json(request)
        if not is_valid:
            return create_error_response(error_msg)
        
        data = request.get_json()
        is_valid, error_msg, image_string = validate_image_data(data)
        if not is_valid:
            return create_error_response(error_msg)
        
        print(f"📊 Image data received for road classification, length: {len(image_string)}")
        
        # Get confidence threshold (default to 0.6 if not specified)
        confidence_threshold = validate_confidence_threshold(data)
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Save image for debugging if needed
        success, save_path, image_url = save_image_from_base64(
            image_string, UPLOAD_FOLDER, "road_classification"
        )
        
        if not success:
            print("⚠️ Failed to save image, continuing with classification")
        
        try:
            # Validate image with road classifier
            validation_result = validate_func(image_string)
            
            # Apply custom threshold if needed
            is_road = validation_result["is_road"]
            confidence = validation_result["confidence"]
            
            # Override if confidence is below threshold
            if confidence < confidence_threshold:
                is_road = False
                
            print(f"🛣️  Road classification result: is_road={is_road}, confidence={confidence:.2f}")
            
            # Add model and heuristic details to response
            model_prediction = validation_result.get("model_prediction")
            model_confidence = validation_result.get("model_confidence", 0.0)
            heuristic_features = validation_result.get("heuristic_features", {})
            
            if is_road:
                message = "Road surface detected"
                if model_prediction is not None:
                    message += f" (model confidence: {model_confidence:.2f})"
            else:
                message = "Not a road surface"
                if model_prediction is not None:
                    message += f" (model confidence: {model_confidence:.2f})"
            
            response_data = {
                "isRoad": bool(is_road),
                "confidence": float(confidence),
                "message": message,
                "modelPrediction": model_prediction,
                "modelConfidence": float(model_confidence),
                "heuristicScore": float(heuristic_features.get("road_score", 0.0)),
                "features": heuristic_features,
                "imageUrl": image_url,
                "fallback": bool(validation_result.get("fallback", False))
            }
            
            return add_cors_headers(create_success_response(response_data))
            
        except Exception as classifier_error:
            print(f"❌ Error in road classification: {classifier_error}")
            traceback.print_exc()
            # Fall back to permissive response
            response_data = {
                "isRoad": True,  # Default to True to allow processing
                "confidence": 0.6,
                "message": f"Road surface detected (classifier error: {str(classifier_error)}, using fallback)",
                "features": {"error": str(classifier_error)},
                "imageUrl": image_url,
                "fallback": True
            }
            return add_cors_headers(create_fallback_response(response_data))
        
    except Exception as e:
        print(f"❌ Error in road classification endpoint: {e}")
        traceback.print_exc()
        return create_error_response(f"Server error processing image: {str(e)}", 500)


def handle_yolo_info_endpoint():
    """Handle YOLO model info endpoint"""
    try:
        yolo_status = model_manager.get_model_status('yolo')
        
        if not yolo_status['loaded']:
            return create_success_response({
                "status": "error",
                "message": "YOLO model not loaded",
                "modelType": "YOLOv5",
                "fallbackMode": True,
                "error": yolo_status['error']
            })
        
        # Get model information
        yolo_model = model_manager.get_model('yolo_model')
        
        if yolo_model is None:
            return create_success_response({
                "status": "error",
                "message": "YOLO model not loaded",
                "modelType": "YOLOv5",
                "fallbackMode": True
            })
        
        model_info = {
            "model_type": "YOLOv5",
            "classes": list(yolo_model.names.values()),
            "confidence_threshold": float(yolo_model.conf),
            "status": "loaded",
            "fallbackMode": False
        }
        
        return create_success_response(model_info)
        
    except Exception as e:
        print(f"❌ Error getting YOLO model info: {e}")
        traceback.print_exc()
        return create_success_response({
            "error": str(e),
            "modelType": "YOLOv5",
            "fallbackMode": True,
            "status": "error"
        })


def handle_models_status_endpoint():
    """Handle models status endpoint"""
    try:
        all_status = model_manager.get_all_model_status()
        missing_models = model_manager.get_missing_models()
        
        response_data = {
            "models": all_status,
            "missing_models": missing_models,
            "server_status": "running",
            "version": API_VERSION
        }
        
        return create_success_response(response_data)
        
    except Exception as e:
        print(f"❌ Error getting models status: {e}")
        traceback.print_exc()
        return create_error_response(f"Error getting models status: {str(e)}", 500)
