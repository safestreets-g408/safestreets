"""
API routes for the AI Models Server
"""
from flask import send_from_directory
from .handlers import (
    handle_health_check,
    handle_predict_endpoint,
    handle_summary_endpoint,
    handle_yolo_detection_endpoint,
    handle_road_classification_endpoint,
    handle_yolo_info_endpoint,
    handle_models_status_endpoint
)
from ..core.config import UPLOAD_FOLDER


def register_routes(app):
    """Register all application routes"""
    
    # Static file serving
    @app.route('/static/uploads/<filename>')
    def serve_uploaded_file(filename):
        return send_from_directory(UPLOAD_FOLDER, filename)
    
    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health_check():
        return handle_health_check()
    
    # API-specific health check endpoint for Docker/Kubernetes
    @app.route("/api/health", methods=["GET"])
    def api_health_check():
        return handle_health_check()
    
    # ViT prediction endpoint
    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        return handle_predict_endpoint()
    
    # Summary generation endpoint
    @app.route("/generate-summary", methods=["POST", "OPTIONS"])
    def generate_summary():
        return handle_summary_endpoint()
    
    # YOLO detection endpoint
    @app.route("/detect-yolo", methods=["POST", "OPTIONS"])
    def detect_yolo():
        return handle_yolo_detection_endpoint()
    
    # Road classification endpoint
    @app.route("/classify-road", methods=["POST", "OPTIONS"])
    def classify_road():
        return handle_road_classification_endpoint()
    
    # YOLO model info endpoint
    @app.route("/yolo-info", methods=["GET"])
    def yolo_info():
        return handle_yolo_info_endpoint()
    
    # Models status endpoint
    @app.route("/models-status", methods=["GET"])
    def models_status():
        return handle_models_status_endpoint()
