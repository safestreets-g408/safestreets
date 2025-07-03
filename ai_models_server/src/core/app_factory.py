"""
App factory for creating Flask application instances
"""
import os
from flask import Flask, send_from_directory
from flask_cors import CORS

from .config import (
    CORS_ORIGINS, MAX_CONTENT_LENGTH, UPLOAD_FOLDER
)
from ..api.routes import register_routes
from ..api.error_handlers import register_error_handlers


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure Flask
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    # Configure CORS
    CORS(app, 
         resources={r"/*": {"origins": CORS_ORIGINS}},
         allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
         methods=["GET", "POST", "OPTIONS"],
         supports_credentials=True
    )
    
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Register routes
    register_routes(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app
