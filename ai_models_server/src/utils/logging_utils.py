"""
Logging utilities for AI Models Server
"""
import logging
import sys
from datetime import datetime
from config import LOG_LEVEL


def setup_logger(name: str = "ai_models_server"):
    """Setup and configure logger"""
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set log level
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def log_request(endpoint: str, method: str, image_size: int = None):
    """Log incoming requests"""
    logger = logging.getLogger("ai_models_server")
    
    log_message = f"Request: {method} {endpoint}"
    if image_size:
        log_message += f" (image size: {image_size} bytes)"
    
    logger.info(log_message)


def log_prediction(endpoint: str, success: bool, prediction: str = None, confidence: float = None):
    """Log prediction results"""
    logger = logging.getLogger("ai_models_server")
    
    if success:
        log_message = f"Prediction successful: {prediction}"
        if confidence:
            log_message += f" (confidence: {confidence:.2f})"
    else:
        log_message = f"Prediction failed for {endpoint}"
    
    logger.info(log_message)


def log_error(error: Exception, context: str = ""):
    """Log errors with context"""
    logger = logging.getLogger("ai_models_server")
    
    log_message = f"Error{' in ' + context if context else ''}: {str(error)}"
    logger.error(log_message, exc_info=True)


def log_model_status(model_name: str, loaded: bool, error: str = None):
    """Log model loading status"""
    logger = logging.getLogger("ai_models_server")
    
    if loaded:
        logger.info(f"Model '{model_name}' loaded successfully")
    else:
        logger.warning(f"Model '{model_name}' failed to load: {error}")


def log_fallback_mode(endpoint: str, reason: str):
    """Log when fallback mode is used"""
    logger = logging.getLogger("ai_models_server")
    logger.warning(f"Using fallback mode for {endpoint}: {reason}")


# Initialize logger
logger = setup_logger()
