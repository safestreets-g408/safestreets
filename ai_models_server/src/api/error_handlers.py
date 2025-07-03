"""
Error handlers for the AI Models Server
"""
from ..utils.response_utils import create_error_response


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found(error):
        return create_error_response("Endpoint not found", 404)

    @app.errorhandler(405)
    def method_not_allowed(error):
        return create_error_response("Method not allowed", 405)

    @app.errorhandler(500)
    def internal_error(error):
        return create_error_response("Internal server error", 500)

    @app.errorhandler(413)
    def payload_too_large(error):
        return create_error_response("File too large", 413)
