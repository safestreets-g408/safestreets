import gunicorn.app.base
import os
from app import main

class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    options = {
        'bind': f'0.0.0.0:{port}',
        'workers': 1,
        'timeout': 120,
    }
    
    # Create the Flask app
    flask_app = main()
    
    # Run with Gunicorn
    StandaloneApplication(flask_app, options).run()
