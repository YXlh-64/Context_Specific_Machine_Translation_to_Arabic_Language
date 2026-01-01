from flask import Flask, redirect
from flask_cors import CORS
from app.api.config import config
from app.api.routes import api_bp
import os
import logging

def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Enable CORS using configured origins
    CORS(app, resources={r"/api/*": {"origins": app.config.get('CORS_ORIGINS', ['http://localhost:5173'])}})

    # Configure logging
    if app.config['DEBUG']:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Print to console
            ]
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    # Add root route for convenience (redirect to API index)
    @app.route('/', methods=['GET'])
    def root_redirect():
        return redirect('/api')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )