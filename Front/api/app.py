from flask import Flask, redirect
from flask_cors import CORS
from api.config import config
from api.routes import api_bp
import os

def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Configure CORS
    # Allow configured origins; in development also accept all localhost ports
    cors_origins = app.config.get('CORS_ORIGINS')
    # If a comma-separated list, pass it directly (flask-cors accepts list or string)
    CORS(app, origins=cors_origins)

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