from flask import Flask, redirect
from flask_cors import CORS
from config import config
from routes import api_bp
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
    
    # Preload Gemma model to avoid delay on first translation
    # Only preload in the main process (not Flask's reloader process)
    # Check if we're in the reloader process by looking for WERKZEUG_RUN_MAIN
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    skip_preload = os.environ.get('SKIP_MODEL_PRELOAD', 'false').lower() == 'true'
    
    if is_reloader_process and not skip_preload:
        print("\n" + "="*60)
        print("🚀 PRELOADING GEMMA MODEL...")
        print("="*60)
        try:
            from rag_service import preload_gemma_model
            preload_gemma_model()
            print("="*60)
            print("✅ GEMMA MODEL LOADED SUCCESSFULLY!")
            print("="*60 + "\n")
        except Exception as e:
            print("="*60)
            print(f"⚠️  WARNING: Failed to preload Gemma model: {e}")
            print("Model will be loaded on first translation request.")
            print("="*60 + "\n")
    elif skip_preload:
        print("\n" + "="*60)
        print("⏭️  MODEL PRELOAD SKIPPED (SKIP_MODEL_PRELOAD=true)")
        print("Model will load on first translation request.")
        print("="*60 + "\n")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )