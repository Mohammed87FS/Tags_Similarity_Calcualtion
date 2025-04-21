"""
Main application entry point for research field similarity app.
"""

import os
import logging
from flask import Flask, render_template

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    # Initialize Flask application
    app = Flask(__name__)
    
    # Import from routes module
    from routes.api import (
        api_bp, add_field, get_subgroups, get_similarity, 
        get_all_similarities_for_field, download_similarities, test
    )
    
    # Register the blueprint with /api prefix
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register the same routes without prefix for backward compatibility
    app.add_url_rule('/add_field', view_func=add_field, methods=['POST'])
    app.add_url_rule('/get_subgroups', view_func=get_subgroups)
    app.add_url_rule('/get_similarity', view_func=get_similarity)
    app.add_url_rule('/get_all_similarities_for_field', view_func=get_all_similarities_for_field)
    app.add_url_rule('/download_similarities', view_func=download_similarities)
    app.add_url_rule('/test', view_func=test)
    
    # Home route
    @app.route('/')
    def index():
        """Home page."""
        from services.data_service import DataService
        
        # Load data
        data_service = DataService()
        nested_data, similarities = data_service.load_data()
        field_names = data_service.get_all_field_names(nested_data)
        groups, subgroups = data_service.get_all_groups_and_subgroups(nested_data)
        
        # Debug information
        logger.info(f"Rendering template with {len(field_names)} fields and {len(groups)} groups")
        
        return render_template(
            'final_app.html', 
            field_names=field_names,
            groups=groups,
            subgroups=subgroups
        )
    
    # Ensure data directory exists
    from config import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    
    return app

# Application factory pattern
app = create_app()

if __name__ == '__main__':
    # Run the application in debug mode
    app.run(debug=True)