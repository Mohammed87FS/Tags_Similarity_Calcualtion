"""
WSGI entry point for the Research Field Similarity Application.
This file imports the Flask application from the python subdirectory
and makes it accessible to the WSGI server.
"""

import sys
import os

# Add the python directory to the path so we can import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Import the Flask app instance from your main application file
# The Gunicorn command will look for an object named 'application' in this file.
from python.app import app as application

# If you want to be able to run this file directly for development (optional)
if __name__ == "__main__":
    application.run()
