"""
WSGI entry point for the Research Field Similarity Application.
This file imports the Flask application from the python subdirectory
and makes it accessible to the WSGI server.
"""

import sys
import os

# Add the python directory to the path so we can import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Now import the app
from app import app

# This allows the app to be run by any WSGI server
if __name__ == "__main__":
    app.run()
