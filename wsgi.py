"""
WSGI entry point for the Research Field Similarity Application.
This file imports the Flask application from the python subdirectory
and makes it accessible to the WSGI server.
"""

from python.app import app

# This allows the app to be run by any WSGI server
if __name__ == "__main__":
    app.run()
