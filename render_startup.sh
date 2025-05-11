#!/bin/bash

# Ensure spaCy model is installed
echo "Installing spaCy model..."
python -m spacy download en_core_web_sm

# Set environment variables from .env.render if not already set
if [ -f .env.render ]; then
  echo "Loading environment variables from .env.render"
  export $(grep -v '^#' .env.render | xargs)
fi

# Start the application with gunicorn
echo "Starting application with gunicorn..."
exec gunicorn wsgi:app
