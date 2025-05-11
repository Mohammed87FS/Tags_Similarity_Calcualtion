#!/bin/bash

# This script handles the build process for Render deployment

echo "Starting build process..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install spaCy model
echo "Installing spaCy model..."
python -m spacy download en_core_web_sm

# Make render_startup.sh executable
echo "Making render_startup.sh executable..."
chmod +x render_startup.sh

echo "Build process complete!"
