#!/bin/bash
# filepath: c:\Users\moham\Coding-Projects\Tags_Similarity_Calcualtion\build.sh

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Ensure data directory exists
mkdir -p python/data