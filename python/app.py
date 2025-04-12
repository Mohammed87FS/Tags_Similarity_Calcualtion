from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model once at startup
model = None

def load_model():
    global model
    if model is None:
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model



@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def js():
    return send_from_directory('.', 'script.js')

@app.route('/calculate-similarities', methods=['POST'])
def calculate_similarities():
    # Get the research fields from the request
    data = request.json
    research_fields = data.get('fields', [])
    
    if len(research_fields) < 2:
        return jsonify({"error": "Need at least 2 research fields to calculate similarities"}), 400
    
    # Extract field names and descriptions
    field_names = [field['name'] for field in research_fields]
    field_descriptions = [field['description'] for field in research_fields]
    
    # Load model
    model = load_model()
    
    # Compute embeddings
    print(f"Computing embeddings for {len(field_descriptions)} research field descriptions...")
    embeddings = model.encode(field_descriptions)
    
    # Compute similarity matrix
    print("Computing semantic similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create list of pairs with similarity scores
    similarity_pairs = []
    for i in range(len(field_names)):
        for j in range(i+1, len(field_names)):
            sim = float(similarity_matrix[i][j])  # Convert numpy float to Python float
            similarity_pairs.append({
                "field1": field_names[i],
                "field2": field_names[j],
                "similarity_score": sim
            })
    
    # Sort by similarity (highest first)
    similarity_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return jsonify({
        "pairs": similarity_pairs,
        "matrix": similarity_matrix.tolist()  # Convert numpy array to list
    })

if __name__ == '__main__':
    app.run(debug=True)