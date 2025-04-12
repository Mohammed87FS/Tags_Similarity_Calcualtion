# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
research_fields = []
field_embeddings = {}

def load_model():
    global model
    if model is None:
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def load_research_fields():
    global research_fields
    try:
     
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, '../research_fields.json')
        print(f"Looking for file at: {json_path}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as file:
                research_fields = json.load(file)
            print(f"Loaded {len(research_fields)} research fields from file.")
        else:
            print("No research_fields.json file found. Starting with empty list.")
            research_fields = []
    except Exception as e:
        print(f"Error loading research fields: {e}")
        research_fields = []

def precompute_embeddings():
    global field_embeddings
    if not research_fields:
        return
        
    model = load_model()
    for field in research_fields:
        if field['name'] not in field_embeddings:
            field_embeddings[field['name']] = model.encode(field['description'])
    
    print(f"Precomputed embeddings for {len(field_embeddings)} fields")

# Load research fields and precompute embeddings on startup
load_research_fields()
precompute_embeddings()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def js():
    return send_from_directory('.', 'script.js')

@app.route('/get-fields', methods=['GET'])
def get_fields():
    field_names = [field['name'] for field in research_fields]
    return jsonify({"fields": field_names})

@app.route('/compare-fields', methods=['POST'])
def compare_fields():
    data = request.json
    field1 = data.get('field1', '')
    field2 = data.get('field2', '')
    
    # Find the fields in our data
    field1_data = next((f for f in research_fields if f['name'] == field1), None)
    field2_data = next((f for f in research_fields if f['name'] == field2), None)
    
    if not field1_data or not field2_data:
        return jsonify({"error": "One or both fields not found"}), 400
    
    # Get embeddings
    model = load_model()
    
    # Check if embeddings are already computed
    if field1 in field_embeddings:
        embedding1 = field_embeddings[field1]
    else:
        embedding1 = model.encode(field1_data['description'])
        field_embeddings[field1] = embedding1
    
    if field2 in field_embeddings:
        embedding2 = field_embeddings[field2]
    else:
        embedding2 = model.encode(field2_data['description'])
        field_embeddings[field2] = embedding2
    
    # Calculate similarity
    similarity = float(cosine_similarity([embedding1], [embedding2])[0][0])
    
    return jsonify({
        "field1": field1,
        "field2": field2,
        "similarity_score": similarity
    })
@app.route('/calculate-similarities', methods=['POST'])
def calculate_similarities():
    global research_fields  # Declare global ONCE at the beginning of the function
    
    data = request.json
    new_fields = data.get('fields', [])
    
    # Combine with existing fields if any
    combined_fields = research_fields.copy()
    
    # Add any new fields that don't exist already
    existing_names = {field['name'] for field in combined_fields}
    for field in new_fields:
        if field['name'] not in existing_names:
            combined_fields.append(field)
            existing_names.add(field['name'])
    
    if len(combined_fields) < 2:
        return jsonify({"error": "Need at least 2 research fields to calculate similarities"}), 400
    
    # Extract field names and descriptions
    field_names = [field['name'] for field in combined_fields]
    field_descriptions = [field['description'] for field in combined_fields]
    
    # Load model
    model = load_model()
    
    # Compute embeddings for all fields
    print(f"Computing embeddings for {len(field_descriptions)} research field descriptions...")
    all_embeddings = []
    for i, (name, desc) in enumerate(zip(field_names, field_descriptions)):
        if name in field_embeddings:
            all_embeddings.append(field_embeddings[name])
        else:
            embedding = model.encode(desc)
            field_embeddings[name] = embedding
            all_embeddings.append(embedding)
    
    all_embeddings = np.array(all_embeddings)
    
    # Compute similarity matrix
    print("Computing semantic similarity matrix...")
    similarity_matrix = cosine_similarity(all_embeddings)
    
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
    
    # Save all fields to JSON file for future use
    with open('research_fields.json', 'w', encoding='utf-8') as f:
        json.dump(combined_fields, f, indent=2)
    
    # Update our global research fields
    research_fields = combined_fields  # No need to redeclare global here
    
    return jsonify({
        "pairs": similarity_pairs,
        "matrix": similarity_matrix.tolist()  # Convert numpy array to list
    })

if __name__ == '__main__':
    app.run(debug=True)