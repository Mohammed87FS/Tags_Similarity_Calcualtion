# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os


app = Flask(__name__)
CORS(app) 


research_fields = []
similarity_data = []
similarity_lookup = {}  

def load_research_fields():
    """Load the research fields data from nested research_groups_fields.json"""
    global research_fields
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, '../research_groups_fields.json')
        print(f"Looking for research fields at: {json_path}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
      
            research_fields = []
            for category in data.get('categories', []):
                for field in category.get('fields', []):
                    research_fields.append(field)
                    
            print(f"Loaded {len(research_fields)} research fields from nested categories.")
        else:
            print("No research_groups_fields.json file found. Starting with empty list.")
            research_fields = []
    except Exception as e:
        print(f"Error loading research fields: {e}")
        research_fields = []

def load_enhanced_similarities():
    """Load the enhanced similarities from the JSON file"""
    global similarity_data, similarity_lookup
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'outputs_sub_groups/field_similarities.json')
        print(f"Looking for enhanced similarities at: {json_path}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as file:
                similarity_data = json.load(file)
            print(f"Loaded {len(similarity_data)} similarity pairs from enhanced file.")
            
            # Create lookup dictionary for quick access
            for pair in similarity_data:
                # Store both directions for easy lookup
                key1 = f"{pair['field1']}|{pair['field2']}"
                key2 = f"{pair['field2']}|{pair['field1']}"
                similarity_lookup[key1] = pair['similarity_score']
                similarity_lookup[key2] = pair['similarity_score']
        else:
            print("No enhanced_research_field_similarities.json file found.")
            similarity_data = []
    except Exception as e:
        print(f"Error loading enhanced similarities: {e}")
        similarity_data = []

# Load data on startup
load_research_fields()
load_enhanced_similarities()

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
    
    # Ensure fields exist in our data
    field1_exists = any(f['name'] == field1 for f in research_fields)
    field2_exists = any(f['name'] == field2 for f in research_fields)
    
    if not field1_exists or not field2_exists:
        return jsonify({"error": "One or both fields not found in research fields"}), 400
    
    # Use the pre-computed enhanced similarity
    lookup_key = f"{field1}|{field2}"
    
    # If we have this pair in our lookup
    if lookup_key in similarity_lookup:
        similarity = similarity_lookup[lookup_key]
    else:
        # Fields exist but similarity not in pre-computed data
        # This could happen if the enhanced similarity file is incomplete
        return jsonify({
            "error": "Similarity for this pair not found in pre-computed data",
            "field1": field1,
            "field2": field2
        }), 400
    
    return jsonify({
        "field1": field1,
        "field2": field2,
        "similarity_score": similarity
    })

@app.route('/get-field-network', methods=['POST'])
def get_field_network():
    """Get network data showing how a field is connected to other fields"""
    data = request.json
    field_name = data.get('field', '')
    similarity_threshold = data.get('threshold', 0.5)  # Default threshold
    
    # Ensure field exists
    if not any(f['name'] == field_name for f in research_fields):
        return jsonify({"error": f"Field '{field_name}' not found in research fields"}), 400
    
    # Find all related fields above the threshold
    nodes = [{"id": field_name, "group": 1}]  # Center node (the selected field)
    links = []
    related_fields = set()
    
    # First level connections
    for pair in similarity_data:
        if pair['field1'] == field_name and pair['similarity_score'] >= similarity_threshold:
            related_fields.add(pair['field2'])
            links.append({
                "source": field_name,
                "target": pair['field2'],
                "value": pair['similarity_score']
            })
        elif pair['field2'] == field_name and pair['similarity_score'] >= similarity_threshold:
            related_fields.add(pair['field1'])
            links.append({
                "source": field_name,
                "target": pair['field1'],
                "value": pair['similarity_score']
            })
    
    # Add nodes for all related fields
    for field in related_fields:
        nodes.append({"id": field, "group": 2})  # Group 2 for direct connections
    
    # Second level connections (bridges) - connections between related fields
    for pair in similarity_data:
        if pair['field1'] in related_fields and pair['field2'] in related_fields:
            if pair['similarity_score'] >= similarity_threshold:
                links.append({
                    "source": pair['field1'],
                    "target": pair['field2'],
                    "value": pair['similarity_score']
                })
    
    return jsonify({
        "nodes": nodes,
        "links": links,
        "center": field_name
    })

@app.route('/get-all-similarities', methods=['GET'])
def get_all_similarities():
    """Return all pre-computed similarity pairs"""
    return jsonify({"pairs": similarity_data})

@app.route('/get-similar-fields', methods=['POST'])
def get_similar_fields():
    """Get the most similar fields to a given field"""
    data = request.json
    field_name = data.get('field', '')
    limit = data.get('limit', 10)  # Default to top 10
    
    # Ensure field exists
    if not any(f['name'] == field_name for f in research_fields):
        return jsonify({"error": f"Field '{field_name}' not found in research fields"}), 400
    
    # Find all pairs involving this field
    similar_fields = []
    for pair in similarity_data:
        if pair['field1'] == field_name:
            similar_fields.append({
                "field": pair['field2'],
                "similarity_score": pair['similarity_score']
            })
        elif pair['field2'] == field_name:
            similar_fields.append({
                "field": pair['field1'],
                "similarity_score": pair['similarity_score']
            })
    
    # Sort by similarity score (descending)
    similar_fields.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Limit results
    similar_fields = similar_fields[:limit]
    
    return jsonify({
        "field": field_name,
        "similar_fields": similar_fields
    })

@app.route('/get-field-details', methods=['POST'])
def get_field_details():
    """Get detailed information about a specific field"""
    data = request.json
    field_name = data.get('field', '')
    
    # Find the field in our data
    field_data = next((f for f in research_fields if f['name'] == field_name), None)
    
    if not field_data:
        return jsonify({"error": f"Field '{field_name}' not found"}), 400
    
    return jsonify(field_data)

@app.route('/search-fields', methods=['POST'])
def search_fields():
    """Search for fields by keyword in name or description"""
    data = request.json
    query = data.get('query', '').lower()
    
    if not query:
        return jsonify({"error": "No search query provided"}), 400
    
    results = []
    for field in research_fields:
        # Check if query appears in name or description
        if query in field['name'].lower() or query in field['description'].lower():
            results.append({
                "name": field['name'],
                "description": field['description'][:200] + "..." if len(field['description']) > 200 else field['description']
            })
    
    return jsonify({
        "query": query,
        "results": results,
        "count": len(results)
    })

if __name__ == '__main__':
    app.run(debug=True)