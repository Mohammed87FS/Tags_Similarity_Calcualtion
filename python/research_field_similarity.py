"""
Research Field Similarity Analyzer

This script analyzes the semantic similarity between research fields based on 
their detailed descriptions using state-of-the-art language models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sentence_transformers import SentenceTransformer
import networkx as nx
import pandas as pd 
import json  # Added JSON module

def load_research_fields(json_file_path="research_fields.json"):
    """Load research fields from the JSON file."""
    print(f"Loading research fields from {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            research_fields = json.load(file)
        print(f"Successfully loaded {len(research_fields)} research fields.")
        return research_fields
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file.")
        exit(1)

def load_model():
    """Load the sentence transformer model for semantic analysis."""
    print("Loading sentence transformer model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_field_data(research_fields):
    """Extract field names and descriptions from the dataset."""
    field_names = [field["name"] for field in research_fields]
    field_descriptions = [field["description"].strip() for field in research_fields]
    return field_names, field_descriptions

def compute_embeddings(descriptions, model):
    """Compute embeddings for the field descriptions."""
    print(f"Computing embeddings for {len(descriptions)} research field descriptions...")
    return model.encode(descriptions)

def compute_similarity_matrix(embeddings):
    """Compute the cosine similarity matrix between embeddings."""
    print("Computing semantic similarity matrix...")
    return cosine_similarity(embeddings)

def display_similarity_results(field_names, similarity_matrix):
    """Display the most significant similarity pairs."""
    print("\nMost Significant Relationships Between Research Fields:")
    print("=" * 60)
    
    # Create a list of pairs with their similarity scores
    pairs = []
    for i in range(len(field_names)):
        for j in range(i+1, len(field_names)):  # Only upper triangle to avoid duplicates
            sim = similarity_matrix[i][j]
            pairs.append((field_names[i], field_names[j], sim))
    
    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
  
    
    return pairs

def export_to_json(similarity_pairs, output_path="research_field_similarities.json"):
    """
    Export the similarity pairs to a JSON file.
    
    Args:
        similarity_pairs: List of tuples (field1, field2, similarity_score)
        output_path: Path to save the JSON file
    """
    print(f"\nExporting similarity data to {output_path}...")
    
    # Convert list of tuples to list of dictionaries for JSON
    similarity_data = []
    for field1, field2, score in similarity_pairs:
        similarity_data.append({
            "field1": field1,
            "field2": field2,
            "similarity_score": float(score)  # Convert numpy float to Python float for JSON serialization
        })
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(similarity_data, json_file, indent=2)
    
    print(f"Data successfully exported to {output_path}")


def export_to_excel(similarity_pairs, output_path="research_field_similarities.xlsx"):
    """
    Export the similarity pairs to an Excel file.

    Args:
        similarity_pairs: List of tuples (field1, field2, similarity_score)
        output_path: Path to save the Excel file
    """
    print(f"\nExporting similarity data to {output_path}...")

    # Create a DataFrame from the similarity pairs
    df = pd.DataFrame(similarity_pairs, columns=["Field 1", "Field 2", "Similarity Score"])

    # Export to Excel
    df.to_excel(output_path, index=False)

    print(f"Data successfully exported to {output_path}")

def main():
    """Main function to run the research field similarity analysis."""
    print("=" * 60)
    print("  RESEARCH FIELD SEMANTIC SIMILARITY ANALYZER")
    print("=" * 60)
    
    # Load research fields from JSON file
    research_fields = load_research_fields()
    
    # Load model and prepare data
    model = load_model()
    field_names, field_descriptions = extract_field_data(research_fields)
    
    # Compute embeddings and similarity
    embeddings = compute_embeddings(field_descriptions, model)
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Get similarity pairs and export to JSON
    similarity_pairs = display_similarity_results(field_names, similarity_matrix)
    export_to_json(similarity_pairs)
    export_to_excel(similarity_pairs)

if __name__ == "__main__":
    main()