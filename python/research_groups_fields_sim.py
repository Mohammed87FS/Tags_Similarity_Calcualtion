import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_field_similarities(json_file_path, model_name='all-mpnet-base-v2', output_prefix='field_similarity'):
    """
    Calculate similarities between academic fields using sentence transformers with custom rules:
    - Fields in same category: similarity = 0.75 + (raw_similarity * 0.18)
    - Fields in different categories: similarity = raw_similarity
    
    Args:
        json_file_path: Path to JSON file with field descriptions
        model_name: Pre-trained sentence transformer model to use
        output_prefix: Prefix for output files
        
    Returns:
        Tuple containing:
        - Raw similarity DataFrame
        - Adjusted similarity DataFrame
        - List of field names
        - List of field categories
        - List of field-to-field comparisons with scores
    """
    print(f"Loading data from {json_file_path}")
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract fields, descriptions, and categories
    field_names = []
    field_descriptions = []
    field_categories = []
    
    for category in data['categories']:
        category_name = category['name']
        for field in category['fields']:
            field_names.append(field['name'])
            field_descriptions.append(field['description'])
            field_categories.append(category_name)
    
    print(f"Analyzing {len(field_names)} fields across {len(set(field_categories))} categories")
    
    # Load the sentence transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Encode all field descriptions
    print("Encoding field descriptions...")
    embeddings = model.encode(field_descriptions)
    
    # Calculate raw cosine similarity between all pairs
    print("Calculating similarities...")
    raw_similarity_matrix = cosine_similarity(embeddings)
    
    # Apply the custom similarity rule
    adjusted_similarity = np.zeros_like(raw_similarity_matrix)
    for i in range(len(field_names)):
        for j in range(len(field_names)):
            if i == j:
                # Same field, perfect similarity
                adjusted_similarity[i, j] = 1.0
            elif field_categories[i] == field_categories[j]:
                # Fields in same category: 0.75 + (raw_similarity * 0.18)
                adjusted_similarity[i, j] = 0.75 + (raw_similarity_matrix[i, j] * 0.18)
            else:
                # Fields in different categories: just use raw similarity
                adjusted_similarity[i, j] = raw_similarity_matrix[i, j]
    
    # Create DataFrames for better visualization and analysis
    raw_similarity_df = pd.DataFrame(raw_similarity_matrix, index=field_names, columns=field_names)
    adjusted_similarity_df = pd.DataFrame(adjusted_similarity, index=field_names, columns=field_names)
    
    # Save results to CSV files
    raw_similarity_df.to_csv(f"{output_prefix}_raw.csv")
    adjusted_similarity_df.to_csv(f"{output_prefix}_adjusted.csv")
    print(f"Saved similarity matrices to CSV files")
    

    # Create a list of dictionaries for Excel and JSON export
    comparison_data = []
    for i in range(len(field_names)):
        for j in range(len(field_names)):
            # Skip self-comparisons
            if i != j:
                comparison_data.append({
                    'field1': field_names[i],
                    'field2': field_names[j],
                    'similarity_score': float(adjusted_similarity[i, j])  # Convert numpy.float32 to Python float
                })
    
    # Create DataFrame for Excel export
    comparison_df = pd.DataFrame(comparison_data)
    
    # Export to Excel
    comparison_df.to_excel(f"{output_prefix}_comparisons.xlsx", index=False)
    print(f"Saved field comparisons to Excel: {output_prefix}_comparisons.xlsx")
    
    # Export to JSON
    with open(f"{output_prefix}_comparisons.json", 'w') as json_file:
        json.dump(comparison_data, json_file, indent=2)
    print(f"Saved field comparisons to JSON: {output_prefix}_comparisons.json")
    
    # Return both similarity matrices and the comparison data for further analysis
    return raw_similarity_df, adjusted_similarity_df, field_names, field_categories, comparison_data

def find_most_similar_fields(similarity_df, field_name, field_names, field_categories, top_n=5):
    """Find the most similar fields to a given field"""
    if field_name not in field_names:
        print(f"Field '{field_name}' not found.")
        return []
    
    idx = field_names.index(field_name)
    similarities = similarity_df.iloc[idx].sort_values(ascending=False)
    
    results = []
    for similar_field, score in similarities[1:top_n+1].items():  # Skip the first one (itself)
        similar_idx = field_names.index(similar_field)
        same_category = field_categories[idx] == field_categories[similar_idx]
        results.append({
            'field': similar_field,
            'score': float(score),
            'category': field_categories[similar_idx],
            'same_category': same_category
        })
    
    return results

def main():
    # Path to the JSON file with enhanced field descriptions
    json_file_path = "../research_groups_fields.json"
    
    # Calculate field similarities
    raw_similarities, adjusted_similarities, field_names, field_categories, comparison_data = calculate_field_similarities(
        json_file_path=json_file_path
    )
    
    # Example: Print top similar fields for selected fields
    example_fields = ["Artificial Intelligence", "Machine Learning", "Cybersecurity", "Game Design", "Health"]
    
    print("\n=== Example Field Similarity Analysis ===")
    for field in example_fields:
        similar_fields = find_most_similar_fields(
            adjusted_similarities, field, field_names, field_categories
        )
        
        field_idx = field_names.index(field)
        field_category = field_categories[field_idx]
        
        print(f"\nTop 5 fields most similar to '{field}' (Category: {field_category}):")
        for item in similar_fields:
            category_info = "(Same category)" if item['same_category'] else f"(Category: {item['category']})"
            print(f"- {item['field']}: {item['score']:.4f} {category_info}")
    
    # Example: Compare specific field pairs
    print("\n=== Field Pair Comparisons ===")
    field_pairs = [
        ("Artificial Intelligence", "Machine Learning"),
        ("Artificial Intelligence", "Cybersecurity"),
        ("Game Design", "Game Development"),
        ("Cybersecurity", "Information Security")
    ]
    
    for field1, field2 in field_pairs:
        if field1 in field_names and field2 in field_names:
            similarity = adjusted_similarities.loc[field1, field2]
            idx1 = field_names.index(field1)
            idx2 = field_names.index(field2)
            same_category = field_categories[idx1] == field_categories[idx2]
            
            category_info = "same category" if same_category else "different categories"
            print(f"Similarity between '{field1}' and '{field2}': {similarity:.4f} ({category_info})")

if __name__ == "__main__":
    main()