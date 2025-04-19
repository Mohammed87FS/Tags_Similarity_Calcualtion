import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import os
from typing import Dict, List, Tuple, Set

#############################################################
#                   CONFIGURATION SECTION                   #
#############################################################

# File path for the JSON data
JSON_FILE_PATH = "../nested_descriptions_research_groups.json"  # Use your existing JSON file

# Sentence Transformer model to use
# Options include: 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L3-v2', 'all-mpnet-base-v2'
MODEL_NAME = 'all-mpnet-base-v2'

# Similarity calculation parameters
SAME_GROUP_BASELINE = 0.7       # Baseline similarity for fields in same general group
SAME_SUBGROUP_BASELINE = 0.8    # Baseline similarity for fields in same subgroup
SIMILARITY_WEIGHT_SUB = 0.2        # Weight of calculated similarity to add to baseline
SIMILARITY_WEIGHT_GENERAL = 0.15        # Weight of calculated similarity to add to baseline
MAX_CROSS_GROUP_SIMILARITY = 0.8  # Maximum similarity for fields not in same group/subgroup

# Field description property weights (must sum to 1.0)
DESCRIPTION_WEIGHTS = {
    "definition": 0.50,
    "methodologies": 0.25,
    "applications": 0.00,
    "technologies": 0.00,
    "challenges": 0.15,
    "future_directions": 0.10
}

# Verify weights sum to 1.0
if abs(sum(DESCRIPTION_WEIGHTS.values()) - 1.0) > 0.001:
    raise ValueError(f"Description weights must sum to 1.0, got {sum(DESCRIPTION_WEIGHTS.values())}")

# Output configuration
OUTPUT_CSV = "outputs_sub_groups/field_similarities.csv"
OUTPUT_JSON = "outputs_sub_groups/field_similarities.json"  

GENERATE_HEATMAP = False
HEATMAP_FILENAME = "outputs_sub_groups/field_similarities_heatmap.png"
TOP_N_SIMILAR = 5  # Number of most similar fields to display for each field

#############################################################
#                     UTILITY FUNCTIONS                     #
#############################################################

def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON.")
        return None

def extract_fields_info(data: Dict) -> Tuple[List[Dict], Dict, Dict]:
    """
    Extract fields and their relationships from the JSON data.
    
    Returns:
        - List of field dictionaries with name and description
        - Dictionary mapping field names to their group
        - Dictionary mapping field names to their subgroup
    """
    all_fields = []
    field_to_group = {}
    field_to_subgroup = {}
    
    for category in data["categories"]:
        group_name = category["name"]
        
        for subgroup in category["subgroups"]:
            subgroup_name = subgroup["name"]
            
            for field in subgroup["fields"]:
                field_name = field["name"]
                field_description = field["description"]
                
                all_fields.append({
                    "name": field_name,
                    "description": field_description
                })
                
                field_to_group[field_name] = group_name
                field_to_subgroup[field_name] = subgroup_name
    
    return all_fields, field_to_group, field_to_subgroup

def calculate_field_embeddings(fields: List[Dict], model: SentenceTransformer) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate embeddings for each property of each field's description.
    
    Args:
        fields: List of field dictionaries with name and description
        model: Loaded SentenceTransformer model
        
    Returns:
        Dictionary mapping field names to their property embeddings
    """
    field_embeddings = {}
    
    for field in fields:
        field_name = field["name"]
        description = field["description"]
        
        # Create embeddings for each property of the description
        property_embeddings = {}
        for prop, text in description.items():
            if prop in DESCRIPTION_WEIGHTS:
                embedding = model.encode(text, show_progress_bar=False)
                property_embeddings[prop] = embedding
        
        field_embeddings[field_name] = property_embeddings
    
    return field_embeddings

def calculate_property_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def calculate_weighted_similarity(
    field1_embeddings: Dict[str, np.ndarray],
    field2_embeddings: Dict[str, np.ndarray]
) -> float:
    """
    Calculate weighted similarity between two fields based on their property embeddings.
    
    Args:
        field1_embeddings: Dictionary of property embeddings for field 1
        field2_embeddings: Dictionary of property embeddings for field 2
        
    Returns:
        Weighted similarity score
    """
    weighted_sim = 0.0
    total_weight = 0.0
    
    for prop, weight in DESCRIPTION_WEIGHTS.items():
        # Skip if either field doesn't have this property
        if prop not in field1_embeddings or prop not in field2_embeddings:
            continue
        
        # Calculate similarity for this property
        prop_sim = calculate_property_similarity(field1_embeddings[prop], field2_embeddings[prop])
        weighted_sim += prop_sim * weight
        total_weight += weight
    
    # Normalize by the actual weights used (in case some properties were missing)
    if total_weight > 0:
        weighted_sim /= total_weight
    
    return weighted_sim

def calculate_semantic_similarities(fields: List[Dict]) -> pd.DataFrame:
    """
    Calculate semantic similarities between all field descriptions using Sentence Transformers
    and weighted averages of property similarities.
    """
    model = SentenceTransformer(MODEL_NAME)
    field_names = [field["name"] for field in fields]
    
    # Calculate embeddings for all field properties
    print("Calculating embeddings for field properties...")
    field_embeddings = calculate_field_embeddings(fields, model)
    
    # Calculate weighted similarities between all field pairs
    print("Calculating weighted similarities between fields...")
    similarities = np.zeros((len(fields), len(fields)))
    
    for i, field1_name in enumerate(field_names):
        for j, field2_name in enumerate(field_names):
            if i == j:
                similarities[i, j] = 1.0  # Self-similarity
            else:
                # Calculate weighted similarity between the two fields
                weighted_sim = calculate_weighted_similarity(
                    field_embeddings[field1_name],
                    field_embeddings[field2_name]
                )
                similarities[i, j] = weighted_sim
    
    # Create DataFrame of similarities
    similarity_df = pd.DataFrame(similarities, index=field_names, columns=field_names)
    return similarity_df

def adjust_similarities(
    similarity_df: pd.DataFrame, 
    field_to_group: Dict[str, str], 
    field_to_subgroup: Dict[str, str]
) -> pd.DataFrame:
    """
    Adjust the similarity scores based on group and subgroup relationships.
    
    Rules:
    1. Same subgroup: baseline_subgroup + (sim * weight)
    2. Same group but different subgroup: baseline_group + (sim * weight)
    3. Different group: Normalize between 0 and max_cross_group
    """
    field_names = similarity_df.index.tolist()
    
    # Create a copy of the original similarities for scaling between 0-1
    raw_similarities = similarity_df.copy()
    
    # Create a new DataFrame for adjusted similarities
    adjusted_df = pd.DataFrame(0.0, index=field_names, columns=field_names)
    
    # Create a cross-group similarity matrix to later scale
    cross_group_df = pd.DataFrame(0.0, index=field_names, columns=field_names)
    cross_group_mask = np.zeros((len(field_names), len(field_names)), dtype=bool)
    
    for i, field1 in enumerate(field_names):
        for j, field2 in enumerate(field_names):
            # Skip self-comparisons
            if i == j:
                adjusted_df.loc[field1, field2] = 1.0
                continue
            
            group1 = field_to_group[field1]
            group2 = field_to_group[field2]
            subgroup1 = field_to_subgroup[field1]
            subgroup2 = field_to_subgroup[field2]
            
            raw_similarity = similarity_df.loc[field1, field2]
            
            if group1 == group2:
                if subgroup1 == subgroup2:
                    # Same subgroup
                    adjusted_similarity = SAME_SUBGROUP_BASELINE + (raw_similarity * SIMILARITY_WEIGHT_SUB)
                else:
                    # Same group, different subgroup
                    adjusted_similarity = SAME_GROUP_BASELINE + (raw_similarity * SIMILARITY_WEIGHT_GENERAL)
                adjusted_df.loc[field1, field2] = adjusted_similarity
            else:
                # Different groups - mark for scaling later
                cross_group_df.loc[field1, field2] = raw_similarity
                cross_group_mask[i, j] = True
    
    # Now scale the cross-group similarities to the range [0, MAX_CROSS_GROUP_SIMILARITY]
    if cross_group_mask.any():
        cross_group_values = cross_group_df.values[cross_group_mask]
        scaler = MinMaxScaler(feature_range=(0, MAX_CROSS_GROUP_SIMILARITY))
        
        # Reshape for scaling
        scaled_values = scaler.fit_transform(cross_group_values.reshape(-1, 1)).flatten()
        
        # Put scaled values back
        cross_group_df.values[cross_group_mask] = scaled_values
        
        # Merge with adjusted_df
        for i, field1 in enumerate(field_names):
            for j, field2 in enumerate(field_names):
                if cross_group_mask[i, j]:
                    adjusted_df.loc[field1, field2] = cross_group_df.loc[field1, field2]
    
    return adjusted_df

def find_top_similar_fields(similarity_df: pd.DataFrame, n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """Find the top N most similar fields for each field."""
    result = {}
    
    for field in similarity_df.index:
        # Get similarities, sort, and drop the field itself (which would have similarity 1.0)
        similarities = similarity_df.loc[field].drop(field).sort_values(ascending=False)
        top_n = similarities.head(n)
        
        result[field] = [(name, score) for name, score in top_n.items()]
    
    return result

def generate_heatmap(similarity_df: pd.DataFrame, field_to_group: Dict[str, str], filename: str = "heatmap.png"):
    """Generate a heatmap visualization of the similarity matrix."""
    plt.figure(figsize=(20, 16))
    
    # Sort the DataFrame by group and subgroup for better visualization
    field_names = similarity_df.index.tolist()
    simplified_group_names = {field: group.split(' & ')[0] for field, group in field_to_group.items()}
    
    # Sort by group
    sorted_fields = sorted(field_names, key=lambda x: simplified_group_names.get(x, ''))
    sorted_df = similarity_df.loc[sorted_fields, sorted_fields]
    
    # Create heatmap
    sns.heatmap(sorted_df, cmap="viridis", vmin=0, vmax=1, annot=False)
    
    plt.title("Field Similarity Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {filename}")

def create_similarity_json(similarity_df: pd.DataFrame) -> List[Dict]:
    """Convert similarity DataFrame to a list of field pairs with similarity scores."""
    result = []
    field_names = similarity_df.index.tolist()
    
    # Only include each pair once (avoid duplicates like A-B and B-A)
    for i, field1 in enumerate(field_names):
        for j, field2 in enumerate(field_names[i+1:], i+1):
            similarity = similarity_df.loc[field1, field2]
            result.append({
                "field1": field1,
                "field2": field2,
                "similarity_score": float(similarity)  # Convert numpy float to Python float for JSON serialization
            })
    
    return result

def create_similarity_table(similarity_df: pd.DataFrame) -> pd.DataFrame:
    """Convert similarity DataFrame to a tabular format with field1, field2, similarity_score columns."""
    rows = []
    field_names = similarity_df.index.tolist()
    
    # Only include each pair once (avoid duplicates like A-B and B-A)
    for i, field1 in enumerate(field_names):
        for j, field2 in enumerate(field_names[i+1:], i+1):
            similarity = similarity_df.loc[field1, field2]
            rows.append({
                "field1": field1,
                "field2": field2,
                "similarity_score": float(similarity)  # Convert numpy float to Python float
            })
    
    return pd.DataFrame(rows)

def print_weights_configuration():
    """Print the current weights configuration for description properties."""
    print("\nDescription property weights configuration:")
    for prop, weight in DESCRIPTION_WEIGHTS.items():
        print(f"  - {prop}: {weight:.2f}")
    print()

#############################################################
#                      MAIN EXECUTION                       #
#############################################################

def main():
    # Print weights configuration
    print_weights_configuration()
    
    # Check if the output directory exists, if not create it
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load the JSON data
    print(f"Loading data from {JSON_FILE_PATH}...")
    data = load_json_data(JSON_FILE_PATH)
    if not data:
        return
    
    # Step 2: Extract fields and their relationships
    print("Extracting field information...")
    fields, field_to_group, field_to_subgroup = extract_fields_info(data)
    print(f"Found {len(fields)} fields across {len(set(field_to_group.values()))} groups and {len(set(field_to_subgroup.values()))} subgroups.")
    
    # Step 3: Calculate semantic similarities with weighted properties
    print(f"Calculating semantic similarities using {MODEL_NAME} with weighted property similarities...")
    similarity_df = calculate_semantic_similarities(fields)
    
    # Step 4: Adjust similarities based on group/subgroup relationships
    print("Adjusting similarities based on group and subgroup relationships...")
    adjusted_similarity_df = adjust_similarities(similarity_df, field_to_group, field_to_subgroup)
    
    print(f"Saving similarity data to {OUTPUT_CSV}...")
    similarity_table = create_similarity_table(adjusted_similarity_df)
    similarity_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(similarity_table)} field pairs to CSV file.")

    print(f"Saving similarity data to {OUTPUT_JSON}...")
    similarity_pairs = create_similarity_json(adjusted_similarity_df)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as json_file:
        json.dump(similarity_pairs, json_file, indent=2)
    print(f"Exported {len(similarity_pairs)} field pairs to JSON file.")
    
    # Step 6: Generate heatmap if requested
    if GENERATE_HEATMAP:
        print("Generating heatmap visualization...")
        generate_heatmap(adjusted_similarity_df, field_to_group, HEATMAP_FILENAME)
    
    # Step 7: Find and print top similar fields
    print(f"\nTop {TOP_N_SIMILAR} most similar fields for each field:")
    top_similar = find_top_similar_fields(adjusted_similarity_df, TOP_N_SIMILAR)
    
    for field, similar_fields in top_similar.items():
        print(f"\n{field}:")
        for similar_field, score in similar_fields:
            print(f"  - {similar_field}: {score:.4f}")

if __name__ == "__main__":
    main()