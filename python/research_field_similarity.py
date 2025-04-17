"""
Enhanced Research Field Similarity Analyzer

This script analyzes the semantic similarity between research fields based on 
their detailed descriptions using state-of-the-art language models and network-based
enhancement techniques for more coherent similarity scores with contrast management.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sentence_transformers import SentenceTransformer
import networkx as nx
import pandas as pd 
import json

def load_research_fields(json_file_path="../research_fields.json"):
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
    return SentenceTransformer('all-mpnet-base-v2')

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

def enhance_similarity_coherence(similarity_matrix, alpha=0.3):
    """
    Enhance the coherence of similarity scores by accounting for transitive relationships.
    
    Parameters:
    - similarity_matrix: Original pairwise similarity matrix
    - alpha: Weight for the transitive influence (0-1)
    
    Returns:
    - Enhanced similarity matrix
    """
    print("Enhancing similarity coherence based on transitive relationships...")
    
    # Create a copy of the original matrix
    enhanced_sim = similarity_matrix.copy()
    
    # Get the number of tags
    n = similarity_matrix.shape[0]
    
    # For each pair of tags
    for i in range(n):
        for j in range(i+1, n):  # Only process each pair once
            # Skip if direct similarity is already very high
            if similarity_matrix[i, j] > 0.9:
                continue
                
            # Calculate the transitive similarity through all other tags
            transitive_similarities = []
            for k in range(n):
                if k != i and k != j:
                    # The transitive similarity through tag k
                    trans_sim = min(similarity_matrix[i, k], similarity_matrix[j, k])
                    transitive_similarities.append(trans_sim)
            
            # If we found transitive paths
            if transitive_similarities:
                # Use the strongest transitive connection
                best_transitive = max(transitive_similarities)
                
                # Adjust the similarity based on the transitive relationship
                # If the direct similarity is lower than what the network suggests
                if best_transitive > similarity_matrix[i, j]:
                    enhanced_sim[i, j] = (1-alpha) * similarity_matrix[i, j] + alpha * best_transitive
                    enhanced_sim[j, i] = enhanced_sim[i, j]  # Ensure symmetry
    
    return enhanced_sim

def network_reinforcement(similarity_matrix, alpha=0.3, iterations=1):
    """
    Reinforce similarities based on network structure using all common neighbors.
    
    Parameters:
    - similarity_matrix: Original similarity matrix
    - alpha: Weight for indirect similarities (0-1)
    - iterations: Number of reinforcement iterations
    
    Returns:
    - Reinforced similarity matrix
    """
    print("Applying network reinforcement with all common neighbors...")
    
    n = len(similarity_matrix)
    reinforced_matrix = similarity_matrix.copy()
    
    for _ in range(iterations):
        new_matrix = reinforced_matrix.copy()
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate indirect similarity through common neighbors
                    indirect_sim = 0
                    for k in range(n):
                        if k != i and k != j:
                            indirect_sim += reinforced_matrix[i, k] * reinforced_matrix[k, j]
                    
                    # Normalize by the number of possible intermediate nodes
                    indirect_sim /= max(1, n - 2)
                    
                    # Combine direct and indirect similarity
                    new_matrix[i, j] = (1-alpha) * reinforced_matrix[i, j] + alpha * indirect_sim
        
        reinforced_matrix = new_matrix
    
    return reinforced_matrix

def apply_graph_diffusion(similarity_matrix, alpha=0.2, iterations=2):
    """
    Apply graph diffusion to refine similarities based on the network structure.
    
    Parameters:
    - similarity_matrix: Original similarity matrix
    - alpha: Damping factor (0-1)
    - iterations: Number of diffusion iterations
    
    Returns:
    - Refined similarity matrix
    """
    print("Applying graph diffusion to propagate similarity through the network...")
    
    # Start with the original matrix
    refined_sim = similarity_matrix.copy()
    
    # Apply multiple iterations of diffusion
    for i in range(iterations):
        # Matrix multiplication captures second-order connections
        second_order = np.matmul(refined_sim, refined_sim)
        
        # Combine direct and second-order connections
        refined_sim = (1-alpha) * similarity_matrix + alpha * second_order
        
        # Normalize to keep values in a reasonable range
        refined_sim = refined_sim / np.max(refined_sim)
    
    return refined_sim

def penalize_contradictions(similarity_matrix, penalty_strength=0.2):
    """
    Penalize topic pairs that have contradictory relationship patterns.
    
    Parameters:
    - similarity_matrix: Similarity matrix to be adjusted
    - penalty_strength: Strength of the penalty to apply (0-1)
    
    Returns:
    - Adjusted similarity matrix with contradictions penalized
    """
    print("Penalizing contradictory relationship patterns...")
    
    n = len(similarity_matrix)
    result = similarity_matrix.copy()
    total_penalties = 0
    max_penalty = 0
    
    for i in range(n):
        for j in range(i+1, n):  # Only process each pair once
            # Skip if already very dissimilar
            if similarity_matrix[i, j] < 0.2:
                continue
                
            contradiction_score = 0
            common_neighbors = 0
            
            # Check for contradictory patterns with common neighbors
            for k in range(n):
                if k != i and k != j:
                    # If connections to k are very different, this is a contradiction
                    difference = abs(similarity_matrix[i, k] - similarity_matrix[j, k])
                    if difference > 0.4:  # Significant difference threshold
                        contradiction_score += difference
                        common_neighbors += 1
            
            if common_neighbors > 0:
                avg_contradiction = contradiction_score / common_neighbors
                
                # Apply penalty proportional to the contradiction
                penalty = penalty_strength * avg_contradiction
                result[i, j] = max(0.0, similarity_matrix[i, j] - penalty)
                result[j, i] = result[i, j]  # Maintain symmetry
                
                total_penalties += penalty
                max_penalty = max(max_penalty, penalty)
    
    print(f"Applied contradictory penalties: {total_penalties:.4f} total, {max_penalty:.4f} max")
    return result

def enhance_contrast(similarity_matrix, power=1.5):
    """
    Apply a power transformation to enhance contrast in similarity scores.
    
    Parameters:
    - similarity_matrix: Similarity matrix to enhance
    - power: Power for the transformation (>1 increases contrast)
    
    Returns:
    - Contrast-enhanced similarity matrix
    """
    print(f"Enhancing contrast with power transformation (power={power})...")
    
    # Copy the matrix to avoid modifying the original
    result = similarity_matrix.copy()
    
    # Get min and max values for scaling (excluding diagonal)
    mask = ~np.eye(len(similarity_matrix), dtype=bool)
    values = similarity_matrix[mask]
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Skip if all values are identical
    if max_val == min_val:
        print("All similarity values are identical, skipping contrast enhancement")
        return result
    
    # Calculate original variance
    original_variance = np.var(values)
    
    # Apply power transformation to non-diagonal elements
    for i in range(len(result)):
        for j in range(len(result)):
            if i != j:  # Preserve diagonal elements (self-similarity)
                # Normalize to [0,1]
                normalized = (result[i, j] - min_val) / (max_val - min_val)
                
                # Apply power transformation
                stretched = normalized ** power
                
                # Scale back to original range
                result[i, j] = stretched * (max_val - min_val) + min_val
    
    # Calculate new variance
    new_values = result[mask]
    new_variance = np.var(new_values)
    print(f"Variance increased from {original_variance:.6f} to {new_variance:.6f}")
    
    return result

def balanced_similarity_enhancement(similarity_matrix, config=None):
    """
    Apply a balanced enhancement pipeline with both reinforcement and contrast.
    
    Parameters:
    - similarity_matrix: Original similarity matrix
    - config: Configuration dictionary with parameters
    
    Returns:
    - Enhanced similarity matrix
    """
    if config is None:
        config = {
            'reinforcement_alpha': 0.3,
            'reinforcement_iterations': 1,
            'penalty_strength': 0.2,
            'contrast_power': 1.5
        }
    
    print("\nApplying balanced similarity enhancement pipeline...")
    
    # Step 1: Network reinforcement to enhance connections with common neighbors
    reinforced = network_reinforcement(
        similarity_matrix, 
        alpha=config['reinforcement_alpha'],
        iterations=config['reinforcement_iterations']
    )
    
    # Step 2: Penalize contradictory relationships
    penalized = penalize_contradictions(
        reinforced,
        penalty_strength=config['penalty_strength']
    )
    
    # Step 3: Enhance contrast to maintain sufficient variance
    enhanced = enhance_contrast(
        penalized,
        power=config['contrast_power']
    )
    
    print("Balanced enhancement pipeline complete!")
    return enhanced

def visualize_similarity_matrix(field_names, similarity_matrix, title="Research Field Similarities"):
    """
    Visualize the similarity matrix as a heatmap.
    
    Parameters:
    - field_names: List of research field names
    - similarity_matrix: Matrix of similarity scores
    - title: Title for the plot
    """
    plt.figure(figsize=(16, 14))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
    
    # Create the heatmap
    sns.heatmap(similarity_matrix, mask=mask, cmap="viridis",
                xticklabels=field_names, yticklabels=field_names,
                vmin=0, vmax=1, annot=False, square=True, linewidths=.5)
    
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    # Save the figure
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Similarity matrix visualization saved as '{filename}'")

def create_similarity_network(field_names, similarity_matrix, threshold=0.65):
    """
    Create a network visualization of field similarities.
    
    Parameters:
    - field_names: List of research field names
    - similarity_matrix: Matrix of similarity scores
    - threshold: Minimum similarity to include as an edge
    """
    print(f"Creating similarity network visualization with threshold {threshold}...")
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes
    for name in field_names:
        G.add_node(name)
    
    # Add edges for similarities above threshold
    n = len(field_names)
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(field_names[i], field_names[j], 
                           weight=similarity_matrix[i, j])
    
    # Set up the plot
    plt.figure(figsize=(16, 14))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Research Field Similarity Network", fontsize=16)
    plt.axis('off')
    
    # Save the figure
    plt.savefig("research_field_network.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Network visualization saved as 'research_field_network.png'")
    return G

def compare_similarity_matrices(field_names, original_sim, enhanced_sim, top_n=15):
    """
    Compare original and enhanced similarity matrices.
    
    Parameters:
    - field_names: List of research field names
    - original_sim: Original similarity matrix
    - enhanced_sim: Enhanced similarity matrix
    - top_n: Number of top pairs to show
    """
    print("\nComparison of Original vs. Enhanced Similarities:")
    print("=" * 60)
    
    # Calculate variance of both matrices for comparison
    orig_variance = np.var(original_sim[~np.eye(len(original_sim), dtype=bool)])
    enhanced_variance = np.var(enhanced_sim[~np.eye(len(enhanced_sim), dtype=bool)])
    
    print(f"Original similarity variance: {orig_variance:.6f}")
    print(f"Enhanced similarity variance: {enhanced_variance:.6f}")
    print(f"Variance ratio: {enhanced_variance/orig_variance:.2f}x")
    
    # Find pairs with the largest changes
    changes = []
    n = len(field_names)
    for i in range(n):
        for j in range(i+1, n):
            change = enhanced_sim[i, j] - original_sim[i, j]
            changes.append((field_names[i], field_names[j], 
                           original_sim[i, j], enhanced_sim[i, j], change))
    
    # Sort by absolute change (largest first)
    changes.sort(key=lambda x: abs(x[4]), reverse=True)
    
    # Show pairs with largest positive changes
    print("\nLargest Positive Changes (Reinforced Relationships):")
    positive_changes = [c for c in changes if c[4] > 0]
    for i, (field1, field2, orig, enhanced, change) in enumerate(positive_changes[:top_n]):
        change_pct = (change / orig) * 100 if orig > 0 else float('inf')
        print(f"{i+1}. {field1} ↔ {field2}: {orig:.4f} → {enhanced:.4f} " + 
              f"(Δ {change:+.4f}, {change_pct:+.1f}%)")
    
    # Show pairs with largest negative changes
    print("\nLargest Negative Changes (Penalized Relationships):")
    negative_changes = [c for c in changes if c[4] < 0]
    for i, (field1, field2, orig, enhanced, change) in enumerate(negative_changes[:top_n]):
        change_pct = (change / orig) * 100 if orig > 0 else float('inf')
        print(f"{i+1}. {field1} ↔ {field2}: {orig:.4f} → {enhanced:.4f} " + 
              f"(Δ {change:+.4f}, {change_pct:+.1f}%)")
    
    # Export the comparison to Excel for further analysis
    comparison_df = pd.DataFrame(changes, columns=[
        "Field 1", "Field 2", "Original Similarity", 
        "Enhanced Similarity", "Absolute Change"
    ])
    comparison_df['Percentage Change'] = comparison_df.apply(
        lambda row: (row['Absolute Change'] / row['Original Similarity'] * 100) 
        if row['Original Similarity'] > 0 else float('inf'), axis=1
    )
    comparison_df.to_excel("similarity_comparison.xlsx", index=False)
    print("Comparison exported to 'similarity_comparison.xlsx'")
    
    return changes

def display_similarity_results(field_names, similarity_matrix):
    """Display the most significant similarity pairs."""
    print("\nMost Significant Relationships Between Research Fields:")
    print("=" * 60)
    
    # Create a list of pairs with their similarity scores
    pairs = []
    for i in range(len(field_names)):
        for j in range(i+1, len(field_names)):  
            sim = similarity_matrix[i][j]
            pairs.append((field_names[i], field_names[j], sim))
    
    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Display top pairs
    for i, (field1, field2, score) in enumerate(pairs[:20]):
        print(f"{i+1}. {field1} ↔ {field2}: {score:.4f}")
    
    return pairs

def calculate_similarity_stats(similarity_matrix):
    """
    Calculate statistics about the similarity matrix to assess contrast.
    
    Parameters:
    - similarity_matrix: Similarity matrix
    
    Returns:
    - Dictionary with statistics
    """
    # Exclude diagonal elements (self-similarity)
    mask = ~np.eye(len(similarity_matrix), dtype=bool)
    values = similarity_matrix[mask]
    
    # Calculate distribution statistics
    stats = {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'variance': np.var(values),
        # Calculate percentiles
        'p10': np.percentile(values, 10),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p90': np.percentile(values, 90),
        # Calculate ratio between high and low similarities
        'contrast_ratio_90_10': np.percentile(values, 90) / max(0.001, np.percentile(values, 10))
    }
    
    return stats

def print_similarity_stats(stats, title="Similarity Statistics"):
    """Print statistics about similarity distribution."""
    print(f"\n{title}:")
    print("=" * 40)
    print(f"Min: {stats['min']:.4f}  Max: {stats['max']:.4f}  Range: {stats['max']-stats['min']:.4f}")
    print(f"Mean: {stats['mean']:.4f}  Median: {stats['median']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}  Variance: {stats['variance']:.6f}")
    print(f"10th percentile: {stats['p10']:.4f}  90th percentile: {stats['p90']:.4f}")
    print(f"90:10 percentile ratio: {stats['contrast_ratio_90_10']:.2f}x")
    print("=" * 40)

def export_to_json(similarity_pairs, output_path="json_sim/research_field_similarities.json"):
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
    """Main function to run the enhanced research field similarity analysis."""
    print("=" * 60)
    print("  ENHANCED RESEARCH FIELD SEMANTIC SIMILARITY ANALYZER")
    print("  WITH CONTRAST MANAGEMENT")
    print("=" * 60)
    
    # Load research fields from JSON file
    research_fields = load_research_fields()
    
    # Load model and prepare data
    model = load_model()
    field_names, field_descriptions = extract_field_data(research_fields)
    
    # Compute embeddings and initial similarity
    embeddings = compute_embeddings(field_descriptions, model)
    initial_similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Calculate and display statistics for the initial matrix
    initial_stats = calculate_similarity_stats(initial_similarity_matrix)
    print_similarity_stats(initial_stats, "Initial Similarity Statistics")
    
    # Save the original similarity matrix for comparison
    original_pairs = display_similarity_results(field_names, initial_similarity_matrix)
    export_to_json(original_pairs, "json_sim/original_research_field_similarities.json")
    
    # Visualize the original similarity matrix
    visualize_similarity_matrix(field_names, initial_similarity_matrix, "Original Field Similarities")
    
    # Apply the balanced enhancement pipeline
    config = {
        'reinforcement_alpha': 0.3,     # Weight for network reinforcement
        'reinforcement_iterations': 1,   # Number of reinforcement iterations
        'penalty_strength': 0.2,         # Strength of contradiction penalties
        'contrast_power': 1.5            # Power for contrast enhancement
    }
    
    balanced_similarity = balanced_similarity_enhancement(initial_similarity_matrix, config)
    
    # Calculate and display statistics for the balanced matrix
    balanced_stats = calculate_similarity_stats(balanced_similarity)
    print_similarity_stats(balanced_stats, "Balanced Similarity Statistics")
    
    # Visualize the balanced similarity matrix
    visualize_similarity_matrix(field_names, balanced_similarity, "Balanced Field Similarities")
    
    # For comparison, also apply the original enhancement methods
    print("\nApplying original enhancement methods for comparison...")
    coherent_similarity = enhance_similarity_coherence(initial_similarity_matrix, alpha=0.3)
    refined_similarity = apply_graph_diffusion(coherent_similarity, alpha=0.2, iterations=2)
    
    # Calculate and display statistics for the original enhancement
    refined_stats = calculate_similarity_stats(refined_similarity)
    print_similarity_stats(refined_stats, "Original Enhancement Statistics")
    
    # Visualize the original enhanced similarity matrix
    visualize_similarity_matrix(field_names, refined_similarity, "Original Enhanced Field Similarities")
    
    # Create network visualizations for both approaches
    create_similarity_network(field_names, balanced_similarity, threshold=0.65)
    create_similarity_network(field_names, refined_similarity, threshold=0.65)
    
    # Compare the matrices to see the difference
    print("\nComparing initial vs. balanced enhancement:")
    compare_similarity_matrices(field_names, initial_similarity_matrix, balanced_similarity)
    
    print("\nComparing initial vs. original enhancement:")
    compare_similarity_matrices(field_names, initial_similarity_matrix, refined_similarity)
    
    print("\nComparing original enhancement vs. balanced enhancement:")
    compare_similarity_matrices(field_names, refined_similarity, balanced_similarity)
    
    # Get similarity pairs and export using both enhancement methods
    balanced_pairs = display_similarity_results(field_names, balanced_similarity)
    export_to_json(balanced_pairs, "json_simbalanced_research_field_similarities.json")
    export_to_excel(balanced_pairs, "json_simbalanced_research_field_similarities.xlsx")
    
    refined_pairs = display_similarity_results(field_names, refined_similarity)
    export_to_json(refined_pairs, "json_simrefined_research_field_similarities.json")
    export_to_excel(refined_pairs, "json_simrefined_research_field_similarities.xlsx")
    
    print("\nEnhanced similarity analysis complete!")

if __name__ == "__main__":
    main()