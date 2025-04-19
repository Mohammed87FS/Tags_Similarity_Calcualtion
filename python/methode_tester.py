from sentence_transformers import SentenceTransformer
import numpy as np

def calculate_text_similarity(text1: str, text2: str, model_name: str = 'all-mpnet-base-v2') -> float:
    """
    Calculate semantic similarity between two text strings using sentence transformers.
    
    Args:
        text1: First text string
        text2: Second text string
        model_name: Name of the sentence transformer model to use
                   Default is 'all-mpnet-base-v2' (same as in original code)
                   Other options: 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L3-v2'
    
    Returns:
        Similarity score between 0 and 1, where 1 means identical meaning
    """
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings for both texts
    embedding1 = model.encode(text1, show_progress_bar=False)
    embedding2 = model.encode(text2, show_progress_bar=False)
    
    # Calculate cosine similarity between the embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    return similarity

# Example usage
if __name__ == "__main__":
    text1 = "respect the father"
    text2 = "hate the father"
    
    similarity = calculate_text_similarity(text1, text2)
    print(f"Similarity score: {similarity:.4f}")