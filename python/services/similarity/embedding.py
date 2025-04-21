"""
Service for embedding-based similarity calculations.
"""

import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer

from utils.text_processing import TextProcessor

class EmbeddingService:
    """Service for text embedding operations."""
    
    def __init__(self):
        """Initialize the embedding service with a pre-trained model."""
        # Load model
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.embedding_cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching for efficiency.
        
        Args:
            text: Input text to embed
            
        Returns:
            NumPy array containing the text embedding
        """
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
            
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.model.encode(text, show_progress_bar=False)
        return self.embedding_cache[text]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between text embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0,1] range
        """
        if not text1.strip() or not text2.strip():
            return 0.0
            
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Scale similarity
        return TextProcessor.scale_similarity(similarity)