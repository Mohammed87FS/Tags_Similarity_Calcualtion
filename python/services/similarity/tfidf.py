"""
Service for TF-IDF-based similarity calculations.
"""

import re
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class TfidfService:
    """Service for TF-IDF operations."""
    
    def __init__(self):
        """Initialize the TF-IDF service."""
        self.tfidf_similarity_cache = {}
        
        # Technical patterns for term extraction
        self.technical_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)+\b',   # CamelCase terms
            r'\b[a-z]+(?:-[a-z]+)+\b',                    # hyphenated terms
            r'\b[A-Za-z]+\d+[A-Za-z]*\b',                # terms with numbers
            r'\b[A-Za-z]+\.[A-Za-z]+\b',                 # software libraries
            r'\b[A-Z][A-Z0-9]+\b',                       # acronyms
        ]
        
        # Load spaCy for NLP tasks if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            print("Warning: spaCy model not found. Using basic text processing.")
            self.use_spacy = False
            self.nlp = None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0,1] range
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Check cache for this pair
        cache_key = f"{hash(text1)}_{hash(text2)}"
        if cache_key in self.tfidf_similarity_cache:
            return self.tfidf_similarity_cache[cache_key]
            
        # Pre-process to emphasize technical terms
        def preprocess(text):
            # Extract potential technical terms using regex patterns
            technical_terms = []
            for pattern in self.technical_patterns:
                terms = re.findall(pattern, text)
                technical_terms.extend(terms)
            
            # Add domain-specific terms found in the text
            if self.use_spacy and self.nlp:
                doc = self.nlp(text.lower())
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:  # Multi-word terms
                        technical_terms.append(chunk.text)
            
            # Add these terms back to the text with repetition to increase weight
            enhanced = text + " "
            if technical_terms:
                enhanced += " ".join(technical_terms) + " " + " ".join(technical_terms)
            
            return enhanced
        
        # Create corpus with enhanced texts
        corpus = [preprocess(text1), preprocess(text2)]
        
        # Use TF-IDF vectorizer with ngrams
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            stop_words='english',
            max_features=10000
        )
        
        # Calculate TF-IDF matrix
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            dot_product = tfidf_matrix[0].dot(tfidf_matrix[1].T).toarray()[0][0]
            norm1 = np.sqrt(tfidf_matrix[0].dot(tfidf_matrix[0].T).toarray()[0][0])
            norm2 = np.sqrt(tfidf_matrix[1].dot(tfidf_matrix[1].T).toarray()[0][0])
            
            if norm1 * norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Cache the result
            self.tfidf_similarity_cache[cache_key] = similarity
            
            return similarity
        except Exception as e:
            # If vectorization fails (e.g., with very short texts)
            print(f"TF-IDF vectorization failed: {e}")
            return 0.0