"""
Utility functions for text processing and similarity calculation.
"""

import numpy as np
from typing import List, Dict, Set, Any

class TextProcessor:
    """Utility class for text processing operations."""
    
    @staticmethod
    def scale_similarity(raw_similarity: float) -> float:
        """
        Scale similarity with sigmoid-like function.
        
        Args:
            raw_similarity: Raw similarity score in [0,1] range
            
        Returns:
            Scaled similarity score in [0,1] range
        """
        # Parameters for scaling
        midpoint = 0.45
        steepness = 7.0
        
        # Handle extreme values
        if raw_similarity >= 0.95: return 1.0
        if raw_similarity <= 0.05: return 0.0
        
        # Apply sigmoid transformation
        scaled = 1.0 / (1.0 + np.exp(-steepness * (raw_similarity - midpoint)))
        
        # Normalize to [0,1] range
        min_val = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
        max_val = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        scaled_normalized = (scaled - min_val) / (max_val - min_val)
        
        return scaled_normalized
    
    @staticmethod
    def calibrate_final_score(score: float) -> float:
        """
        Apply sigmoid calibration to score.
        
        Args:
            score: Input score in [0,1] range
            
        Returns:
            Calibrated score in [0,1] range
        """
        # Parameters for calibration
        midpoint = 0.40
        steepness = 8.0
        
        # Special cases
        if score >= 0.95: return 1.0
        if score <= 0.05: return 0.0
            
        # Apply sigmoid function and normalize
        calibrated = 1.0 / (1.0 + np.exp(-steepness * (score - midpoint)))
        min_val = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
        max_val = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        
        calibrated_normalized = (calibrated - min_val) / (max_val - min_val)
        
        return calibrated_normalized