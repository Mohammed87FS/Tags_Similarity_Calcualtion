"""
Utility functions for text processing and similarity calculation.
"""

import numpy as np
from typing import List, Dict, Set, Any

class TextProcessor:
    """Utility class for text processing operations."""
    
    
    @staticmethod
    def calibrate_similarity_score(score: float) -> float:
      
  
        midpoint = 0.50 # Midpoint of the sigmoid function,
                        #where the curve is steepest to help with calibration by pushing the values towards 0 or 1

        steepness = 7.0 # Higher values make the curve steeper and it helps make the calibration more aggressive
        
     
        if score <= 0.05: return 0.0
            
        # Apply sigmoid function and normalize
        calibrated = 1.0 / (1.0 + np.exp(-steepness * (score - midpoint)))
        min_val = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
        max_val = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        
        calibrated_normalized = (calibrated - min_val) / (max_val - min_val)
        
        return calibrated_normalized