"""
Service for calculating similarity between research fields.
"""

import logging
from typing import Dict, List, Any, Optional

from config import (
    COMPONENT_WEIGHTS, DESCRIPTION_WEIGHTS, 
    SAME_GROUP_BASELINE, SAME_SUBGROUP_BASELINE,
    SIMILARITY_WEIGHT_SUB, SIMILARITY_WEIGHT_GENERAL, 
    MAX_CROSS_GROUP_SIMILARITY, DOMAIN_BOOST_THRESHOLD,
    MAX_BOOST_FACTOR, ENABLE_DOMAIN_BOOSTING,DOMAIN_GROUP_SIMILARITY
)
from utils.text_processing import TextProcessor
from services.similarity.embedding import EmbeddingService
from services.similarity.tfidf import TfidfService
from services.similarity.domain import DomainService

logger = logging.getLogger(__name__)

class FieldSimilarityService:
    """Service for calculating similarity between research fields."""
    
    def __init__(self):
        """Initialize the field similarity service."""
        self.embedding_service = EmbeddingService()
        self.tfidf_service = TfidfService()
        self.domain_service = DomainService()
        
        # Mappings for efficient group lookup
        self.field_to_group = {}
        self.field_to_subgroup = {}
    
    def update_field_mappings(self, nested_data: Dict[str, Any]):
        """
        Update field to group/subgroup mappings from nested data.
        
        Args:
            nested_data: The hierarchical research field data
        """
        # Reset mappings
        self.field_to_group = {}
        self.field_to_subgroup = {}
        
        # Build mappings from nested data
        for category in nested_data.get("categories", []):
            group_name = category["name"]
            
            for subgroup in category.get("subgroups", []):
                subgroup_name = subgroup["name"]
                
                for field in subgroup.get("fields", []):
                    field_name = field["name"]
                    self.field_to_group[field_name] = group_name
                    self.field_to_subgroup[field_name] = subgroup_name
                    
        logger.info(f"Updated mappings for {len(self.field_to_group)} fields")
    
    def calculate_faceted_similarity(self, field1: Dict[str, Any], field2: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate similarities for each facet of field descriptions.
        
        Args:
            field1: First field data
            field2: Second field data
            
        Returns:
            Dictionary mapping facets to similarity scores
        """
        facet_similarities = {}
        
        # Get description dictionaries
        desc1 = field1.get('description', {})
        desc2 = field2.get('description', {})
        
        # If descriptions are strings instead of dictionaries, convert them
        if isinstance(desc1, str):
            desc1 = {'definition': desc1}
        if isinstance(desc2, str):
            desc2 = {'definition': desc2}
        
        # Calculate similarity for each facet
        for facet, weight in DESCRIPTION_WEIGHTS.items():
            text1 = desc1.get(facet, '')
            text2 = desc2.get(facet, '')
            
            if not text1 or not text2:
                facet_similarities[facet] = 0.0
                continue
            
            # Calculate all component similarities
            embedding_sim = self.embedding_service.calculate_similarity(text1, text2)
            tfidf_sim = self.tfidf_service.calculate_similarity(text1, text2)
            domain_sim = self.domain_service.calculate_similarity(text1, text2)
            
            # Weight the components for this facet
            facet_sim = (
                COMPONENT_WEIGHTS['embedding'] * embedding_sim +
                COMPONENT_WEIGHTS['tfidf'] * tfidf_sim +
                COMPONENT_WEIGHTS['domain'] * domain_sim
            )
            
            facet_similarities[facet] = facet_sim
        
        return facet_similarities
    
    def _get_full_text(self, field: Dict[str, Any]) -> str:
        """
        Extract all text from a field description.
        
        Args:
            field: Field data dictionary
            
        Returns:
            Concatenated text from all description fields
        """
        desc = field.get('description', {})
        
        if isinstance(desc, str):
            return desc
        
        return ' '.join(str(text) for text in desc.values() if text)
    
    def compare_fields(self, field1: Dict[str, Any], field2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two research fields.
        
        Args:
            field1: First field data
            field2: Second field data
            
        Returns:
            Similarity score in [0,1] range
        """
        # Handle identity comparison
        if field1['name'] == field2['name']:
            return 1.0
        
        # Get faceted similarities
        facet_similarities = self.calculate_faceted_similarity(field1, field2)
        
        # Calculate weighted facet score
        weighted_facet_sim = 0.0
        total_weight = 0.0
        
        for facet, sim in facet_similarities.items():
            weight = DESCRIPTION_WEIGHTS.get(facet, 0.0)
            weighted_facet_sim += sim * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_facet_sim /= total_weight
        
        # Get full text from both fields
        full_text1 = self._get_full_text(field1)
        full_text2 = self._get_full_text(field2)
        
        # Calculate component similarities
        embedding_sim = self.embedding_service.calculate_similarity(full_text1, full_text2)
        tfidf_sim = self.tfidf_service.calculate_similarity(full_text1, full_text2)
        domain_sim = self.domain_service.calculate_similarity(full_text1, full_text2)
        
        # Calculate overall similarity
        overall_similarity = (
            COMPONENT_WEIGHTS['embedding'] * embedding_sim +
            COMPONENT_WEIGHTS['tfidf'] * tfidf_sim +
            COMPONENT_WEIGHTS['domain'] * domain_sim +
            COMPONENT_WEIGHTS['facet'] * weighted_facet_sim
        )
        
        # Apply domain boosting if enabled
        boosted_similarity = overall_similarity
        
        if ENABLE_DOMAIN_BOOSTING:
            field1_domains = self.domain_service.detect_primary_domains(full_text1)
            field2_domains = self.domain_service.detect_primary_domains(full_text2)
            
            # Calculate max domain relatedness
            max_domain_sim = 0.0
            for d1 in field1_domains:
                for d2 in field2_domains:
                    domain_sim = DOMAIN_GROUP_SIMILARITY.get(d1, {}).get(d2, 0.0)
                    max_domain_sim = max(max_domain_sim, domain_sim)
            
            # Apply boosting for highly related domains
            if max_domain_sim > DOMAIN_BOOST_THRESHOLD:
                boost_factor = MAX_BOOST_FACTOR * (max_domain_sim - DOMAIN_BOOST_THRESHOLD) / (1.0 - DOMAIN_BOOST_THRESHOLD)
                boosted_similarity = min(1.0, overall_similarity + boost_factor)
        
        # Apply sigmoid calibration
        calibrated_similarity = TextProcessor.calibrate_final_score(boosted_similarity)
        
        # FINAL STEP: Apply group-based adjustments
        field1_name = field1['name']
        field2_name = field2['name']
        
        # Determine group and subgroup relationships
        final_similarity = 0.0
        
        # Check if fields are in the same subgroup
        if (field1_name in self.field_to_subgroup and 
            field2_name in self.field_to_subgroup and 
            self.field_to_subgroup[field1_name] == self.field_to_subgroup[field2_name]):
            # Fields in same subgroup: baseline + weighted similarity
            final_similarity = SAME_SUBGROUP_BASELINE + (calibrated_similarity * SIMILARITY_WEIGHT_SUB)
            
        # Check if fields are in the same general group
        elif (field1_name in self.field_to_group and 
              field2_name in self.field_to_group and 
              self.field_to_group[field1_name] == self.field_to_group[field2_name]):
            # Fields in same group: baseline + weighted similarity
            final_similarity = SAME_GROUP_BASELINE + (calibrated_similarity * SIMILARITY_WEIGHT_GENERAL)
            
        # Fields are in different groups
        else:
            # Linearly scale the similarity to the range [0, MAX_CROSS_GROUP_SIMILARITY]
            final_similarity = calibrated_similarity * MAX_CROSS_GROUP_SIMILARITY
        
        # Ensure similarity is in valid range [0, 1]
        final_similarity = max(0.0, min(1.0, final_similarity))
        
        return final_similarity
    
    def calculate_new_similarities(self, nested_data: Dict[str, Any], 
                                   similarities: List[Dict[str, Any]], 
                                   new_field: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate similarities between new field and existing fields.
        
        Args:
            nested_data: The hierarchical research field data
            similarities: List of existing similarity records
            new_field: Newly added field data
            
        Returns:
            Updated list of similarity records
        """
        # Extract all existing fields
        all_fields = []
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for field in subgroup.get("fields", []):
                    if field["name"] != new_field["name"]:  # Skip the new field itself
                        all_fields.append(field)
        
        # Calculate similarities between new field and all existing fields
        new_similarities = []
        for existing_field in all_fields:
            # Check if this pair already exists in similarities
            pair_exists = False
            for sim in similarities:
                if ((sim.get("field1") == new_field["name"] and sim.get("field2") == existing_field["name"]) or
                    (sim.get("field1") == existing_field["name"] and sim.get("field2") == new_field["name"])):
                    pair_exists = True
                    break
            
            # Skip if pair already exists
            if pair_exists:
                continue
            
            # Calculate similarity
            similarity = self.compare_fields(new_field, existing_field)
            
            # Add to new similarities
            new_similarities.append({
                "field1": new_field["name"],
                "field2": existing_field["name"],
                "similarity_score": float(similarity)  # Convert numpy float to Python float
            })
        
        # Combine existing and new similarities
        updated_similarities = similarities + new_similarities
        logger.info(f"Calculated {len(new_similarities)} new similarity scores for {new_field['name']}")
        
        return updated_similarities
    
    
    
    # def calculate_all_similarities(self, nested_data: Dict[str, Any]):
    #     """
    #     Calculate all pairwise similarities between fields in the nested data.
        
    #     Returns:
    #         List of all similarity records
    #     """
    #     # Load nested data and existing similarities
    #     nested_data, existing_similarities = self.load_data()
        
    #     # Update field mappings for efficient group lookup
    #     self.update_field_mappings(nested_data)
        
    #     # Extract all fields from nested data
    #     all_fields = []
    #     for category in nested_data.get("categories", []):
    #         for subgroup in category.get("subgroups", []):
    #             for field in subgroup.get("fields", []):
    #                 all_fields.append(field)
        
    #     # Calculate pairwise similarities
    #     all_similarities = []
    #     for i, field1 in enumerate(all_fields):
    #         for j in range(i + 1, len(all_fields)):
    #             field2 = all_fields[j]
                
    #             # Skip if pair already exists in existing similarities
    #             pair_exists = False
    #             for sim in existing_similarities:
    #                 if ((sim.get("field1") == field1["name"] and sim.get("field2") == field2["name"]) or
    #                     (sim.get("field1") == field2["name"] and sim.get("field2") == field1["name"])):
    #                     pair_exists = True
    #                     break
                
    #             if pair_exists:
    #                 continue