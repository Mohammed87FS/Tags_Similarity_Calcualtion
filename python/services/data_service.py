"""
Service for loading and saving data to JSON files.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

from config import DATA_DIR, NESTED_DESCRIPTIONS_FILE, SIMILARITY_FILE

logger = logging.getLogger(__name__)

class DataService:
    """Service for loading and saving application data."""
    
    @staticmethod
    def load_data() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load data from JSON files.
        
        Returns:
            Tuple containing (nested_data, similarities)
        """
        # Initialize with empty structures if files don't exist
        nested_data = {"categories": []}
        similarities = []
        
        # Load nested descriptions if file exists
        if os.path.exists(NESTED_DESCRIPTIONS_FILE):
            try:
                with open(NESTED_DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
                    nested_data = json.load(f)
                logger.info(f"Successfully loaded nested data with {len(nested_data.get('categories', []))} categories")
            except Exception as e:
                logger.error(f"Error loading nested descriptions: {e}")
        else:
            logger.warning(f"Nested descriptions file not found at {NESTED_DESCRIPTIONS_FILE}")
        
        # Load similarities if file exists
        if os.path.exists(SIMILARITY_FILE):
            try:
                with open(SIMILARITY_FILE, 'r', encoding='utf-8') as f:
                    similarities = json.load(f)
                logger.info(f"Successfully loaded {len(similarities)} similarity records")
            except Exception as e:
                logger.error(f"Error loading similarities: {e}")
        else:
            logger.warning(f"Similarities file not found at {SIMILARITY_FILE}")
        
        return nested_data, similarities
    
    @staticmethod
    def save_data(nested_data: Dict[str, Any], similarities: List[Dict[str, Any]]) -> bool:
        """
        Save data to JSON files.
        
        Args:
            nested_data: The hierarchical research field data
            similarities: List of similarity records
            
        Returns:
            True if saving was successful, False otherwise
        """
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(NESTED_DESCRIPTIONS_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(SIMILARITY_FILE), exist_ok=True)
        
        # Save nested descriptions
        try:
            with open(NESTED_DESCRIPTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(nested_data, f, indent=2)
            logger.info(f"Successfully saved nested descriptions to {NESTED_DESCRIPTIONS_FILE}")
        except Exception as e:
            logger.error(f"Error saving nested descriptions: {e}")
            return False
        
        # Save similarities
        try:
            with open(SIMILARITY_FILE, 'w', encoding='utf-8') as f:
                json.dump(similarities, f, indent=2)
            logger.info(f"Successfully saved {len(similarities)} similarity records to {SIMILARITY_FILE}")
        except Exception as e:
            logger.error(f"Error saving similarities: {e}")
            return False
        
        return True
    
    @staticmethod
    def add_field_to_data(nested_data: Dict[str, Any], field_data: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Add a new field to the nested data structure.
        
        Args:
            nested_data: The hierarchical research field data
            field_data: Information about the new field
            
        Returns:
            Tuple containing (updated_nested_data, new_field)
        """
        group_name = field_data.get("group")
        subgroup_name = field_data.get("subgroup")
        
        # Extract field information
        new_field = {
            "name": field_data.get("name"),
            "description": {
                "definition": field_data.get("definition", ""),
                "methodologies": field_data.get("methodologies", ""),
                "applications": field_data.get("applications", ""),
                "technologies": field_data.get("technologies", ""),
                
            }
        }
        
        # Find the specified group
        group_found = False
        for category in nested_data.get("categories", []):
            if category["name"] == group_name:
                group_found = True
                
                # Find the specified subgroup
                subgroup_found = False
                for subgroup in category.get("subgroups", []):
                    if subgroup["name"] == subgroup_name:
                        subgroup_found = True
                        
                        # Add the field to the subgroup
                        subgroup.setdefault("fields", []).append(new_field)
                        break
                
                # If subgroup not found, create it
                if not subgroup_found:
                    new_subgroup = {
                        "name": subgroup_name,
                        "fields": [new_field]
                    }
                    category.setdefault("subgroups", []).append(new_subgroup)
                
                break
        
        # If group not found, create it
        if not group_found:
            new_category = {
                "name": group_name,
                "subgroups": [{
                    "name": subgroup_name,
                    "fields": [new_field]
                }]
            }
            nested_data.setdefault("categories", []).append(new_category)
        
        return nested_data, new_field
    
    @staticmethod
    def get_all_field_names(nested_data: Dict[str, Any]) -> List[str]:
        """
        Extract all field names from nested data.
        
        Args:
            nested_data: The hierarchical research field data
            
        Returns:
            List of all field names
        """
        field_names = []
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for field in subgroup.get("fields", []):
                    field_names.append(field["name"])
        return sorted(field_names)
    
    @staticmethod
    def get_all_groups_and_subgroups(nested_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Extract all group and subgroup names from nested data.
        
        Args:
            nested_data: The hierarchical research field data
            
        Returns:
            Tuple containing (groups, subgroups_by_group)
        """
        groups = []
        subgroups = {}
        
        for category in nested_data.get("categories", []):
            group_name = category["name"]
            groups.append(group_name)
            subgroups[group_name] = []
            
            for subgroup in category.get("subgroups", []):
                subgroups[group_name].append(subgroup["name"])
        
        return groups, subgroups
    
    @staticmethod
    def find_similarity(similarities: List[Dict[str, Any]], field1: str, field2: str) -> Optional[float]:
        """
        Find similarity between two specific fields.
        
        Args:
            similarities: List of similarity records
            field1: Name of the first field
            field2: Name of the second field
            
        Returns:
            Similarity score if found, None otherwise
        """
        for sim in similarities:
            if ((sim.get("field1") == field1 and sim.get("field2") == field2) or
                (sim.get("field1") == field2 and sim.get("field2") == field1)):
                return sim.get("similarity_score")
        
        return None
    
    @staticmethod
    def get_field_data(nested_data: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get field data for a specific field.
        
        Args:
            nested_data: The hierarchical research field data
            field_name: Name of the field to find
            
        Returns:
            Field data dictionary if found, None otherwise
        """
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for field in subgroup.get("fields", []):
                    if field["name"] == field_name:
                        return field
        
        return None
    
    @staticmethod
    def get_field_group_info(nested_data: Dict[str, Any], field_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get group and subgroup for a specific field.
        
        Args:
            nested_data: The hierarchical research field data
            field_name: Name of the field to find
            
        Returns:
            Tuple containing (group_name, subgroup_name) or (None, None) if not found
        """
        for category in nested_data.get("categories", []):
            group_name = category["name"]
            for subgroup in category.get("subgroups", []):
                subgroup_name = subgroup["name"]
                for field in subgroup.get("fields", []):
                    if field["name"] == field_name:
                        return group_name, subgroup_name
        
        return None, None