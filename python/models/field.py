"""
Field model definition for representing research fields.
"""
from typing import Dict, List, Any, Optional


class Field:
    """Model representing a research field."""
    
    def __init__(self, name: str, 
                 description: Dict[str, str] = None, 
                 group: str = None, 
                 subgroup: str = None):
        """
        Initialize a Field object.
        
        Args:
            name: The name of the field
            description: Dictionary of field descriptions with facet keys
            group: The general research group this field belongs to
            subgroup: The specific subgroup within the research group
        """
        self.name = name
        self.description = description or {}
        self.group = group
        self.subgroup = subgroup
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Field':
        """
        Create a Field object from a dictionary.
        
        Args:
            data: Dictionary containing field data
            
        Returns:
            A new Field object
        """
        if isinstance(data.get('description'), dict):
            description = data['description']
        else:
            description = {'definition': data.get('description', '')}
            
        return cls(
            name=data['name'],
            description=description,
            group=data.get('group'),
            subgroup=data.get('subgroup')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Field object to a dictionary.
        
        Returns:
            Dictionary representation of the field
        """
        return {
            'name': self.name,
            'description': self.description,
            'group': self.group,
            'subgroup': self.subgroup
        }
    
    def get_full_text(self) -> str:
        """
        Get all text content from the field description.
        
        Returns:
            Concatenated string of all description text
        """
        if isinstance(self.description, str):
            return self.description
            
        return ' '.join(str(text) for text in self.description.values() if text)