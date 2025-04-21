"""
Service for domain-specific similarity calculations.
"""

import re
import logging
from typing import Dict, List, Set
import spacy

from config import DOMAIN_TERM_GROUPS, DOMAIN_GROUP_SIMILARITY

logger = logging.getLogger(__name__)

class DomainService:
    """Service for domain-specific operations."""
    
    def __init__(self):
        """Initialize the domain service."""
        self.domain_concept_cache = {}
        
        # Load spaCy for NLP tasks if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except Exception as e:
            logger.warning(f"spaCy model not found. Using basic text processing: {e}")
            self.use_spacy = False
            self.nlp = None
            
        # Technical patterns for term extraction
        self.technical_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)+\b',   # CamelCase terms
            r'\b[a-z]+(?:-[a-z]+)+\b',                    # hyphenated terms
            r'\b[A-Za-z]+\d+[A-Za-z]*\b',                # terms with numbers
            r'\b[A-Za-z]+\.[A-Za-z]+\b',                 # software libraries
            r'\b[A-Z][A-Z0-9]+\b',                       # acronyms
        ]
        
        # Initialize term lookups
        self.domain_term_lookup = {}
        for domain, terms in DOMAIN_TERM_GROUPS.items():
            for term in terms:
                self.domain_term_lookup[term] = domain
    
    def extract_domain_concepts(self, text: str) -> Dict[str, List[str]]:
        """
        Extract domain-specific concepts from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping domains to lists of concepts
        """
        if not text.strip():
            return {domain: [] for domain in DOMAIN_TERM_GROUPS}
            
        # Check cache first
        if text in self.domain_concept_cache:
            return self.domain_concept_cache[text]
            
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Initialize result dictionary
        domain_concepts = {domain: [] for domain in DOMAIN_TERM_GROUPS}
        domain_concepts['general'] = []
        
        # Process with spaCy if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
            domain_concepts['general'].extend(noun_phrases)
            
            # Extract technical terms
            for token in doc:
                if token.pos_ == "NOUN" and token.dep_ == "compound":
                    head = token.head.text
                    compound_term = f"{token.text} {head}".lower()
                    domain_concepts['general'].append(compound_term)
                
                # Extract specialized technical terms
                if token.pos_ == "NOUN" and any(mod.pos_ == "ADJ" for mod in token.children):
                    adj_mods = [mod.text for mod in token.children if mod.pos_ == "ADJ"]
                    if adj_mods:
                        tech_term = f"{' '.join(adj_mods)} {token.text}".lower()
                        domain_concepts['general'].append(tech_term)
        
        # Extract technical terms using regex patterns
        for pattern in self.technical_patterns:
            tech_terms = re.findall(pattern, text)
            if tech_terms:
                domain_concepts['general'].extend([t.lower() for t in tech_terms])
        
        # Match with domain-specific terminology
        for term, domain in self.domain_term_lookup.items():
            if term in text_lower:
                domain_concepts[domain].append(term)
        
        # Store in cache
        self.domain_concept_cache[text] = domain_concepts
        
        return domain_concepts
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity based on domain concepts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0,1] range
        """
        # Extract domain concepts
        domain_concepts1 = self.extract_domain_concepts(text1)
        domain_concepts2 = self.extract_domain_concepts(text2)
        
        # Calculate domain group similarity scores
        domain_similarities = []
        domain_weights = []
        
        # Compare each pair of domains
        for domain1, concepts1 in domain_concepts1.items():
            if domain1 == 'general' or not concepts1:
                continue
                
            for domain2, concepts2 in domain_concepts2.items():
                if domain2 == 'general' or not concepts2:
                    continue
                
                # Look up similarity between these domain groups
                group_similarity = DOMAIN_GROUP_SIMILARITY.get(domain1, {}).get(domain2, 0.1)
                
                # Calculate weight based on concept counts
                weight = len(concepts1) * len(concepts2)
                
                # Apply higher weight for exact same domain
                if domain1 == domain2 and len(concepts1) >= 2 and len(concepts2) >= 2:
                    weight *= 2.0  # Double the weight for same-domain matches
                
                domain_similarities.append(group_similarity)
                domain_weights.append(weight)
        
        # Calculate general concept overlap using Jaccard similarity
        general_concepts1 = set(domain_concepts1.get('general', []))
        general_concepts2 = set(domain_concepts2.get('general', []))
        
        if general_concepts1 and general_concepts2:
            intersection = general_concepts1.intersection(general_concepts2)
            union = general_concepts1.union(general_concepts2)
            
            if union:
                jaccard = len(intersection) / len(union)
                general_weight = min(20, len(general_concepts1) + len(general_concepts2))
                
                domain_similarities.append(jaccard)
                domain_weights.append(general_weight)
        
        # Calculate weighted average
        if domain_similarities and sum(domain_weights) > 0:
            domain_sim = sum(s * w for s, w in zip(domain_similarities, domain_weights)) / sum(domain_weights)
            return domain_sim
        
        return 0.0
    
    def detect_primary_domains(self, field_text: str) -> List[str]:
        """
        Detect the primary domains of a field based on its terminology.
        
        Args:
            field_text: Text content of the field
            
        Returns:
            List of primary domain names
        """
        domain_concepts = self.extract_domain_concepts(field_text)
        
        # Count concepts in each domain
        domain_counts = {domain: len(concepts) for domain, concepts in domain_concepts.items() 
                        if domain != 'general'}
        
        # Find domains with significant concept counts
        if not domain_counts:
            return []
            
        max_count = max(domain_counts.values())
        if max_count == 0:
            return []
            
        # Consider domains with at least 50% as many concepts as the top domain
        primary_domains = [domain for domain, count in domain_counts.items() 
                          if count >= max_count * 0.5]
        
        return primary_domains