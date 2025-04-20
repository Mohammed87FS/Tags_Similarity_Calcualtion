import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import os
import re
from typing import Dict, List, Tuple, Set, Union, Optional

#############################################################
#                   CONFIGURATION SECTION                   #
#############################################################

# File path for the JSON data
JSON_FILE_PATH = "../nested_descriptions_research_groups.json"  # Use your existing JSON file

# Sentence Transformer model to use
# Options include: 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L3-v2', 'all-mpnet-base-v2'
MODEL_NAME = 'all-mpnet-base-v2'

# Similarity calculation parameters
SAME_GROUP_BASELINE = 0.6       # Baseline similarity for fields in same general group
SAME_SUBGROUP_BASELINE = 0.7    # Baseline similarity for fields in same subgroup
SIMILARITY_WEIGHT = 0.3         # Weight of calculated similarity to add to baseline
MAX_CROSS_GROUP_SIMILARITY = 0.7  # Maximum similarity for fields not in same group/subgroup

# Field description property weights (must sum to 1.0)
DESCRIPTION_WEIGHTS = {
    "definition": 0.35,
    "methodologies": 0.30,
    "applications": 0.15,
    "technologies": 0.10,
    "challenges": 0.05,
    "future_directions": 0.05
}

# UPDATED: Adjusted component weights to emphasize embedding more and reduce TF-IDF
COMPONENT_WEIGHTS = {
    "embedding": 0.4,   # Increased from 0.35
    "tfidf": 0.10,      # Decreased from 0.25
    "domain": 0.4,     # Same
    "facet": 0.10       # Same
}

# Domain boosting configuration
ENABLE_DOMAIN_BOOSTING = True    # Whether to apply domain-based score boosting
MAX_BOOST_FACTOR = 0.15          # Maximum boost to apply (0.15 = up to 15% boost)
DOMAIN_BOOST_THRESHOLD = 0.7     # Minimum domain similarity to trigger boost

# Output configuration
OUTPUT_DIR = "outputs_enhanced_multi"
OUTPUT_CSV = f"{OUTPUT_DIR}/field_similarities.csv"
OUTPUT_JSON = f"{OUTPUT_DIR}/field_similarities.json"  

GENERATE_HEATMAP = True
HEATMAP_FILENAME = f"{OUTPUT_DIR}/field_similarities_heatmap.png"
TOP_N_SIMILAR = 5  # Number of most similar fields to display for each field

# Verify weights sum to 1.0
if abs(sum(DESCRIPTION_WEIGHTS.values()) - 1.0) > 0.001:
    raise ValueError(f"Description weights must sum to 1.0, got {sum(DESCRIPTION_WEIGHTS.values())}")

if abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) > 0.001:
    raise ValueError(f"Component weights must sum to 1.0, got {sum(COMPONENT_WEIGHTS.values())}")

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#############################################################
#                DOMAIN KNOWLEDGE SECTION                   #
#############################################################

# ENHANCED: Extended domain-specific term groups with more technical terminology
DOMAIN_TERM_GROUPS = {
    'ai_ml': [
        'artificial intelligence', 'machine learning', 'neural networks', 'deep learning', 
        'supervised learning', 'unsupervised learning', 'reinforcement learning', 
        'natural language processing', 'computer vision', 'data mining', 'knowledge representation',
        'transformer', 'classification', 'clustering', 'regression', 'bayesian', 'generative models',
        'feature engineering', 'backpropagation', 'gradient descent', 'convolutional networks',
        'recurrent neural networks', 'lstm', 'gru', 'attention mechanism', 'transfer learning',
        'fine-tuning', 'hyperparameter', 'overfitting', 'regularization', 'dimensionality reduction',
        'ensemble methods', 'decision trees', 'random forest', 'boosting', 'bagging', 'support vector',
        'nlp', 'tokenization', 'embedding', 'word2vec', 'bert', 'gpt', 'transformer', 'autoencoder',
        'gan', 'generative adversarial', 'self-supervised', 'semi-supervised', 'federated learning',
        'inference', 'prediction', 'algorithm', 'pytorch', 'tensorflow', 'keras', 'scikit-learn'
    ],
    
    'security': [
        'cybersecurity', 'encryption', 'authentication', 'firewall', 'vulnerability',
        'penetration testing', 'intrusion detection', 'security audit', 'threat',
        'malware', 'phishing', 'cryptography', 'zero-day', 'exploit', 'security breach',
        'ransomware', 'access control', 'secure', 'privacy', 'confidentiality', 'integrity',
        'availability', 'threat model', 'risk assessment', 'security policy', 'compliance',
        'data protection', 'identity management', 'multi-factor', 'symmetric encryption',
        'asymmetric encryption', 'public key', 'private key', 'digital signature', 'certificate',
        'pki', 'hash function', 'sha', 'aes', 'rsa', 'elliptic curve', 'key management',
        'network security', 'endpoint security', 'security operations', 'siem', 'incident response',
        'forensics', 'threat intelligence', 'security framework', 'vpn', 'ips', 'ids', 'dlp',
        'waf', 'antivirus', 'patch management', 'vulnerability scanning', 'penetration testing'
    ],
    
    'data_analytics': [
        'analytics', 'big data', 'data science', 'statistics', 'data visualization',
        'business intelligence', 'predictive analytics', 'data mining', 'data warehouse',
        'exploratory analysis', 'regression', 'classification', 'data cleaning', 'etl',
        'dashboard', 'kpi', 'metric', 'database', 'data modeling', 'data engineering',
        'data pipeline', 'data governance', 'data lake', 'data mart', 'olap', 'oltp',
        'sql', 'nosql', 'hadoop', 'spark', 'data streaming', 'real-time analytics',
        'descriptive analytics', 'prescriptive analytics', 'diagnostic analytics',
        'statistical analysis', 'hypothesis testing', 'correlation', 'causation',
        'data quality', 'master data', 'metadata', 'data catalog', 'data dictionary',
        'tableau', 'power bi', 'looker', 'data studio', 'jupyter', 'r studio',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'data preprocessing', 'feature selection',
        'cross-validation', 'time series', 'anomaly detection', 'segmentation'
    ],
    
    'hci': [
        'human-computer interaction', 'user interface', 'user experience', 'usability',
        'interaction design', 'human factors', 'accessibility', 'cognitive load',
        'user research', 'user testing', 'information architecture', 'wireframe',
        'prototype', 'user-centered', 'responsive design', 'affordance', 'mental model',
        'usability testing', 'heuristic evaluation', 'cognitive walkthrough', 'personas',
        'user journey', 'user flow', 'card sorting', 'a/b testing', 'eye tracking',
        'gesture recognition', 'touch interface', 'voice interface', 'multimodal interface',
        'design thinking', 'interaction patterns', 'design system', 'design principles',
        'user needs', 'user goals', 'user feedback', 'user behavior', 'user satisfaction',
        'ui components', 'navigation', 'information hierarchy', 'visual hierarchy',
        'interaction model', 'design critique', 'contextual inquiry', 'ethnography',
        'participatory design', 'accessibility guidelines', 'wcag', 'inclusive design'
    ],
    
    'graphics_media': [
        'rendering', 'visualization', 'animation', 'modeling', '3d graphics',
        'computer graphics', 'virtual reality', 'augmented reality', 'game development',
        'digital media', 'image processing', 'visual effects', 'shader', 'texture',
        'polygon', 'mesh', 'lighting', 'animation', 'ray tracing', 'path tracing',
        'global illumination', 'physically based rendering', 'pbr', 'graphics pipeline',
        'rasterization', 'vertex shader', 'fragment shader', 'geometry shader',
        'tessellation', 'level of detail', 'lod', 'motion capture', 'rigging',
        'skinning', 'inverse kinematics', 'forward kinematics', 'keyframe animation',
        'procedural animation', 'particle system', 'vfx', 'compositing', 'modeling',
        'sculpting', 'texturing', 'uv mapping', 'normal mapping', 'bump mapping',
        'displacement mapping', 'volumetric rendering', 'subsurface scattering',
        'opengl', 'directx', 'vulkan', 'unity', 'unreal engine', 'blender', 'maya'
    ],
    
    'software_development': [
        'software engineering', 'programming', 'code', 'algorithm', 'data structure',
        'framework', 'api', 'software development', 'version control', 'devops',
        'agile', 'testing', 'debugging', 'deployment', 'microservice',
        'full-stack', 'frontend', 'backend', 'web development', 'object-oriented',
        'functional programming', 'declarative programming', 'imperative programming',
        'software architecture', 'design patterns', 'continuous integration',
        'continuous deployment', 'continuous delivery', 'test-driven development',
        'behavior-driven development', 'unit testing', 'integration testing',
        'system testing', 'acceptance testing', 'regression testing', 'code review',
        'pair programming', 'scrum', 'kanban', 'waterfall', 'git', 'github',
        'bitbucket', 'jira', 'jenkins', 'docker', 'kubernetes', 'terraform',
        'infrastructure as code', 'technical debt', 'refactoring', 'code quality',
        'scalability', 'performance optimization', 'caching', 'load balancing'
    ],
    
    'hardware_systems': [
        'hardware', 'cpu', 'gpu', 'processor', 'memory', 'storage', 'network',
        'architecture', 'embedded system', 'circuit', 'sensor', 'actuator',
        'robotics', 'iot', 'edge computing', 'fpga', 'asic', 'microcontroller',
        'microprocessor', 'soc', 'system on chip', 'ram', 'dram', 'sram', 'cache',
        'memory hierarchy', 'virtual memory', 'paging', 'direct memory access',
        'dma', 'pcie', 'usb', 'sata', 'nvme', 'instruction set', 'isa', 'risc',
        'cisc', 'pipelining', 'superscalar', 'branch prediction', 'out-of-order',
        'speculative execution', 'register', 'alu', 'interrupt', 'dma', 'i/o',
        'peripheral', 'bus', 'motherboard', 'firmware', 'bios', 'uefi',
        'hardware acceleration', 'parallel computing', 'distributed systems',
        'fault tolerance', 'redundancy', 'high availability', 'raid'
    ],
    
    'healthcare': [
        'health', 'medical', 'clinical', 'patient', 'diagnosis', 'therapy', 'treatment',
        'healthcare', 'biomedical', 'disease', 'drug', 'hospital', 'physician',
        'telemedicine', 'electronic health record', 'wellness', 'public health',
        'epidemiology', 'preventive medicine', 'health informatics', 'health policy',
        'clinical trial', 'evidence-based medicine', 'personalized medicine',
        'precision medicine', 'genomics', 'proteomics', 'bioinformatics',
        'medical device', 'medical imaging', 'radiology', 'pathology', 'surgery',
        'anesthesia', 'mental health', 'psychiatry', 'psychology', 'chronic disease',
        'acute care', 'primary care', 'secondary care', 'tertiary care',
        'patient-centered', 'health equity', 'health disparities', 'health literacy',
        'health promotion', 'health education', 'health screening', 'vaccination'
    ]
}

# UPDATED: Enhanced domain group similarity matrix with more precise relationships
DOMAIN_GROUP_SIMILARITY = {
    'ai_ml': {
        'ai_ml': 1.0, 
        'data_analytics': 0.85,   # Increased from 0.7
        'security': 0.35,         # Slightly increased
        'hci': 0.45,              # Slightly increased
        'graphics_media': 0.45,   # Slightly increased
        'software_development': 0.55, 
        'hardware_systems': 0.35, 
        'healthcare': 0.35
    },
    'security': {
        'ai_ml': 0.35, 
        'security': 1.0, 
        'data_analytics': 0.35, 
        'hci': 0.25, 
        'graphics_media': 0.15, 
        'software_development': 0.60,  # Increased from 0.5
        'hardware_systems': 0.45, 
        'healthcare': 0.35
    },
    'data_analytics': {
        'ai_ml': 0.85,            # Increased from 0.7
        'security': 0.35, 
        'data_analytics': 1.0, 
        'hci': 0.35, 
        'graphics_media': 0.30, 
        'software_development': 0.45, 
        'hardware_systems': 0.25, 
        'healthcare': 0.55
    },
    'hci': {
        'ai_ml': 0.45, 
        'security': 0.25, 
        'data_analytics': 0.35, 
        'hci': 1.0, 
        'graphics_media': 0.65,   # Increased from 0.6
        'software_development': 0.55, 
        'hardware_systems': 0.35, 
        'healthcare': 0.45
    },
    'graphics_media': {
        'ai_ml': 0.45, 
        'security': 0.15, 
        'data_analytics': 0.30, 
        'hci': 0.65,              # Increased from 0.6
        'graphics_media': 1.0, 
        'software_development': 0.35, 
        'hardware_systems': 0.35, 
        'healthcare': 0.20
    },
    'software_development': {
        'ai_ml': 0.55, 
        'security': 0.60,         # Increased from 0.5
        'data_analytics': 0.45, 
        'hci': 0.55, 
        'graphics_media': 0.35, 
        'software_development': 1.0, 
        'hardware_systems': 0.65, 
        'healthcare': 0.30
    },
    'hardware_systems': {
        'ai_ml': 0.35, 
        'security': 0.45, 
        'data_analytics': 0.25, 
        'hci': 0.35, 
        'graphics_media': 0.35, 
        'software_development': 0.65, 
        'hardware_systems': 1.0, 
        'healthcare': 0.35
    },
    'healthcare': {
        'ai_ml': 0.35, 
        'security': 0.35, 
        'data_analytics': 0.55, 
        'hci': 0.45, 
        'graphics_media': 0.20, 
        'software_development': 0.30, 
        'hardware_systems': 0.35, 
        'healthcare': 1.0
    }
}

#############################################################
#                     UTILITY FUNCTIONS                     #
#############################################################

def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON.")
        return None

def extract_fields_info(data: Dict) -> Tuple[List[Dict], Dict, Dict]:
    """
    Extract fields and their relationships from the JSON data.
    
    Returns:
        - List of field dictionaries with name and description
        - Dictionary mapping field names to their group
        - Dictionary mapping field names to their subgroup
    """
    all_fields = []
    field_to_group = {}
    field_to_subgroup = {}
    
    for category in data["categories"]:
        group_name = category["name"]
        
        for subgroup in category["subgroups"]:
            subgroup_name = subgroup["name"]
            
            for field in subgroup["fields"]:
                field_name = field["name"]
                field_description = field["description"]
                
                all_fields.append({
                    "name": field_name,
                    "description": field_description
                })
                
                field_to_group[field_name] = group_name
                field_to_subgroup[field_name] = subgroup_name
    
    return all_fields, field_to_group, field_to_subgroup

#############################################################
#               MULTI-FACETED SIMILARITY CLASS              #
#############################################################

class EnhancedFieldComparator:
    """
    An enhanced approach to compare research fields using multiple similarity measures
    with improved calibration, technical term extraction, and domain boosting.
    """
    
    def __init__(self, 
                 model_name: str = MODEL_NAME,
                 facet_weights: Dict[str, float] = DESCRIPTION_WEIGHTS,
                 component_weights: Dict[str, float] = COMPONENT_WEIGHTS,
                 domain_terms: Dict[str, List[str]] = DOMAIN_TERM_GROUPS,
                 domain_similarities: Dict[str, Dict[str, float]] = DOMAIN_GROUP_SIMILARITY,
                 enable_domain_boost: bool = ENABLE_DOMAIN_BOOSTING,
                 max_boost_factor: float = MAX_BOOST_FACTOR,
                 domain_boost_threshold: float = DOMAIN_BOOST_THRESHOLD,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize the comparator with various sub-components
        """
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Load spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            print("Warning: spaCy model 'en_core_web_sm' not found. Using basic text processing instead.")
            print("To install: python -m spacy download en_core_web_sm")
            self.use_spacy = False
        
        # Configuration
        self.facet_weights = facet_weights
        self.component_weights = component_weights
        self.domain_terms = domain_terms
        self.domain_similarities = domain_similarities
        self.enable_domain_boost = enable_domain_boost
        self.max_boost_factor = max_boost_factor
        self.domain_boost_threshold = domain_boost_threshold
        
        # Cache for embeddings and domain concepts to avoid recalculation
        self.embedding_cache = {}
        self.domain_concept_cache = {}
        self.tfidf_similarity_cache = {}
        
        # Preprocess domain terms for faster lookup
        self.domain_term_lookup = {}
        for domain, terms in self.domain_terms.items():
            for term in terms:
                self.domain_term_lookup[term] = domain
        
        # Technical patterns for term extraction
        self.technical_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)+\b',   # CamelCase terms (e.g., "Machine Learning")
            r'\b[a-z]+(?:-[a-z]+)+\b',                    # hyphenated terms (e.g., "client-server")
            r'\b[A-Za-z]+\d+[A-Za-z]*\b',                # terms with numbers (e.g., "IPv6")
            r'\b[A-Za-z]+\.[A-Za-z]+\b',                 # software libraries (e.g., "TensorFlow.js")
            r'\b[A-Z][A-Z0-9]+\b',                       # acronyms (e.g., "API", "GPU")
        ]
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching for efficiency"""
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
            
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.model.encode(text, show_progress_bar=False)
        return self.embedding_cache[text]
    
    def extract_domain_concepts(self, text: str) -> Dict[str, List[str]]:
        """
        Enhanced method to extract domain-specific concepts from text, organized by domain.
        Now with better technical term extraction.
        """
        if not text.strip():
            return {domain: [] for domain in self.domain_terms}
            
        # Check cache first
        if text in self.domain_concept_cache:
            return self.domain_concept_cache[text]
            
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Initialize result dictionary
        domain_concepts = {domain: [] for domain in self.domain_terms}
        
        # Add general concepts category
        domain_concepts['general'] = []
        
        # Process with spaCy if available for better NLP
        if self.use_spacy:
            doc = self.nlp(text)
            
            # Extract noun phrases and other potentially relevant terms
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
            domain_concepts['general'].extend(noun_phrases)
            
            # Extract technical terms (nouns with compound modifiers)
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
    
    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between text embeddings"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Apply scaling to improve contrast between similar vs dissimilar fields
        return self._scale_similarity(similarity)
    
    def _scale_similarity(self, raw_similarity: float) -> float:
        """
        ENHANCED: Improved sigmoid-like scaling to provide better contrast
        between similar and dissimilar fields
        """
        # Parameters for scaling
        midpoint = 0.45  # Lowered from 0.5 to boost scores above this threshold
        steepness = 7.0  # Increased from 5.0 for sharper contrast
        
        # Handle extreme values directly
        if raw_similarity >= 0.95: return 1.0
        if raw_similarity <= 0.05: return 0.0
        
        # Apply sigmoid transformation centered at midpoint
        scaled = 1.0 / (1.0 + np.exp(-steepness * (raw_similarity - midpoint)))
        
        # Normalize to [0,1] range
        min_val = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
        max_val = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        scaled_normalized = (scaled - min_val) / (max_val - min_val)
        
        return scaled_normalized
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        ENHANCED: Improved TF-IDF similarity with better handling of technical terms,
        now with preprocessing to emphasize technical terminology
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
            if self.use_spacy:
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
            max_features=10000   # Increased from 5000
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
        except:
            # If vectorization fails (e.g., with very short texts)
            return 0.0
    
    def calculate_domain_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity based on domain-specific concepts with improved
        weighting and more comprehensive domain knowledge
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
                group_similarity = self.domain_similarities.get(domain1, {}).get(domain2, 0.1)
                
                # Calculate the weight based on number and quality of concepts
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
                
                # Give more weight to general concept overlap if there are many matching terms
                general_weight = min(20, len(general_concepts1) + len(general_concepts2))
                
                domain_similarities.append(jaccard)
                domain_weights.append(general_weight)
        
        # Calculate weighted average
        if domain_similarities and sum(domain_weights) > 0:
            domain_sim = sum(s * w for s, w in zip(domain_similarities, domain_weights)) / sum(domain_weights)
            return domain_sim
        
        return 0.0
    
    def _detect_primary_domains(self, field: Dict) -> List[str]:
        """
        Detect the primary domains of a field based on its terminology.
        Used for domain boosting.
        """
        full_text = self._get_full_text(field)
        domain_concepts = self.extract_domain_concepts(full_text)
        
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
        
    def calculate_faceted_similarity(self, field1: Dict, field2: Dict) -> Dict[str, float]:
        """Calculate similarities for each facet between two fields"""
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
        for facet, weight in self.facet_weights.items():
            text1 = desc1.get(facet, '')
            text2 = desc2.get(facet, '')
            
            if not text1 or not text2:
                facet_similarities[facet] = 0.0
                continue
            
            # For each facet, calculate a composite similarity
            embedding_sim = self.calculate_embedding_similarity(text1, text2)
            tfidf_sim = self.calculate_tfidf_similarity(text1, text2)
            domain_sim = self.calculate_domain_similarity(text1, text2)
            
            # Weight the components for this facet
            facet_sim = (
                self.component_weights['embedding'] * embedding_sim +
                self.component_weights['tfidf'] * tfidf_sim +
                self.component_weights['domain'] * domain_sim
            )
            
            facet_similarities[facet] = facet_sim
        
        return facet_similarities
    
    def compare_fields(self, field1: Dict, field2: Dict, detailed: bool = False) -> Union[float, Dict]:
        """
        ENHANCED: Calculate similarity between two research fields using multiple methods,
        now with domain boosting and improved calibration
        
        Args:
            field1: First field dictionary with name and description
            field2: Second field dictionary with name and description
            detailed: Whether to return detailed breakdown of similarity
            
        Returns:
            Either a single similarity score or a detailed breakdown
        """
        # Handle identity comparison
        if field1['name'] == field2['name']:
            if detailed:
                return {
                    'overall_similarity': 1.0,
                    'facet_similarities': {facet: 1.0 for facet in self.facet_weights},
                    'component_similarities': {comp: 1.0 for comp in self.component_weights}
                }
            return 1.0
        
        # Get faceted similarities
        facet_similarities = self.calculate_faceted_similarity(field1, field2)
        
        # Calculate weighted facet score
        weighted_facet_sim = 0.0
        total_weight = 0.0
        
        for facet, sim in facet_similarities.items():
            weight = self.facet_weights.get(facet, 0.0)
            weighted_facet_sim += sim * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_facet_sim /= total_weight
        
        # Get full text from both fields
        full_text1 = self._get_full_text(field1)
        full_text2 = self._get_full_text(field2)
        
        # Calculate component similarities on full text
        component_similarities = {
            'embedding': self.calculate_embedding_similarity(full_text1, full_text2),
            'tfidf': self.calculate_tfidf_similarity(full_text1, full_text2),
            'domain': self.calculate_domain_similarity(full_text1, full_text2),
            'facet': weighted_facet_sim  # Also include the faceted similarity
        }
        
        # Calculate overall similarity
        overall_similarity = sum(
            component_similarities[comp] * weight
            for comp, weight in self.component_weights.items()
        )
        
        # Apply domain boosting if enabled
        boosted_similarity = overall_similarity
        boost_applied = 0.0
        
        if self.enable_domain_boost:
            field1_domains = self._detect_primary_domains(field1)
            field2_domains = self._detect_primary_domains(field2)
            
            # Calculate max domain relatedness
            max_domain_sim = 0.0
            for d1 in field1_domains:
                for d2 in field2_domains:
                    domain_sim = self.domain_similarities.get(d1, {}).get(d2, 0.0)
                    max_domain_sim = max(max_domain_sim, domain_sim)
            
            # Apply boosting for highly related domains
            if max_domain_sim > self.domain_boost_threshold:
                boost_factor = self.max_boost_factor * (max_domain_sim - self.domain_boost_threshold) / (1.0 - self.domain_boost_threshold)
                boost_applied = boost_factor
                boosted_similarity = min(1.0, overall_similarity + boost_factor)
        
        # Apply final calibration for more intuitive scores
        calibrated_similarity = self._calibrate_final_score(boosted_similarity)
        
        if detailed:
            return {
                'overall_similarity': calibrated_similarity,
                'raw_similarity': overall_similarity,
                'boosted_similarity': boosted_similarity if self.enable_domain_boost else overall_similarity,
                'boost_applied': boost_applied,
                'facet_similarities': facet_similarities,
                'component_similarities': component_similarities
            }
        
        return calibrated_similarity
    
    def _get_full_text(self, field: Dict) -> str:
        """Extract all text from a field description"""
        desc = field.get('description', {})
        
        if isinstance(desc, str):
            return desc
        
        return ' '.join(str(text) for text in desc.values() if text)
    
    def _calibrate_final_score(self, score: float) -> float:
        """
        ENHANCED: Apply stronger calibration to get more intuitive similarity scores
        with better separation between related and unrelated fields
        """
        # Parameters for final calibration (more aggressive than before)
        midpoint = 0.40  # Lowered from 0.5 to boost scores above this threshold
        steepness = 8.0  # Increased from 6.0 for even sharper contrast
        
        # Special cases
        if score >= 0.95: return 1.0
        if score <= 0.05: return 0.0
            
        # Apply sigmoid function and normalize
        calibrated = 1.0 / (1.0 + np.exp(-steepness * (score - midpoint)))
        min_val = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
        max_val = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        
        calibrated_normalized = (calibrated - min_val) / (max_val - min_val)
        
        return calibrated_normalized
    
    def calculate_all_similarities(self, fields: List[Dict]) -> pd.DataFrame:
        """Calculate similarities between all field pairs"""
        n = len(fields)
        field_names = [field['name'] for field in fields]
        
        # Initialize similarity matrix
        similarities = np.zeros((n, n))
        
        # Calculate similarities
        print(f"Calculating similarities for {n} fields using enhanced multi-faceted approach...")
        for i in range(n):
            similarities[i, i] = 1.0  # Self-similarity
            
            for j in range(i+1, n):
                similarity = self.compare_fields(fields[i], fields[j])
                similarities[i, j] = similarity
                similarities[j, i] = similarity  # Matrix is symmetric
        
        # Create DataFrame
        similarity_df = pd.DataFrame(similarities, index=field_names, columns=field_names)
        return similarity_df

#############################################################
#                OUTPUT & VISUALIZATION FUNCTIONS           #
#############################################################

def find_top_similar_fields(similarity_df: pd.DataFrame, n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """Find the top N most similar fields for each field."""
    result = {}
    
    for field in similarity_df.index:
        # Get similarities, sort, and drop the field itself (which would have similarity 1.0)
        similarities = similarity_df.loc[field].drop(field).sort_values(ascending=False)
        top_n = similarities.head(n)
        
        result[field] = [(name, score) for name, score in top_n.items()]
    
    return result

def generate_heatmap(similarity_df: pd.DataFrame, field_to_group: Dict[str, str], filename: str = "heatmap.png"):
    """Generate a heatmap visualization of the similarity matrix."""
    plt.figure(figsize=(20, 16))
    
    # Sort the DataFrame by group and subgroup for better visualization
    field_names = similarity_df.index.tolist()
    simplified_group_names = {field: group.split(' & ')[0] for field, group in field_to_group.items()}
    
    # Sort by group
    sorted_fields = sorted(field_names, key=lambda x: simplified_group_names.get(x, ''))
    sorted_df = similarity_df.loc[sorted_fields, sorted_fields]
    
    # Create heatmap
    sns.heatmap(sorted_df, cmap="viridis", vmin=0, vmax=1, annot=False)
    
    plt.title("Field Similarity Heatmap (Enhanced Multi-Faceted Approach)", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {filename}")

def create_similarity_json(similarity_df: pd.DataFrame) -> List[Dict]:
    """Convert similarity DataFrame to a list of field pairs with similarity scores."""
    result = []
    field_names = similarity_df.index.tolist()
    
    # Only include each pair once (avoid duplicates like A-B and B-A)
    for i, field1 in enumerate(field_names):
        for j, field2 in enumerate(field_names[i+1:], i+1):
            similarity = similarity_df.loc[field1, field2]
            result.append({
                "field1": field1,
                "field2": field2,
                "similarity_score": float(similarity)  # Convert numpy float to Python float for JSON serialization
            })
    
    return result

def create_similarity_table(similarity_df: pd.DataFrame) -> pd.DataFrame:
    """Convert similarity DataFrame to a tabular format with field1, field2, similarity_score columns."""
    rows = []
    field_names = similarity_df.index.tolist()
    
    # Only include each pair once (avoid duplicates like A-B and B-A)
    for i, field1 in enumerate(field_names):
        for j, field2 in enumerate(field_names[i+1:], i+1):
            similarity = similarity_df.loc[field1, field2]
            rows.append({
                "field1": field1,
                "field2": field2,
                "similarity_score": float(similarity)  # Convert numpy float to Python float
            })
    
    return pd.DataFrame(rows)

def print_similarity_analysis(comparator: EnhancedFieldComparator, fields: List[Dict]) -> None:
    """Print detailed analysis of similarities between a few example fields"""
    # Select a few interesting fields to compare
    example_fields = []
    field_names = [field["name"] for field in fields]
    
    target_fields = ["Artificial Intelligence", "Machine Learning", "Cybersecurity", 
                    "Data Science", "Human-Computer Interaction"]
    
    for target in target_fields:
        for field in fields:
            if field["name"] == target:
                example_fields.append(field)
                break
    
    # If we didn't find all target fields, add some from the available fields
    if len(example_fields) < 3:
        for field in fields[:5-len(example_fields)]:
            if field not in example_fields:
                example_fields.append(field)
    
    print("\nDetailed similarity analysis for selected field pairs:")
    print("-------------------------------------------------------")
    
    # Compare pairs of fields
    for i, field1 in enumerate(example_fields):
        for j, field2 in enumerate(example_fields[i+1:], i+1):
            print(f"\nComparing {field1['name']} and {field2['name']}:")
            detailed = comparator.compare_fields(field1, field2, detailed=True)
            
            print(f"  Overall similarity: {detailed['overall_similarity']:.4f}")
            if comparator.enable_domain_boost and detailed['boost_applied'] > 0:
                print(f"  Domain boost applied: +{detailed['boost_applied']:.4f}")
            
            print("\n  Component similarities:")
            for comp, score in detailed['component_similarities'].items():
                print(f"    - {comp}: {score:.4f}")
            
            print("\n  Facet similarities:")
            for facet, score in detailed['facet_similarities'].items():
                if score > 0:  # Only show non-zero facets
                    print(f"    - {facet}: {score:.4f}")
            
            print("-" * 50)

#############################################################
#                      MAIN EXECUTION                       #
#############################################################

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Step 1: Load the JSON data
    print(f"Loading data from {JSON_FILE_PATH}...")
    data = load_json_data(JSON_FILE_PATH)
    if not data:
        return
    
    # Step 2: Extract fields and their relationships
    print("Extracting field information...")
    fields, field_to_group, field_to_subgroup = extract_fields_info(data)
    print(f"Found {len(fields)} fields across {len(set(field_to_group.values()))} groups and {len(set(field_to_subgroup.values()))} subgroups.")
    
    # Step 3: Calculate similarities using enhanced multi-faceted approach
    print(f"Creating enhanced field comparator using {MODEL_NAME}...")
    comparator = EnhancedFieldComparator(
        model_name=MODEL_NAME,
        facet_weights=DESCRIPTION_WEIGHTS,
        component_weights=COMPONENT_WEIGHTS,
        domain_terms=DOMAIN_TERM_GROUPS,
        domain_similarities=DOMAIN_GROUP_SIMILARITY,
        enable_domain_boost=ENABLE_DOMAIN_BOOSTING,
        max_boost_factor=MAX_BOOST_FACTOR,
        domain_boost_threshold=DOMAIN_BOOST_THRESHOLD,
        random_seed=RANDOM_SEED
    )
    
    # Calculate all field similarities
    similarity_df = comparator.calculate_all_similarities(fields)
    
    # Print detailed analysis of specific field pairs
    print_similarity_analysis(comparator, fields)
    
    # Step 4: Generate output files
    print(f"Saving similarity data to {OUTPUT_CSV}...")
    similarity_table = create_similarity_table(similarity_df)
    similarity_table.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(similarity_table)} field pairs to CSV file.")

    print(f"Saving similarity data to {OUTPUT_JSON}...")
    similarity_pairs = create_similarity_json(similarity_df)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as json_file:
        json.dump(similarity_pairs, json_file, indent=2)
    print(f"Exported {len(similarity_pairs)} field pairs to JSON file.")
    
    # Step 5: Generate heatmap if requested
    if GENERATE_HEATMAP:
        print("Generating heatmap visualization...")
        generate_heatmap(similarity_df, field_to_group, HEATMAP_FILENAME)
    
    # Step 6: Find and print top similar fields
    print(f"\nTop {TOP_N_SIMILAR} most similar fields for each field:")
    top_similar = find_top_similar_fields(similarity_df, TOP_N_SIMILAR)
    
    for field, similar_fields in list(top_similar.items())[:10]:  # Show only first 10 fields to avoid too much output
        print(f"\n{field}:")
        for similar_field, score in similar_fields:
            print(f"  - {similar_field}: {score:.4f}")
    
    print(f"\n... and {len(top_similar) - 10} more fields (check CSV for complete results)")

if __name__ == "__main__":
    main()