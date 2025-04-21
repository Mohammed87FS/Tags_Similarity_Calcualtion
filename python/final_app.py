import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from typing import Dict, List, Tuple, Union

# Configuration
DATA_DIR = 'data'
NESTED_DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'nested_descriptions_research_groups.json')
SIMILARITY_FILE = os.path.join(DATA_DIR, 'field_similarities.json')

# Group-based similarity parameters
SAME_GROUP_BASELINE = 0.7        # Baseline similarity for fields in same general group
SAME_SUBGROUP_BASELINE = 0.75    # Baseline similarity for fields in same subgroup
SIMILARITY_WEIGHT_SUB = 0.2      # Weight of calculated similarity to add to baseline for subgroup
SIMILARITY_WEIGHT_GENERAL = 0.15 # Weight of calculated similarity to add to baseline for general group
MAX_CROSS_GROUP_SIMILARITY = 0.7 # Maximum similarity for fields not in same group/subgroup

# Field description property weights
DESCRIPTION_WEIGHTS = {
    "definition": 0.35,
    "methodologies": 0.30,
    "applications": 0.15,
    "technologies": 0.10,
    "challenges": 0.05,
    "future_directions": 0.05
}

# Component weights
COMPONENT_WEIGHTS = {
    "embedding": 0.4,
    "tfidf": 0.10,
    "domain": 0.4,
    "facet": 0.10
}

# Domain boosting configuration
ENABLE_DOMAIN_BOOSTING = True    # Whether to apply domain-based score boosting
MAX_BOOST_FACTOR = 0.15          # Maximum boost to apply (0.15 = up to 15% boost)
DOMAIN_BOOST_THRESHOLD = 0.7     # Minimum domain similarity to trigger boost

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

# Domain group similarity matrix with precise relationships
DOMAIN_GROUP_SIMILARITY = {
    'ai_ml': {
        'ai_ml': 1.0, 
        'data_analytics': 0.85,
        'security': 0.35,
        'hci': 0.45,
        'graphics_media': 0.45,
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
        'software_development': 0.60,
        'hardware_systems': 0.45, 
        'healthcare': 0.35
    },
    'data_analytics': {
        'ai_ml': 0.85,
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
        'graphics_media': 0.65,
        'software_development': 0.55, 
        'hardware_systems': 0.35, 
        'healthcare': 0.45
    },
    'graphics_media': {
        'ai_ml': 0.45, 
        'security': 0.15, 
        'data_analytics': 0.30, 
        'hci': 0.65,
        'graphics_media': 1.0, 
        'software_development': 0.35, 
        'hardware_systems': 0.35, 
        'healthcare': 0.20
    },
    'software_development': {
        'ai_ml': 0.55, 
        'security': 0.60,
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

# Initialize Flask application
app = Flask(__name__)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Similarity calculation class
class FieldComparator:
    """Complete implementation of the EnhancedFieldComparator for the web app"""
    
    def __init__(self):
        # Load model
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load spaCy for NLP tasks if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            print("Warning: spaCy model not found. Using basic text processing.")
            self.use_spacy = False
            self.nlp = None

        # Caches for efficiency
        self.embedding_cache = {}
        self.domain_concept_cache = {}
        self.tfidf_similarity_cache = {}
        
        # Field to group mappings
        self.field_to_group = {}
        self.field_to_subgroup = {}
        self.load_field_mappings()
        
        # Initialize term lookups
        self.domain_term_lookup = {}
        for domain, terms in DOMAIN_TERM_GROUPS.items():
            for term in terms:
                self.domain_term_lookup[term] = domain
        
        # Technical patterns for term extraction
        self.technical_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)+\b',   # CamelCase terms
            r'\b[a-z]+(?:-[a-z]+)+\b',                    # hyphenated terms
            r'\b[A-Za-z]+\d+[A-Za-z]*\b',                # terms with numbers
            r'\b[A-Za-z]+\.[A-Za-z]+\b',                 # software libraries
            r'\b[A-Z][A-Z0-9]+\b',                       # acronyms
        ]
    
    def load_field_mappings(self):
        """Load field to group/subgroup mappings from JSON file"""
        try:
            with open(NESTED_DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Map fields to their groups and subgroups
            self.field_to_group = {}
            self.field_to_subgroup = {}
            
            for category in data.get("categories", []):
                group_name = category["name"]
                
                for subgroup in category.get("subgroups", []):
                    subgroup_name = subgroup["name"]
                    
                    for field in subgroup.get("fields", []):
                        field_name = field["name"]
                        self.field_to_group[field_name] = group_name
                        self.field_to_subgroup[field_name] = subgroup_name
                        
            print(f"Loaded mappings for {len(self.field_to_group)} fields")
        except Exception as e:
            print(f"Error loading field mappings: {e}")
            self.field_to_group = {}
            self.field_to_subgroup = {}
    
    def get_embedding(self, text):
        """Get embedding for text with caching for efficiency"""
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
            
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.model.encode(text, show_progress_bar=False)
        return self.embedding_cache[text]
    
    def calculate_embedding_similarity(self, text1, text2):
        """Calculate cosine similarity between text embeddings"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Scale similarity
        return self._scale_similarity(similarity)
    
    def _scale_similarity(self, raw_similarity):
        """Scale similarity with sigmoid-like function"""
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
    
    def calculate_tfidf_similarity(self, text1, text2):
        """Calculate TF-IDF similarity between texts"""
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
        except:
            # If vectorization fails (e.g., with very short texts)
            return 0.0
    
    def extract_domain_concepts(self, text):
        """Extract domain-specific concepts from text"""
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
    
    def calculate_domain_similarity(self, text1, text2):
        """Calculate similarity based on domain concepts"""
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
    
    def _detect_primary_domains(self, field):
        """Detect the primary domains of a field based on its terminology"""
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
    
    def calculate_faceted_similarity(self, field1, field2):
        """Calculate similarities for each facet"""
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
            embedding_sim = self.calculate_embedding_similarity(text1, text2)
            tfidf_sim = self.calculate_tfidf_similarity(text1, text2)
            domain_sim = self.calculate_domain_similarity(text1, text2)
            
            # Weight the components for this facet
            facet_sim = (
                COMPONENT_WEIGHTS['embedding'] * embedding_sim +
                COMPONENT_WEIGHTS['tfidf'] * tfidf_sim +
                COMPONENT_WEIGHTS['domain'] * domain_sim
            )
            
            facet_similarities[facet] = facet_sim
        
        return facet_similarities
    
    def _get_full_text(self, field):
        """Extract all text from a field description"""
        desc = field.get('description', {})
        
        if isinstance(desc, str):
            return desc
        
        return ' '.join(str(text) for text in desc.values() if text)
    
    def _calibrate_final_score(self, score):
        """Apply sigmoid calibration to score"""
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
    
    def compare_fields(self, field1, field2):
        """Calculate similarity between two research fields"""
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
        embedding_sim = self.calculate_embedding_similarity(full_text1, full_text2)
        tfidf_sim = self.calculate_tfidf_similarity(full_text1, full_text2)
        domain_sim = self.calculate_domain_similarity(full_text1, full_text2)
        
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
            field1_domains = self._detect_primary_domains(field1)
            field2_domains = self._detect_primary_domains(field2)
            
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
        calibrated_similarity = self._calibrate_final_score(boosted_similarity)
        
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


def load_data():
    """Load data from JSON files"""
    # Initialize with empty structures if files don't exist
    nested_data = {"categories": []}
    similarities = []
    
    # Print file paths for debugging
    print(f"Looking for nested descriptions at: {NESTED_DESCRIPTIONS_FILE}")
    print(f"Looking for similarities at: {SIMILARITY_FILE}")
    
    # Load nested descriptions if file exists
    if os.path.exists(NESTED_DESCRIPTIONS_FILE):
        try:
            with open(NESTED_DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
                nested_data = json.load(f)
            print(f"Successfully loaded nested data with {len(nested_data.get('categories', []))} categories")
        except Exception as e:
            print(f"Error loading nested descriptions: {e}")
    else:
        print(f"Warning: Nested descriptions file not found at {NESTED_DESCRIPTIONS_FILE}")
        # Create an empty structure if file doesn't exist
        nested_data = {"categories": []}
    
    # Load similarities if file exists
    if os.path.exists(SIMILARITY_FILE):
        try:
            with open(SIMILARITY_FILE, 'r', encoding='utf-8') as f:
                similarities = json.load(f)
            print(f"Successfully loaded {len(similarities)} similarity records")
        except Exception as e:
            print(f"Error loading similarities: {e}")
    else:
        print(f"Warning: Similarities file not found at {SIMILARITY_FILE}")
        # Create an empty list if file doesn't exist
        similarities = []
    
    # Extract field names for debugging
    field_names = get_all_field_names(nested_data)
    print(f"Extracted {len(field_names)} field names: {field_names[:5] if field_names else 'None'}")
    
    return nested_data, similarities


def save_data(nested_data, similarities):
    """Save data to JSON files"""
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(NESTED_DESCRIPTIONS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(SIMILARITY_FILE), exist_ok=True)
    
    # Save nested descriptions
    try:
        with open(NESTED_DESCRIPTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(nested_data, f, indent=2)
        print(f"Successfully saved nested descriptions to {NESTED_DESCRIPTIONS_FILE}")
    except Exception as e:
        print(f"Error saving nested descriptions: {e}")
        return False
    
    # Save similarities
    try:
        with open(SIMILARITY_FILE, 'w', encoding='utf-8') as f:
            json.dump(similarities, f, indent=2)
        print(f"Successfully saved {len(similarities)} similarity records to {SIMILARITY_FILE}")
    except Exception as e:
        print(f"Error saving similarities: {e}")
        return False
    
    return True


def add_field_to_data(nested_data, field_data):
    """Add a new field to the nested data structure"""
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
            "challenges": field_data.get("challenges", ""),
            "future_directions": field_data.get("future_directions", "")
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


def calculate_new_similarities(nested_data, similarities, new_field):
    """Calculate similarities between new field and existing fields"""
    # Initialize field comparator
    comparator = FieldComparator()
    
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
        similarity = comparator.compare_fields(new_field, existing_field)
        
        # Add to new similarities
        new_similarities.append({
            "field1": new_field["name"],
            "field2": existing_field["name"],
            "similarity_score": float(similarity)  # Convert numpy float to Python float
        })
    
    # Combine existing and new similarities
    updated_similarities = similarities + new_similarities
    print(f"Calculated {len(new_similarities)} new similarity scores for {new_field['name']}")
    
    return updated_similarities


def get_all_field_names(nested_data):
    """Extract all field names from nested data"""
    field_names = []
    for category in nested_data.get("categories", []):
        for subgroup in category.get("subgroups", []):
            for field in subgroup.get("fields", []):
                field_names.append(field["name"])
    return sorted(field_names)


def get_all_groups_and_subgroups(nested_data):
    """Extract all group and subgroup names from nested data"""
    groups = []
    subgroups = {}
    
    for category in nested_data.get("categories", []):
        group_name = category["name"]
        groups.append(group_name)
        subgroups[group_name] = []
        
        for subgroup in category.get("subgroups", []):
            subgroups[group_name].append(subgroup["name"])
    
    return groups, subgroups


def find_similarity(similarities, field1, field2):
    """Find similarity between two specific fields"""
    for sim in similarities:
        if ((sim.get("field1") == field1 and sim.get("field2") == field2) or
            (sim.get("field1") == field2 and sim.get("field2") == field1)):
            return sim.get("similarity_score")
    
    return None


def get_field_data(nested_data, field_name):
    """Get field data for a specific field"""
    for category in nested_data.get("categories", []):
        for subgroup in category.get("subgroups", []):
            for field in subgroup.get("fields", []):
                if field["name"] == field_name:
                    return field
    
    return None


@app.route('/')
def index():
    """Home page"""
    # Check if data files exist and create sample data if not
    if not os.path.exists(NESTED_DESCRIPTIONS_FILE) or not os.path.exists(SIMILARITY_FILE):
        print("Data files not found")
     
    
    # Load data
    nested_data, similarities = load_data()
    field_names = get_all_field_names(nested_data)
    groups, subgroups = get_all_groups_and_subgroups(nested_data)
    
    # Debug information
    print(f"Rendering template with {len(field_names)} fields and {len(groups)} groups")
    
    # Use index.html (not final_app.html)
    return render_template('final_app.html', 
                          field_names=field_names,
                          groups=groups,
                          subgroups=subgroups)


@app.route('/add_field', methods=['POST'])
def add_field():
    """Add a new field and calculate similarities"""
    # Load current data
    nested_data, similarities = load_data()
    
    # Get form data
    field_data = {
        "name": request.form.get('name'),
        "group": request.form.get('group'),
        "subgroup": request.form.get('subgroup'),
        "definition": request.form.get('definition'),
        "methodologies": request.form.get('methodologies'),
        "applications": request.form.get('applications'),
        "technologies": request.form.get('technologies'),
        "challenges": request.form.get('challenges'),
        "future_directions": request.form.get('future_directions')
    }
    
    # Validate required fields
    if not field_data["name"] or not field_data["group"] or not field_data["subgroup"]:
        return jsonify({"error": "Name, group, and subgroup are required"}), 400
    
    # Check if field name already exists
    field_names = get_all_field_names(nested_data)
    if field_data["name"] in field_names:
        return jsonify({"error": "Field name already exists"}), 400
    
    # Add field to data
    nested_data, new_field = add_field_to_data(nested_data, field_data)
    
    # Calculate new similarities
    updated_similarities = calculate_new_similarities(nested_data, similarities, new_field)
    
    # Save updated data
    if save_data(nested_data, updated_similarities):
        return jsonify({
            "success": True,
            "message": "Field added and similarities calculated",
            "download_ready": True
        })
    else:
        return jsonify({"error": "Error saving data"}), 500


@app.route('/get_subgroups', methods=['GET'])
def get_subgroups():
    """Get subgroups for a specific group"""
    group = request.args.get('group')
    
    if not group:
        return jsonify({"error": "Group parameter is required"}), 400
    
    nested_data, _ = load_data()
    _, subgroups = get_all_groups_and_subgroups(nested_data)
    
    return jsonify({
        "success": True,
        "subgroups": subgroups.get(group, [])
    })


@app.route('/get_similarity', methods=['GET'])
def get_similarity():
    """Get similarity between two fields"""
    field1 = request.args.get('field1')
    field2 = request.args.get('field2')
    
    if not field1 or not field2:
        return jsonify({"error": "Both field1 and field2 parameters are required"}), 400
    
    nested_data, similarities = load_data()
    
    # Get field data
    field1_data = get_field_data(nested_data, field1)
    field2_data = get_field_data(nested_data, field2)
    
    if not field1_data or not field2_data:
        return jsonify({"error": "One or both fields not found"}), 404
    
    # Find similarity
    similarity = find_similarity(similarities, field1, field2)
    
    if similarity is None:
        return jsonify({"error": "Similarity not found"}), 404
    
    return jsonify({
        "success": True,
        "field1": field1,
        "field2": field2,
        "similarity": similarity,
        "field1_data": field1_data,
        "field2_data": field2_data
    })


@app.route('/download_similarities')
def download_similarities():
    """Download similarities file"""
    return send_file(SIMILARITY_FILE, as_attachment=True)


@app.route('/test')
def test():
    """Test route to check if data is loading correctly"""
    nested_data, similarities = load_data()
    field_names = get_all_field_names(nested_data)
    groups, subgroups = get_all_groups_and_subgroups(nested_data)
    
    return jsonify({
        "success": True,
        "data_loaded": True,
        "field_count": len(field_names),
        "group_count": len(groups),
        "similarity_count": len(similarities),
        "fields": field_names[:5] if field_names else [],
        "groups": groups if groups else []
    })


if __name__ == '__main__':
    # Check if data directory exists and create it if not
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Run the application in debug mode
    app.run(debug=True)