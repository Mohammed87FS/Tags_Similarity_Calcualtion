"""
Configuration settings for the research field similarity application.
"""

import os

# Data paths
DATA_DIR = 'data'
NESTED_DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'nested_descriptions_research_groups.json')
SIMILARITY_FILE = os.path.join(DATA_DIR, 'final_outputs_enhanced_multi/field_similarities.json')

# Group-based similarity parameters
SAME_GROUP_BASELINE = 0.7        # Baseline similarity for fields in same general group
SAME_SUBGROUP_BASELINE = 0.75    # Baseline similarity for fields in same subgroup
SIMILARITY_WEIGHT_SUB = 0.2      # Weight of calculated similarity to add to baseline for subgroup
SIMILARITY_WEIGHT_GENERAL = 0.15 # Weight of calculated similarity to add to baseline for general group
MAX_CROSS_GROUP_SIMILARITY = 0.7 # Maximum similarity for fields not in same group/subgroup

# Field description property weights
DESCRIPTION_WEIGHTS = {
    "definition": 0.6,
    "methodologies": 0.20,
    "applications": 0.20,
 

}

# Component weights
COMPONENT_WEIGHTS = {
    "embedding": 0.4,
    "tfidf": 0.20,
    "domain": 0.3,
    "facet": 0.10
}

# Domain boosting configuration
ENABLE_DOMAIN_BOOSTING = True    # Whether to apply domain-based score boosting
MAX_BOOST_FACTOR = 0.15          # Maximum boost to apply (0.15 = up to 15% boost)
DOMAIN_BOOST_THRESHOLD = 0.7     # Minimum domain similarity to trigger boost

# Domain-specific term groups
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

# Domain group similarity matrix
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