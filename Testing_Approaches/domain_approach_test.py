import pandas as pd
import re
import logging
import spacy
from typing import Dict, List, Set


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
        'inference', 'prediction', 'algorithm', 'pytorch', 'tensorflow', 'keras', 'scikit-learn',
        'diffusion models', 'multimodal learning', 'zero-shot learning', 'few-shot learning', 
        'explainable ai', 'interpretable ml', 'neural architecture search', 'quantization',
        'knowledge distillation', 'model compression', 'adversarial training', 'contrastive learning',
        'representation learning', 'foundation models', 'large language models', 'stable diffusion',
        'neuro-symbolic ai', 'graph neural networks', 'transformers', 'self-attention', 'mixture of experts'
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
        'waf', 'antivirus', 'patch management', 'vulnerability scanning', 'penetration testing',
        'zero trust', 'security posture', 'supply chain security', 'devsecops', 'secure coding',
        'security by design', 'threat hunting', 'red team', 'blue team', 'purple team',
        'offensive security', 'defensive security', 'security compliance', 'gdpr', 'hipaa',
        'security orchestration', 'soar', 'security automation', 'threat emulation',
        'sandboxing', 'security analytics', 'secure boot', 'trusted execution', 'tee'
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
        'cross-validation', 'time series', 'anomaly detection', 'segmentation',
        'data observability', 'data lineage', 'data mesh', 'data fabric', 'data ops',
        'data democratization', 'feature store', 'data versioning', 'experiment tracking',
        'A/B testing', 'multivariate testing', 'cohort analysis', 'funnel analysis',
        'retention analysis', 'churn prediction', 'customer segmentation', 'recommender systems',
        'natural language analytics', 'text analytics', 'sentiment analysis', 'entity extraction'
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
        'participatory design', 'accessibility guidelines', 'wcag', 'inclusive design',
        'assistive technology', 'screen readers', 'voice recognition', 'haptic feedback',
        'tangible interfaces', 'spatial interfaces', 'embodied interaction', 'proxemics',
        'microinteractions', 'dark patterns', 'ethical design', 'service design',
        'experience design', 'customer experience', 'conversational interfaces', 'chatbots',
        'voice assistants', 'natural language interfaces', 'gesture controls'
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
        'opengl', 'directx', 'vulkan', 'unity', 'unreal engine', 'blender', 'maya',
        'real-time rendering', 'light field rendering', 'neural rendering', 'stylized rendering',
        'non-photorealistic rendering', 'photogrammetry', 'virtual production', 'digital twins',
        'procedural generation', 'computational geometry', 'spatial computing', 'mixed reality',
        'xr', 'extended reality', 'immersive media', 'holography', 'light field displays',
        'metaverse', 'digital human', 'performance capture', 'facial animation'
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
        'scalability', 'performance optimization', 'caching', 'load balancing',
        'serverless', 'event-driven architecture', 'domain-driven design', 'clean architecture',
        'hexagonal architecture', 'microservices', 'service mesh', 'api gateway',
        'orchestration', 'configuration management', 'feature flags', 'chaos engineering',
        'observability', 'logging', 'monitoring', 'tracing', 'debugging', 'profiling',
        'static analysis', 'dynamic analysis', 'code coverage', 'dependency management'
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
        'fault tolerance', 'redundancy', 'high availability', 'raid',
        'quantum computing', 'neuromorphic computing', 'optical computing',
        'silicon photonics', 'non-volatile memory', 'memristor', 'spintronics',
        'semiconductor', 'integrated circuit', 'analog circuit', 'digital circuit',
        'mixed-signal', 'signal processing', 'vlsi', 'chip design', 'system design',
        'hardware security', 'trusted platform module', 'secure element', 'hardware trojan'
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
        'health promotion', 'health education', 'health screening', 'vaccination',
        'remote patient monitoring', 'digital therapeutics', 'virtual care',
        'population health', 'value-based care', 'social determinants of health',
        'healthcare analytics', 'medical robotics', 'prosthetics', 'implantable devices',
        'wearable health tech', 'point-of-care diagnostics', 'drug discovery',
        'pharmacogenomics', 'immunotherapy', 'gene therapy', 'regenerative medicine'
    ],

    # NEW DOMAINS
    'quantum_computing': [
        'quantum computing', 'qubit', 'quantum gate', 'quantum circuit', 'quantum algorithm',
        'quantum supremacy', 'quantum advantage', 'quantum entanglement', 'quantum teleportation',
        'quantum cryptography', 'quantum key distribution', 'quantum annealing',
        'quantum error correction', 'quantum fault tolerance', 'quantum volume',
        'shor\'s algorithm', 'grover\'s algorithm', 'quantum fourier transform',
        'variational quantum eigensolver', 'quantum approximate optimization algorithm',
        'quantum machine learning', 'quantum neural network', 'quantum simulation',
        'quantum sensing', 'quantum metrology', 'quantum chemistry', 'quantum biology',
        'superconducting qubits', 'ion trap', 'topological qubits', 'photonic qubits',
        'neutral atom qubits', 'quantum dots', 'quantum processor', 'quantum computer',
        'quantum software', 'quantum programming', 'quantum development kit',
        'quantum compiler', 'quantum cloud', 'quantum supremacy', 'quantum coherence',
        'quantum decoherence', 'quantum noise', 'quantum error', 'post-quantum cryptography'
    ],

    'blockchain': [
        'blockchain', 'distributed ledger', 'cryptocurrency', 'bitcoin', 'ethereum',
        'smart contract', 'consensus algorithm', 'proof of work', 'proof of stake',
        'mining', 'hash function', 'merkle tree', 'public key cryptography',
        'private key', 'wallet', 'node', 'decentralized finance', 'defi',
        'non-fungible token', 'nft', 'tokenization', 'initial coin offering',
        'stablecoin', 'digital asset', 'decentralized autonomous organization',
        'dao', 'decentralized application', 'dapp', 'oracle', 'interoperability',
        'sidechain', 'layer 2', 'scaling solution', 'sharding', 'rollup',
        'zero-knowledge proof', 'zk-snark', 'digital identity', 'self-sovereign identity',
        'supply chain tracking', 'blockchain governance', 'token economics',
        'cryptoeconomics', 'permissioned blockchain', 'permissionless blockchain',
        'public blockchain', 'private blockchain', 'consortium blockchain'
    ],

    'bioinformatics': [
        'bioinformatics', 'computational biology', 'genomics', 'proteomics',
        'transcriptomics', 'metabolomics', 'sequence alignment', 'gene expression',
        'phylogenetics', 'phylogenomics', 'structural bioinformatics', 'systems biology',
        'biostatistics', 'genome assembly', 'gene prediction', 'protein structure prediction',
        'motif discovery', 'functional annotation', 'variant calling', 'genome-wide association',
        'gwas', 'gene ontology', 'pathway analysis', 'network biology', 'biomarker discovery',
        'single-cell analysis', 'epigenetics', 'metagenomics', 'dna sequencing',
        'rna sequencing', 'next-generation sequencing', 'variant annotation',
        'comparative genomics', 'evolutionary biology', 'protein-protein interaction',
        'molecular docking', 'molecular dynamics', 'biomedical text mining',
        'biological database', 'gene regulatory network', 'protein folding',
        'sequence homology', 'multiple sequence alignment', 'hidden markov model'
    ],

    'environmental_science': [
        'environmental science', 'ecology', 'sustainability', 'climate change',
        'global warming', 'carbon footprint', 'greenhouse gas', 'renewable energy',
        'solar energy', 'wind energy', 'hydro energy', 'geothermal energy',
        'biomass', 'biodiversity', 'conservation', 'ecosystem', 'habitat',
        'pollution', 'waste management', 'recycling', 'circular economy',
        'environmental impact', 'environmental assessment', 'life cycle assessment',
        'natural resources', 'water quality', 'air quality', 'soil quality',
        'deforestation', 'reforestation', 'carbon sequestration', 'carbon capture',
        'sustainable development', 'environmental policy', 'environmental regulation',
        'climate modeling', 'environmental monitoring', 'remote sensing',
        'geographic information system', 'gis', 'earth observation',
        'environmental remediation', 'green technology', 'clean technology',
        'resilience', 'adaptation', 'mitigation', 'environmental justice'
    ],

    'cloud_computing': [
        'cloud computing', 'infrastructure as a service', 'iaas', 'platform as a service',
        'paas', 'software as a service', 'saas', 'public cloud', 'private cloud',
        'hybrid cloud', 'multi-cloud', 'cloud native', 'cloud migration', 'serverless',
        'function as a service', 'faas', 'container', 'containerization', 'docker',
        'kubernetes', 'orchestration', 'auto-scaling', 'elasticity', 'virtualization',
        'virtual machine', 'hypervisor', 'cloud storage', 'object storage',
        'block storage', 'file storage', 'content delivery network', 'cdn',
        'edge computing', 'fog computing', 'service mesh', 'microservices',
        'service-oriented architecture', 'distributed systems', 'high availability',
        'fault tolerance', 'disaster recovery', 'cloud security', 'identity and access management',
        'iam', 'service level agreement', 'sla', 'cloud provider', 'aws', 'azure',
        'gcp', 'devops', 'gitops', 'infrastructure as code'
    ],

    'nlp': [
        'natural language processing', 'computational linguistics', 'text mining',
        'sentiment analysis', 'named entity recognition', 'part-of-speech tagging',
        'syntactic parsing', 'semantic parsing', 'coreference resolution',
        'word embeddings', 'word2vec', 'glove', 'fasttext', 'language model',
        'transformer', 'bert', 'gpt', 't5', 'llama', 'mistral', 'tokenization',
        'lemmatization', 'stemming', 'text classification', 'topic modeling',
        'document similarity', 'question answering', 'machine translation',
        'speech recognition', 'speech synthesis', 'text-to-speech', 'speech-to-text',
        'dialogue system', 'chatbot', 'information extraction', 'information retrieval',
        'text summarization', 'text generation', 'paraphrasing', 'grammar checking',
        'spell checking', 'language understanding', 'pragmatics', 'discourse analysis',
        'natural language understanding', 'natural language generation'
    ],

    'fintech': [
        'financial technology', 'fintech', 'digital banking', 'online banking',
        'mobile banking', 'digital payment', 'electronic payment', 'mobile payment',
        'peer-to-peer payment', 'cryptocurrency', 'digital currency', 'central bank digital currency',
        'cbdc', 'digital wallet', 'mobile wallet', 'payment processing', 'payment gateway',
        'payment card', 'credit card', 'debit card', 'prepaid card', 'online lending',
        'peer-to-peer lending', 'crowdfunding', 'robo-advisor', 'algorithmic trading',
        'high-frequency trading', 'quantitative finance', 'financial modeling',
        'risk management', 'fraud detection', 'anti-money laundering', 'aml',
        'know your customer', 'kyc', 'regtech', 'insurtech', 'proptech',
        'wealthtech', 'personal finance management', 'financial inclusion',
        'open banking', 'banking as a service', 'embedded finance',
        'buy now pay later', 'bnpl', 'distributed ledger technology', 'blockchain'
    ],

    'ar_vr': [
        'augmented reality', 'virtual reality', 'mixed reality', 'extended reality',
        'xr', 'ar', 'vr', 'mr', 'spatial computing', 'immersive technology',
        'head-mounted display', 'hmd', 'vr headset', 'ar glasses', 'smart glasses',
        'holographic display', 'light field display', 'volumetric display',
        '3d tracking', '6dof', 'inside-out tracking', 'outside-in tracking',
        'motion capture', 'hand tracking', 'eye tracking', 'gaze tracking',
        'gesture recognition', 'spatial mapping', 'depth sensing', 'lidar',
        'slam', 'simultaneous localization and mapping', 'virtual environment',
        'virtual world', 'metaverse', 'telepresence', 'teleportation', 'avatar',
        'digital human', 'virtual human', 'immersive analytics', 'spatial user interface',
        'spatial audio', '3d audio', 'binaural audio', 'haptic feedback',
        'force feedback', 'tactile feedback', 'motion sickness', 'simulator sickness'
    ],

    'robotics': [
        'robotics', 'robot', 'automation', 'autonomous systems', 'robotic system',
        'manipulator', 'robotic arm', 'robotic hand', 'end effector', 'gripper',
        'mobile robot', 'wheeled robot', 'legged robot', 'humanoid robot',
        'industrial robot', 'collaborative robot', 'cobot', 'service robot',
        'social robot', 'swarm robotics', 'micro-robotics', 'nano-robotics',
        'soft robotics', 'bio-inspired robotics', 'robot kinematics', 'inverse kinematics',
        'forward kinematics', 'robot dynamics', 'motion planning', 'path planning',
        'trajectory planning', 'collision avoidance', 'obstacle avoidance',
        'simultaneous localization and mapping', 'slam', 'robot vision',
        'robotic perception', 'sensor fusion', 'robot control', 'robotic manipulation',
        'grasping', 'robot learning', 'reinforcement learning for robotics',
        'imitation learning', 'human-robot interaction', 'robot operating system', 'ros'
    ]
}

# EXPANDED Domain group similarity matrix
DOMAIN_GROUP_SIMILARITY = {
    'ai_ml': {
        'ai_ml': 1.0, 
        'data_analytics': 0.85,
        'security': 0.35,
        'hci': 0.45,
        'graphics_media': 0.45,
        'software_development': 0.55, 
        'hardware_systems': 0.35, 
        'healthcare': 0.35,
        'quantum_computing': 0.65,
        'blockchain': 0.30,
        'bioinformatics': 0.50,
        'environmental_science': 0.25,
        'cloud_computing': 0.45,
        'nlp': 0.90,
        'fintech': 0.40,
        'ar_vr': 0.55,
        'robotics': 0.75
    },
    'security': {
        'ai_ml': 0.35, 
        'security': 1.0, 
        'data_analytics': 0.35, 
        'hci': 0.25, 
        'graphics_media': 0.15, 
        'software_development': 0.60,
        'hardware_systems': 0.45, 
        'healthcare': 0.35,
        'quantum_computing': 0.50,
        'blockchain': 0.80,
        'bioinformatics': 0.25,
        'environmental_science': 0.15,
        'cloud_computing': 0.70,
        'nlp': 0.30,
        'fintech': 0.75,
        'ar_vr': 0.20,
        'robotics': 0.35
    },
    'data_analytics': {
        'ai_ml': 0.85,
        'security': 0.35, 
        'data_analytics': 1.0, 
        'hci': 0.35, 
        'graphics_media': 0.30, 
        'software_development': 0.45, 
        'hardware_systems': 0.25, 
        'healthcare': 0.55,
        'quantum_computing': 0.30,
        'blockchain': 0.40,
        'bioinformatics': 0.65,
        'environmental_science': 0.60,
        'cloud_computing': 0.50,
        'nlp': 0.70,
        'fintech': 0.80,
        'ar_vr': 0.25,
        'robotics': 0.30
    },
    'hci': {
        'ai_ml': 0.45, 
        'security': 0.25, 
        'data_analytics': 0.35, 
        'hci': 1.0, 
        'graphics_media': 0.65,
        'software_development': 0.55, 
        'hardware_systems': 0.35, 
        'healthcare': 0.45,
        'quantum_computing': 0.15,
        'blockchain': 0.20,
        'bioinformatics': 0.25,
        'environmental_science': 0.30,
        'cloud_computing': 0.30,
        'nlp': 0.60,
        'fintech': 0.35,
        'ar_vr': 0.85,
        'robotics': 0.55
    },
    'graphics_media': {
        'ai_ml': 0.45, 
        'security': 0.15, 
        'data_analytics': 0.30, 
        'hci': 0.65,
        'graphics_media': 1.0, 
        'software_development': 0.35, 
        'hardware_systems': 0.35, 
        'healthcare': 0.20,
        'quantum_computing': 0.15,
        'blockchain': 0.10,
        'bioinformatics': 0.15,
        'environmental_science': 0.25,
        'cloud_computing': 0.20,
        'nlp': 0.30,
        'fintech': 0.15,
        'ar_vr': 0.90,
        'robotics': 0.25
    },
    'software_development': {
        'ai_ml': 0.55, 
        'security': 0.60,
        'data_analytics': 0.45, 
        'hci': 0.55, 
        'graphics_media': 0.35, 
        'software_development': 1.0, 
        'hardware_systems': 0.65, 
        'healthcare': 0.30,
        'quantum_computing': 0.45,
        'blockchain': 0.55,
        'bioinformatics': 0.35,
        'environmental_science': 0.25,
        'cloud_computing': 0.85,
        'nlp': 0.50,
        'fintech': 0.55,
        'ar_vr': 0.45,
        'robotics': 0.60
    },
    'hardware_systems': {
        'ai_ml': 0.35, 
        'security': 0.45, 
        'data_analytics': 0.25, 
        'hci': 0.35, 
        'graphics_media': 0.35, 
        'software_development': 0.65, 
        'hardware_systems': 1.0, 
        'healthcare': 0.35,
        'quantum_computing': 0.75,
        'blockchain': 0.30,
        'bioinformatics': 0.20,
        'environmental_science': 0.35,
        'cloud_computing': 0.60,
        'nlp': 0.15,
        'fintech': 0.25,
        'ar_vr': 0.50,
        'robotics': 0.85
    },
    'healthcare': {
        'ai_ml': 0.35, 
        'security': 0.35, 
        'data_analytics': 0.55, 
        'hci': 0.45, 
        'graphics_media': 0.20, 
        'software_development': 0.30, 
        'hardware_systems': 0.35, 
        'healthcare': 1.0,
        'quantum_computing': 0.15,
        'blockchain': 0.25,
        'bioinformatics': 0.85,
        'environmental_science': 0.40,
        'cloud_computing': 0.25,
        'nlp': 0.45,
        'fintech': 0.30,
        'ar_vr': 0.40,
        'robotics': 0.45
    },
    'quantum_computing': {
        'ai_ml': 0.65,
        'security': 0.50,
        'data_analytics': 0.30,
        'hci': 0.15,
        'graphics_media': 0.15,
        'software_development': 0.45,
        'hardware_systems': 0.75,
        'healthcare': 0.15,
        'quantum_computing': 1.0,
        'blockchain': 0.35,
        'bioinformatics': 0.30,
        'environmental_science': 0.20,
        'cloud_computing': 0.40,
        'nlp': 0.25,
        'fintech': 0.25,
        'ar_vr': 0.10,
        'robotics': 0.35
    },
    'blockchain': {
        'ai_ml': 0.30,
        'security': 0.80,
        'data_analytics': 0.40,
        'hci': 0.20,
        'graphics_media': 0.10,
        'software_development': 0.55,
        'hardware_systems': 0.30,
        'healthcare': 0.25,
        'quantum_computing': 0.35,
        'blockchain': 1.0,
        'bioinformatics': 0.15,
        'environmental_science': 0.35,
        'cloud_computing': 0.55,
        'nlp': 0.20,
        'fintech': 0.90,
        'ar_vr': 0.25,
        'robotics': 0.15
    },
    'bioinformatics': {
        'ai_ml': 0.50,
        'security': 0.25,
        'data_analytics': 0.65,
        'hci': 0.25,
        'graphics_media': 0.15,
        'software_development': 0.35,
        'hardware_systems': 0.20,
        'healthcare': 0.85,
        'quantum_computing': 0.30,
        'blockchain': 0.15,
        'bioinformatics': 1.0,
        'environmental_science': 0.55,
        'cloud_computing': 0.25,
        'nlp': 0.45,
        'fintech': 0.15,
        'ar_vr': 0.10,
        'robotics': 0.25
    },
    'environmental_science': {
        'ai_ml': 0.25,
        'security': 0.15,
        'data_analytics': 0.60,
        'hci': 0.30,
        'graphics_media': 0.25,
        'software_development': 0.25,
        'hardware_systems': 0.35,
        'healthcare': 0.40,
        'quantum_computing': 0.20,
        'blockchain': 0.35,
        'bioinformatics': 0.55,
        'environmental_science': 1.0,
        'cloud_computing': 0.30,
        'nlp': 0.25,
        'fintech': 0.25,
        'ar_vr': 0.35,
        'robotics': 0.45
    },
    'cloud_computing': {
        'ai_ml': 0.45,
        'security': 0.70,
        'data_analytics': 0.50,
        'hci': 0.30,
        'graphics_media': 0.20,
        'software_development': 0.85,
        'hardware_systems': 0.60,
        'healthcare': 0.25,
        'quantum_computing': 0.40,
        'blockchain': 0.55,
        'bioinformatics': 0.25,
        'environmental_science': 0.30,
        'cloud_computing': 1.0,
        'nlp': 0.35,
        'fintech': 0.55,
        'ar_vr': 0.30,
        'robotics': 0.40
    },
    'nlp': {
        'ai_ml': 0.90,
        'security': 0.30,
        'data_analytics': 0.70,
        'hci': 0.60,
        'graphics_media': 0.30,
        'software_development': 0.50,
        'hardware_systems': 0.15,
        'healthcare': 0.45,
        'quantum_computing': 0.25,
        'blockchain': 0.20,
        'bioinformatics': 0.45,
        'environmental_science': 0.25,
        'cloud_computing': 0.35,
        'nlp': 1.0,
        'fintech': 0.25,
        'ar_vr': 0.35,
        'robotics': 0.25
    },
    'fintech': {
        'ai_ml': 0.40,
        'security': 0.75,
        'data_analytics': 0.80,
        'hci': 0.35,
        'graphics_media': 0.15,
        'software_development': 0.55,
        'hardware_systems': 0.25,
        'healthcare': 0.30,
        'quantum_computing': 0.25,
        'blockchain': 0.90,
        'bioinformatics': 0.15,
        'environmental_science': 0.25,
        'cloud_computing': 0.55,
        'nlp': 0.25,
        'fintech': 1.0,
        'ar_vr': 0.20,
        'robotics': 0.15
    },
    'ar_vr': {
        'ai_ml': 0.55,
        'security': 0.20,
        'data_analytics': 0.25,
        'hci': 0.85,
        'graphics_media': 0.90,
        'software_development': 0.45,
        'hardware_systems': 0.50,
        'healthcare': 0.40,
        'quantum_computing': 0.10,
        'blockchain': 0.25,
        'bioinformatics': 0.10,
        'environmental_science': 0.35,
        'cloud_computing': 0.30,
        'nlp': 0.35,
        'fintech': 0.20,
        'ar_vr': 1.0,
        'robotics': 0.45
    },
    'robotics': {
        'ai_ml': 0.75,
        'security': 0.35,
        'data_analytics': 0.30,
        'hci': 0.55,
        'graphics_media': 0.25,
        'software_development': 0.60,
        'hardware_systems': 0.85,
        'healthcare': 0.45,
        'quantum_computing': 0.35,
        'blockchain': 0.15,
        'bioinformatics': 0.25,
        'environmental_science': 0.45,
        'cloud_computing': 0.40,
        'nlp': 0.25,
        'fintech': 0.15,
        'ar_vr': 0.45,
        'robotics': 1.0
    }
}

# DomainService implementation (simplified for testing)
class DomainService:
    def __init__(self):
        self.domain_concept_cache = {}
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            self.use_spacy = False
            self.nlp = None
        
        self.technical_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)+\b',
            r'\b[a-z]+(?:-[a-z]+)+\b',
            r'\b[A-Za-z]+\d+[A-Za-z]*\b',
            r'\b[A-Za-z]+\.[A-Za-z]+\b',
            r'\b[A-Z][A-Z0-9]+\b',
        ]
        
        self.domain_term_lookup = {}
        for domain, terms in DOMAIN_TERM_GROUPS.items():
            for term in terms:
                self.domain_term_lookup[term] = domain
    
    def extract_domain_concepts(self, text: str) -> Dict[str, List[str]]:
        if not text.strip():
            return {domain: [] for domain in DOMAIN_TERM_GROUPS}
        if text in self.domain_concept_cache:
            return self.domain_concept_cache[text]
        
        text_lower = text.lower()
        domain_concepts = {domain: [] for domain in DOMAIN_TERM_GROUPS}
        domain_concepts['general'] = []
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
            domain_concepts['general'].extend(noun_phrases)
            for token in doc:
                if token.pos_ == "NOUN" and token.dep_ == "compound":
                    compound_term = f"{token.text} {token.head.text}".lower()
                    domain_concepts['general'].append(compound_term)
                if token.pos_ == "NOUN" and any(mod.pos_ == "ADJ" for mod in token.children):
                    adj_mods = [mod.text for mod in token.children if mod.pos_ == "ADJ"]
                    tech_term = f"{' '.join(adj_mods)} {token.text}".lower()
                    domain_concepts['general'].append(tech_term)
        
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text)
            domain_concepts['general'].extend([m.lower() for m in matches])
        
        for term, domain in self.domain_term_lookup.items():
            if term in text_lower:
                domain_concepts[domain].append(term)
        
        self.domain_concept_cache[text] = domain_concepts
        return domain_concepts
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        d1 = self.extract_domain_concepts(text1)
        d2 = self.extract_domain_concepts(text2)
        sims, weights = [], []
        for dom1, c1 in d1.items():
            if dom1 == 'general' or not c1: continue
            for dom2, c2 in d2.items():
                if dom2 == 'general' or not c2: continue
                sim = DOMAIN_GROUP_SIMILARITY.get(dom1, {}).get(dom2, 0.1)
                w = len(c1) * len(c2)
                if dom1 == dom2 and len(c1) >= 2 and len(c2) >= 2:
                    w *= 2
                sims.append(sim)
                weights.append(w)
        # general
        g1, g2 = set(d1['general']), set(d2['general'])
        if g1 and g2:
            jacc = len(g1 & g2) / len(g1 | g2)
            w = min(20, len(g1) + len(g2))
            sims.append(jacc)
            weights.append(w)
        if sims and sum(weights) > 0:
            return sum(s * w for s, w in zip(sims, weights)) / sum(weights)
        return 0.0
    
    def detect_primary_domains(self, text: str) -> List[str]:
        d = self.extract_domain_concepts(text)
        counts = {dom: len(concepts) for dom, concepts in d.items() if dom != 'general'}
        if not counts:
            return []
        max_count = max(counts.values())
        if max_count == 0:
            return []
        return [dom for dom, cnt in counts.items() if cnt >= max_count * 0.5]

# Instantiate service
service = DomainService()


texts = [
    # AI/ML and Data Science (same domain)
    ("Deep learning models like transformers have revolutionized natural language processing through self-attention mechanisms, allowing large language models to generate coherent text and achieve state-of-the-art results on benchmarks like GLUE and SuperGLUE. Modern architectures employ techniques like transfer learning, fine-tuning, and few-shot learning to adapt pre-trained foundation models to downstream tasks with minimal data.",
     "Statistical machine learning approaches leverage neural networks with sophisticated architectures to process natural language. These systems employ transformer-based models with multi-head attention mechanisms to understand context and semantic relationships in text. Pre-training on large corpora followed by task-specific fine-tuning has become the dominant paradigm for NLP systems."),
    
    # Cybersecurity vs. Blockchain (related domains)
    ("Modern cybersecurity frameworks implement zero-trust architecture with multi-factor authentication, encryption at rest and in transit, and continuous threat monitoring. Security operations centers employ SIEM systems for log analysis and automated incident response, while penetration testing identifies vulnerabilities before they can be exploited. Secure coding practices and threat modeling are integrated throughout the development lifecycle.",
     "Distributed ledger technologies utilize consensus algorithms and cryptographic techniques to ensure data integrity across decentralized networks. Blockchain implementations leverage public key infrastructure, hash functions, and digital signatures to authenticate transactions and maintain immutable records. Smart contracts execute automatically when predefined conditions are met, enabling trust between parties without centralized authorities."),
    
    # Healthcare vs. Robotics (different domains)
    ("Precision medicine integrates genomic sequencing, clinical data, and electronic health records to develop personalized treatment plans tailored to individual patients. Healthcare informatics systems support clinical decision making through evidence-based protocols and predictive analytics. Remote patient monitoring through wearable devices enables continuous assessment of vital signs and early intervention for chronic conditions.",
     "Industrial robot deployment requires precise motion planning algorithms, collision avoidance systems, and real-time control loops to coordinate movements in manufacturing environments. End effectors are customized for specific tasks with specialized grippers, tools, and sensors that provide haptic feedback. Kinematics and dynamics calculations optimize robot arm trajectories while minimizing energy consumption and mechanical wear."),
    
    # Software Development vs. Environmental Science (very different domains)
    ("Modern software engineering practices emphasize continuous integration and continuous delivery pipelines with automated testing and deployment. Microservice architectures decompose applications into independently deployable components that communicate through well-defined APIs. DevOps teams implement infrastructure as code using configuration management tools and container orchestration platforms to ensure consistency across environments.",
     "Conservation biology research focuses on ecosystem resilience, biodiversity preservation, and habitat restoration in the face of climate change pressures. Field researchers track endangered species populations using non-invasive monitoring techniques and genetic sampling. Environmental policy frameworks balance scientific evidence with stakeholder interests to establish protected areas and sustainable resource management practices."),
    
    # Finance vs. Agriculture (very different domains)
    ("Algorithmic trading systems execute complex investment strategies through high-frequency market operations governed by mathematical models and statistical arbitrage. Portfolio managers optimize asset allocation based on modern portfolio theory, risk-adjusted returns, and macroeconomic indicators. Regulatory compliance frameworks ensure transparency in transactions while preventing market manipulation through sophisticated monitoring systems.",
     "Regenerative agriculture practices enhance soil health through cover cropping, reduced tillage, and rotational grazing systems that build organic matter and sequester carbon. Precision irrigation technologies deliver water directly to plant root zones based on soil moisture sensors and evapotranspiration calculations. Integrated pest management reduces chemical interventions through biological controls and habitat diversification strategies."),
    
    # Virtual Reality vs. Graphics (same domain)
    ("Immersive virtual reality environments integrate spatial computing techniques with real-time rendering to create interactive 3D experiences. Modern VR systems employ ray tracing, physically-based rendering, and global illumination to achieve photorealistic visuals. Head-mounted displays with 6DOF tracking enable users to navigate virtual spaces naturally, while haptic feedback provides tactile sensations for enhanced presence.",
     "Computer graphics rendering techniques utilize graphics processing units for real-time visualization of complex 3D models. Shading algorithms simulate light-material interactions through physically-based rendering and global illumination. Game engines provide frameworks for scene composition, animation systems, and particle effects that create immersive digital environments for interactive media and simulations."),
    
    # Quantum Computing vs. Language Learning (very different domains)
    ("Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations impossible for classical systems. Quantum algorithms like Shor's for factorization and Grover's for search offer exponential or quadratic speedups for specific problems. Error correction techniques mitigate the effects of quantum decoherence, while different qubit implementations compete for scalability.",
     "Second language acquisition theories emphasize comprehensible input, motivation factors, and sociocultural contexts in developing language proficiency. Educational linguists study the role of explicit grammar instruction versus immersive communicative approaches in classroom environments. Bilingual education programs balance content knowledge with language development through scaffolded instruction and strategic first-language support."),
    
    # Human-Computer Interaction vs. Maritime Engineering (very different domains)
    ("User experience research employs cognitive walkthroughs, heuristic evaluations, and usability testing to optimize interface designs for human perception and cognition. Design systems establish consistent interaction patterns and component libraries that ensure coherence across digital products. Accessibility guidelines ensure that interfaces accommodate diverse user needs, including screen readers and keyboard navigation.",
     "Naval architecture principles guide the design of ship hulls through computational fluid dynamics and structural analysis to ensure seaworthiness in extreme conditions. Marine engineers develop propulsion systems that optimize fuel efficiency while meeting emissions standards through advanced combustion techniques and exhaust treatment. Offshore platform construction requires specialized materials and structural configurations to withstand harsh ocean environments."),
    
    # Cloud Computing vs. Physical Therapy (very different domains)
    ("Cloud-native applications leverage containerization technology, orchestration platforms like Kubernetes, and serverless functions to achieve scalability and resilience. Multi-cloud strategies distribute workloads across providers to optimize for cost, performance, and reliability. Infrastructure as code enables reproducible deployments, while service meshes handle inter-service communication, security, and observability.",
     "Physical rehabilitation protocols integrate biomechanical analysis, musculoskeletal assessment, and functional movement screening to develop personalized treatment plans. Manual therapy techniques target specific soft tissue restrictions and joint mobilizations to restore normal movement patterns. Exercise prescription progressively challenges patients through appropriate loading parameters, motor control drills, and neuromuscular re-education strategies."),
    
    # Network Infrastructure vs. Fashion Design (very different domains)
    ("Next-generation network architecture employs software-defined networking to decouple control and data planes, enabling programmable network management. 5G infrastructure combines massive MIMO antenna arrays, millimeter wave spectrum, and network slicing to support diverse application requirements from IoT to autonomous vehicles. Edge computing nodes process data closer to sources, reducing latency for time-sensitive applications.",
     "Haute couture design processes begin with concept development through mood boards and material exploration, followed by pattern drafting and garment construction using specialized techniques. Fashion forecasting analyzes cultural trends, consumer behavior, and historical cycles to predict upcoming style directions. Sustainable fashion practices emphasize ethical sourcing, zero-waste pattern cutting, and circular design principles to reduce environmental impact."),
    
    # AI vs. Construction Management (different domains)
    ("Reinforcement learning algorithms optimize agent behavior through trial-and-error interactions with complex environments, balancing exploration of new strategies against exploitation of known effective actions. Neural architecture search automates the discovery of optimal network structures through evolutionary algorithms and gradient-based methods. Foundation models trained on diverse datasets exhibit emergent capabilities not explicitly programmed into their training objectives.",
     "Construction project management integrates scheduling techniques like critical path method and resource leveling to optimize workflow and manage dependencies between trades. Building information modeling enables clash detection, quantity takeoffs, and virtual coordination between structural, mechanical, and electrical systems. Lean construction principles minimize waste through just-in-time material delivery, prefabrication strategies, and continuous improvement processes."),
    
    # Cybersecurity vs. Culinary Arts (very different domains)
    ("Advanced persistent threats employ multi-stage attack vectors that combine social engineering, zero-day exploits, and lateral movement techniques to establish long-term unauthorized access to high-value systems. Threat hunting operations analyze network telemetry, system logs, and user behavior patterns to proactively identify anomalous activity indicative of compromise. Security orchestration automates incident response through playbooks that coordinate containment, eradication, and recovery actions.",
     "Molecular gastronomy applies scientific principles to culinary techniques, investigating the physical and chemical transformations of ingredients during cooking processes. Restaurant kitchens maintain strict food safety protocols through HACCP systems, temperature monitoring, and cross-contamination prevention measures. Flavor development relies on understanding taste receptors, aroma compounds, and mouthfeel characteristics to create balanced and memorable dining experiences."),
    
    # Healthcare vs. Aviation (different domains)
    ("Electronic health record systems integrate clinical documentation, computerized physician order entry, and decision support tools to improve care coordination across medical specialties. Population health management identifies high-risk patient cohorts through predictive analytics and facilitates targeted interventions. Telemedicine platforms enable remote consultations through secure video conferencing, digital health monitoring, and store-and-forward diagnostic capabilities.",
     "Aviation safety management systems incorporate risk assessment frameworks, voluntary reporting mechanisms, and just culture principles to continuously improve operational procedures. Air traffic control coordinates aircraft movements through radar tracking, procedural separation techniques, and communication protocols that maintain safe distances between flights. Aeronautical engineering designs aircraft systems with redundancy, fail-safe mechanisms, and comprehensive maintenance requirements to ensure airworthiness."),
    
    # IoT vs. Urban Planning (somewhat related domains)
    ("Internet of Things deployments utilize low-power wide-area networks and mesh topologies to connect distributed sensors across urban environments. Edge computing gateways preprocess sensor data to reduce bandwidth requirements and enable real-time analytics at the network edge. Device management platforms handle firmware updates, security patching, and authentication across heterogeneous IoT ecosystems with diverse hardware capabilities.",
     "Smart city initiatives integrate transportation systems, utility infrastructure, and public services through data-driven planning and real-time monitoring technologies. Urban designers create walkable neighborhoods through mixed-use zoning, transit-oriented development, and human-scale streetscapes. Public space activation strategies incorporate placemaking principles, flexible infrastructure, and community programming to foster social cohesion and economic vitality."),
    
    # Blockchain vs. Music Production (very different domains)
    ("Decentralized finance applications implement smart contracts that execute lending protocols, automated market makers, and yield optimization strategies without traditional intermediaries. Consensus mechanisms like proof-of-stake secure blockchain networks while reducing energy consumption compared to proof-of-work systems. Non-fungible token standards enable provable digital ownership of unique assets through cryptographic verification on public ledgers.",
     "Music production workflows combine tracking, editing, mixing, and mastering stages to transform raw recordings into polished audio products. Digital audio workstations provide virtual instruments, effect processors, and automation tools that enable precise control over sonic characteristics. Spatial audio techniques manipulate binaural cues, ambience, and frequency response to create immersive soundscapes that translate across different listening environments."),
    
    # Robotics vs. Legal Practice (very different domains)
    ("Autonomous mobile robots navigate dynamic environments through simultaneous localization and mapping algorithms that build and update spatial representations in real-time. Computer vision systems enable object recognition, pose estimation, and scene understanding through convolutional neural networks trained on diverse datasets. Collaborative robots incorporate force-feedback sensors, compliant motion controllers, and safety-rated monitored stops to work alongside human operators.",
     "Constitutional law interpretation balances textualist approaches, historical context, and precedent to resolve complex legal questions about governmental powers and individual rights. Civil procedure governs the rules of litigation through carefully structured processes for discovery, evidence presentation, and judicial review. Alternative dispute resolution mechanisms like mediation and arbitration offer parties more control over outcomes than traditional court proceedings."),
    
    # Data Science vs. Literature (very different domains)
    ("Data preprocessing pipelines handle missing values, outlier detection, and feature engineering to prepare datasets for machine learning algorithms. Exploratory data analysis uncovers patterns through statistical summaries, correlation matrices, and visualization techniques that reveal underlying structures. Model evaluation frameworks assess predictive performance through cross-validation, confusion matrices, and learning curves that detect overfitting and underfitting.",
     "Literary criticism examines narrative structure, character development, and thematic elements through close readings informed by theoretical frameworks like post-structuralism and feminist theory. Comparative literature studies trace influences across national traditions, historical periods, and linguistic boundaries to identify cultural exchange patterns. Creative writing workshops employ peer critique, revision processes, and craft analysis to develop authentic authorial voices."),
    
    # Hardware Systems vs. Economics (very different domains)
    ("System-on-chip designs integrate processors, memory controllers, and specialized accelerators onto single silicon dies that optimize performance per watt for specific workloads. Computer architecture innovations like out-of-order execution, branch prediction, and speculative execution extract instruction-level parallelism from sequential code. Memory hierarchy implementations balance capacity, latency, and bandwidth through multi-level caches, prefetching algorithms, and non-uniform access architectures.",
     "Macroeconomic policy frameworks balance monetary and fiscal tools to maintain price stability, employment targets, and sustainable growth trajectories. Behavioral economics research challenges rational actor assumptions by documenting systematic cognitive biases in decision-making under uncertainty. International trade theories explain specialization patterns through comparative advantage principles while accounting for increasing returns to scale and imperfect competition dynamics."),
    
    # Energy Systems vs. Psychology (very different domains)
    ("Grid modernization initiatives implement advanced metering infrastructure, distribution automation, and wide-area monitoring systems to enhance reliability and enable bidirectional power flows. Renewable integration strategies address intermittency through diverse generation portfolios, geographic dispersion, and flexible resources that match supply with demand. Energy storage technologies from pumped hydro to battery systems provide multiple grid services including peak shaving, frequency regulation, and resilience during outages.",
     "Cognitive behavioral therapy techniques identify and modify maladaptive thought patterns that contribute to emotional distress and behavioral problems. Developmental psychology studies how genetic factors interact with environmental influences to shape personality formation across the lifespan. Neuropsychological assessment evaluates cognitive functions including attention, memory, executive functioning, and language processing to diagnose conditions and inform treatment planning.")
]

# Prepare results
results = []
for t1, t2 in texts:
    concepts1 = service.extract_domain_concepts(t1)
    concepts2 = service.extract_domain_concepts(t2)
    sim = service.calculate_similarity(t1, t2)
    domains = service.detect_primary_domains(t1)
    results.append({
        'Text1': t1,
        'Text2': t2,
        'Extracted Concepts Text1': concepts1,
        'Extracted Concepts Text2': concepts2,
        'Similarity': sim,
        'Primary Domains for Text1': domains
    })


# Display results
df = pd.DataFrame(results)

# At the bottom of your domain.py file, replace the simple DataFrame display with:

# Display results with improved formatting
def display_results(results):
    print("\n===== DOMAIN ANALYSIS RESULTS =====\n")
    
    for i, result in enumerate(results, 1):
        print(f"📊 COMPARISON #{i}")
        print(f"{'='*50}")
        
        # Show texts
        print(f"📝 TEXT 1: {result['Text1']}")
        print(f"📝 TEXT 2: {result['Text2']}")
        print()
        
        # Show similarity score with visual representation
        sim = result['Similarity']
        sim_percentage = int(sim * 100)
        bar_length = int(sim * 20)
        bar = "■" * bar_length + "□" * (20 - bar_length)
        print(f"🔍 SIMILARITY SCORE: {sim:.2f} ({sim_percentage}%)")
        print(f"    {bar} {sim_percentage}%")
        print()
        
        # Show extracted concepts
        print("📋 EXTRACTED CONCEPTS:")
        print("  TEXT 1:")
        for domain, concepts in result['Extracted Concepts Text1'].items():
            if concepts:
                print(f"    • {domain.upper()}: {', '.join(concepts)}")
        
        print("  TEXT 2:")
        for domain, concepts in result['Extracted Concepts Text2'].items():
            if concepts:
                print(f"    • {domain.upper()}: {', '.join(concepts)}")
        print()
        
        # Show primary domains
        print(f"🏷️  PRIMARY DOMAINS: {', '.join(result['Primary Domains for Text1']) or 'None'}")
        print("\n" + "-"*50 + "\n")

# Execute the display function
display_results(results)