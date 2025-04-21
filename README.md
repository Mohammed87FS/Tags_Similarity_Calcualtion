# Research Field Similarity Application

This application uses advanced natural language processing techniques to quantify and explore semantic relationships between different research fields based on their detailed descriptions. It enables researchers, administrators, and knowledge mappers to accurately analyze how different academic disciplines relate to one another through multiple similarity dimensions.

## Core Features

- **Multidimensional Field Comparisons**: Calculate similarity between research fields using a composite approach that incorporates semantic, lexical, and domain-specific measures
- **Hierarchical Research Taxonomy**: Organize fields in a structured hierarchy of groups and subgroups with relationship-aware similarity scoring
- **Domain-Specific Boosting**: Dynamically adjust similarity scores based on shared domain terminology
- **Faceted Field Descriptions**: Capture different aspects of research fields (definitions, methodologies, applications, technologies, challenges, and future directions)
- **Interactive Exploration**: Browse, compare, and visualize field similarities through a web interface
- **Extensible Architecture**: Modular design allowing new similarity measures or domain taxonomies to be incorporated

## Technical Architecture

The application follows a modular, service-oriented architecture:

```
python/
├── app.py                  # Application factory and entry point
├── config.py               # Centralized configuration parameters
├── models/                 # Domain models
│   ├── __init__.py
│   └── field.py            # Field entity with hierarchical relationships
├── data/                   # Data persistence
│   ├── nested_descriptions_research_groups.json    # Field hierarchies
│   └── final_outputs_enhanced_multi/field_similarities.json               # Calculated similarities
├── services/               # Core business logic
│   ├── __init__.py
│   ├── data_service.py     # Data access layer
│   └── similarity/         # Similarity calculation modules
│       ├── __init__.py
│       ├── embedding.py    # Neural embedding (Sentence Transformers)
│       ├── tfidf.py        # Term frequency-inverse document frequency
│       ├── domain.py       # Domain-specific terminology detection
│       └── field.py        # Composite similarity orchestration
├── routes/                 # API endpoints
│   ├── __init__.py
│   └── api.py              # RESTful interfaces
├── utils/                  # Shared utilities
│   ├── __init__.py
│   └── text_processing.py  # Text normalization and scaling functions
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   │   └── styles.css
│   └── js/                 # JavaScript files
│       └── app.js
└── templates/              # UI templates
    └── final_app.html      # Web interface
```

## Advanced Similarity Algorithm

The system employs a sophisticated multi-component similarity calculation approach:

### 1. Embedding-based Similarity (40% weight)
Uses the `all-mpnet-base-v2` sentence transformer model to compute embedding vectors for field descriptions and measures their cosine similarity. This captures deep semantic relationships even when different terminology is used.

### 2. TF-IDF Similarity (10% weight)
Employs n-gram (1-3) TF-IDF vectorization with customized term weighting to identify shared technical vocabulary, with special emphasis on:
- Technical patterns (CamelCase, hyphenated terms, acronyms)
- Multi-word technical terms
- Domain-specific terminology

### 3. Domain Concept Similarity (40% weight)
Identifies research domains present in each field by matching against specialized vocabulary across 8 technical domains:
- AI/ML
- Security
- Data Analytics
- HCI
- Graphics/Media
- Software Development
- Hardware Systems
- Healthcare

The system then computes similarity based on domain overlap, using a pre-defined domain relationship matrix to capture cross-domain relationships.

### 4. Faceted Comparison (10% weight)
Individual facets of field descriptions are compared separately with specific weights:
- Definition (35%)
- Methodologies (30%)
- Applications (15%)
- Technologies (10%)
- Challenges (5%)
- Future Directions (5%)

### 5. Group-based Calibration
The raw similarity score is calibrated based on hierarchical relationships:
- Fields in the same subgroup receive a baseline similarity of 0.75 plus weighted calculated similarity
- Fields in the same general group receive a baseline similarity of 0.7 plus weighted calculated similarity
- Cross-group similarities are capped at 0.7 to preserve taxonomic structure

### 6. Domain Boosting
Fields with significant domain overlap receive similarity boosts of up to 15%, especially when specialized terminology from the same domain is heavily present in both fields.

## Detailed Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/research-similarity.git
   cd research-similarity
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Create data directories:
   ```bash
   mkdir -p data/final_outputs_enhanced_multi
   ```

6. (Optional) Configure application parameters in `config.py` to adjust:
   - Similarity component weights
   - Group baseline similarities
   - Domain boosting factors
   - Domain terminology sets

## Usage Guide

### Starting the Application

Run the application server:
```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

### Web Interface

The web interface allows you to:
- View the hierarchical organization of research fields
- Add new research fields with detailed descriptions
- Compare any two fields and view their similarity score
- View all fields similar to a selected field, ranked by similarity
- Download the complete similarity matrix for external analysis

### Adding a New Field

1. Navigate to the "Add Field" section
2. Select a group and subgroup (or create new ones)
3. Provide a unique field name
4. Fill in detailed descriptions for each facet:
   - Definition: Core description of the field
   - Methodologies: Research approaches and methods
   - Applications: How the field is applied
   - Technologies: Tools and technologies used
   - Challenges: Current research challenges
   - Future Directions: Emerging research areas
5. Submit the form to add the field and calculate similarities

### Comparing Fields

1. Navigate to the "Compare Fields" section
2. Select two fields from the dropdown menus
3. View the calculated similarity score and field descriptions
4. Examine the facet-by-facet breakdown of similarity

## API Reference

The system provides a RESTful API for integration with other tools:

### Field Management
- `POST /api/add_field`: Add a new research field
  - Form data: name, group, subgroup, definition, methodologies, applications, technologies, challenges, future_directions
  - Returns: Success status and operation result

### Taxonomy Navigation
- `GET /api/get_subgroups?group={group_name}`: Get subgroups for a specific group
  - Returns: List of subgroup names

### Similarity Analysis
- `GET /api/get_similarity?field1={field1}&field2={field2}`: Get similarity between two fields
  - Returns: Similarity score and full field data
- `GET /api/get_all_similarities_for_field?field={field_name}`: Get all similarities for a field
  - Returns: Sorted list of all fields with similarity scores

### Data Export
- `GET /api/download_similarities`: Download complete similarity data in JSON format

### System Status
- `GET /api/test`: Check system status and data availability
  - Returns: Debug information and system status

## Configuration Reference

The `config.py` file contains all configurable parameters:

### File Paths
```python
DATA_DIR = 'data'
NESTED_DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'nested_descriptions_research_groups.json')
SIMILARITY_FILE = os.path.join(DATA_DIR, 'final_outputs_enhanced_multi/field_similarities.json')
```

### Similarity Parameters
```python
# Group-based similarity parameters
SAME_GROUP_BASELINE = 0.7        # Baseline for fields in same general group
SAME_SUBGROUP_BASELINE = 0.75    # Baseline for fields in same subgroup
SIMILARITY_WEIGHT_SUB = 0.2      # Weight for calculated similarity (subgroup)
SIMILARITY_WEIGHT_GENERAL = 0.15 # Weight for calculated similarity (general)
MAX_CROSS_GROUP_SIMILARITY = 0.7 # Max similarity for fields in different groups

# Description facet weights
DESCRIPTION_WEIGHTS = {
    "definition": 0.35,
    "methodologies": 0.30,
    "applications": 0.15,
    "technologies": 0.10,
      0.05,
    "future_directions": 0.05
}

# Similarity component weights
COMPONENT_WEIGHTS = {
    "embedding": 0.4,   # Neural embedding similarity
    "tfidf": 0.10,      # Term frequency similarity
    "domain": 0.4,      # Domain-specific similarity
    "facet": 0.10       # Faceted comparison
}

# Domain boosting configuration
ENABLE_DOMAIN_BOOSTING = True    # Enable domain-based score boosting
MAX_BOOST_FACTOR = 0.15          # Maximum boost to apply (up to 15%)
DOMAIN_BOOST_THRESHOLD = 0.7     # Min domain similarity to trigger boost
```

## Technical Requirements

- Python 3.8+
- Flask 2.2+
- Sentence Transformers 2.2+
- scikit-learn 1.2+
- spaCy 3.5+
- NumPy 1.24+

## Data Model

Research fields are stored hierarchically:
- Categories (high-level research domains)
  - Subgroups (specialized areas within domains)
    - Fields (specific research disciplines)

Each field contains:
- Name: Unique identifier
- Description: Multi-faceted description object
  - Definition
  - Methodologies
  - Applications
  - Technologies
  - Challenges
  - Future Directions

Similarities are stored as field pairs with calculated scores:
```json
{
  "field1": "Machine Learning",
  "field2": "Deep Learning",
  "similarity_score": 0.85
}
```

## Extension Points

The modular design allows for several extension points:

1. **New Similarity Components**: Add new similarity calculation methods in the `services/similarity/` directory and adjust weights in `config.py`

2. **Additional Domains**: Extend domain terminology in `config.py` by adding new domain groups

3. **Custom Facets**: Modify the `DESCRIPTION_WEIGHTS` in `config.py` to change or add description facets

4. **Alternative Embedding Models**: Replace the sentence transformer model in `embedding.py` with newer versions