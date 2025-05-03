# Research Field Similarity Application

This application uses advanced natural language processing techniques to quantify and explore semantic relationships between different research fields based on their detailed descriptions. It enables researchers, administrators, and knowledge mappers to accurately analyze how different academic disciplines relate to one another through multiple similarity dimensions.

## Table of Contents

- [Core Features](#core-features)
- [Technical Architecture](#technical-architecture)
- [Advanced Similarity Algorithm](#advanced-similarity-algorithm)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Configuration Guide](#configuration-guide)
- [Technical Requirements](#technical-requirements)
- [Data Model](#data-model)
- [Extending the Application](#extending-the-application)

## Core Features

- **Multidimensional Field Comparisons**: Calculate similarity between research fields using a composite approach that incorporates semantic, lexical, and domain-specific measures
- **Hierarchical Research Taxonomy**: Organize fields in a structured hierarchy of groups and subgroups with relationship-aware similarity scoring
- **Domain-Specific Boosting**: Dynamically adjust similarity scores based on shared domain terminology and conceptual overlap
- **Faceted Field Descriptions**: Capture different aspects of research fields (definitions, methodologies, applications) for nuanced comparison
- **Interactive Exploration**: Browse, compare, and visualize field similarities through an intuitive web interface with dynamic visualizations
- **Extensible Architecture**: Modular design allowing new similarity measures or domain taxonomies to be incorporated with minimal code changes

## Technical Architecture

The application follows a modular, service-oriented architecture designed for maintainability and extensibility:

```
python/
├── app.py                  # Application factory and entry point
├── config.py               # Centralized configuration parameters
├── data/                   # Data persistence
│   ├── nested_descriptions_research_groups.json    # Field hierarchies
│   └── final_outputs_enhanced_multi/               # Output directory
│       └── field_similarities.json                 # Calculated similarities
├── services/               # Core business logic
│   ├── __init__.py
│   ├── data_service.py     # Data access layer
│   └── similarity/         # Similarity calculation modules
│       ├── __init__.py
│       ├── embedding.py    # Neural embedding (Sentence Transformers)
│       ├── domain.py       # Domain-specific terminology detection
│       └── final_calculation.py  # Composite similarity orchestration
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
│       └── app.js          # Frontend application logic
└── templates/              # UI templates
    └── final_app.html      # Web interface
```

### Key Components

- **Data Service**: Handles loading and saving field and similarity data to JSON files
- **Embedding Service**: Computes semantic similarity using state-of-the-art language models
- **Domain Service**: Detects domain-specific terminology and calculates domain-based similarity
- **Field Similarity Service**: Orchestrates the overall similarity calculation process
- **API Routes**: Expose functionality through a RESTful interface
- **Frontend Application**: Provides an interactive user interface for exploring similarities

## Advanced Similarity Algorithm

The system employs a sophisticated multi-component similarity calculation approach that combines multiple NLP techniques:

### 1. Embedding-based Similarity (40% weight)
Uses the `all-mpnet-base-v2` sentence transformer model to compute embedding vectors for field descriptions and measures their cosine similarity. This captures deep semantic relationships even when different terminology is used to describe similar concepts.

### 2. Domain Concept Similarity (40% weight)
Identifies research domains present in each field by matching against specialized vocabulary across 17 technical domains:
- AI/ML
- Security
- Data Analytics
- HCI
- Graphics/Media
- Software Development
- Hardware Systems
- Healthcare
- Quantum Computing
- Blockchain
- Bioinformatics
- Environmental Science
- Cloud Computing
- NLP
- Fintech
- AR/VR
- Robotics

The system then computes similarity based on domain overlap, using a pre-defined domain relationship matrix to capture cross-domain relationships. This allows the system to understand that certain domains (like AI/ML and Data Science) are inherently more related than others.

### 3. Faceted Comparison (20% weight)
Individual facets of field descriptions are compared separately with specific weights:
- Definition (60%): Core description of the field
- Methodologies (20%): Research approaches and techniques
- Applications (20%): Real-world use cases and implementations

This multi-faceted approach ensures that fields with similar applications but different methods (or vice versa) receive appropriate similarity scores.

### 4. Group-based Calibration
The raw similarity score is calibrated based on hierarchical relationships:
- Fields in the same subgroup receive a baseline similarity of 0.75 plus weighted calculated similarity
- Fields in the same general group receive a baseline similarity of 0.7 plus weighted calculated similarity
- Cross-group similarities are capped at 0.7 to preserve taxonomic structure

### 5. Domain Boosting
Fields with significant domain overlap receive similarity boosts of up to 15%, especially when specialized terminology from the same domain is heavily present in both fields. This boosting only occurs when domain similarity exceeds a configurable threshold (default: 0.7).

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Step 1: Get the Code
Clone the repository:
```bash
git clone https://github.com/Mohammed87FS/Tags_Similarity_Calcualtion.git
cd Tags_Similarity_Calcualtion
```

Alternatively, download and extract the ZIP file from GitHub.

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .conda

# Activate on Windows
.conda\Scripts\activate

# Activate on macOS/Linux
source .conda/bin/activate
```

### Step 3: Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Download Language Model
Download the required spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Configure the Application (Optional)
Edit `python/config.py` to adjust:
- Similarity component weights
- Group baseline similarities
- Domain boosting factors
- Domain terminology sets
- File paths

### Step 6: Run the Application
Navigate to the python directory and start the Flask server:
```bash
cd python
python app.py
```

The application will be accessible at `http://localhost:5000` in your web browser.

## Usage Guide

### Starting the Application

From the project root directory:
```bash
cd python
python app.py
```

Then open your web browser and navigate to `http://localhost:5000`.

### Web Interface

The web interface is divided into three main sections:

#### 1. Add Field Section
- Add new research fields to the database
- Located at the top of the page
- Accessible via the "Add Field" link in the navigation bar

#### 2. Delete Field Section
- Remove fields from the database
- Located in the middle of the page
- Accessible via the "Delete Field" link in the navigation bar

#### 3. View Similarities Section
- Compare and analyze similarities between fields
- Located at the bottom of the page
- Accessible via the "View Similarities" link in the navigation bar

### Adding a New Research Field

1. Navigate to the "Add Field" section
2. Fill in the form:
   - **Field Name**: Enter a unique name (required)
   - **Group**: Select an existing group or create a new one (required)
   - **Subgroup**: Select an existing subgroup or create a new one (required)
   - **Definition**: Provide a comprehensive definition of the field (highly recommended)
   - **Methodologies**: Describe the research approaches and methods used in this field
   - **Applications**: Explain how this field is applied in practice

3. Click "Add Field & Calculate Similarities"
4. Wait for the calculation to complete (may take a few moments)
5. Review the success message and download the updated data if needed

### Deleting a Field

1. Navigate to the "Delete Field" section
2. Select the field you want to delete from the dropdown menu
3. Click "Delete Field"
4. Confirm the deletion in the modal that appears
5. Wait for the system to delete the field and recalculate all similarities
6. Review the results and download the updated data if needed

### Viewing Field Similarities

1. Navigate to the "View Similarities" section
2. Select a field from the dropdown menu
3. Click "View Similarities"
4. Review the sorted list of all fields with their similarity scores
5. Toggle between sorting by similarity score or alphabetically
6. Click the eye icon to view detailed comparison between any two fields
7. Explore the similarity interpretation and field details in the comparison modal

### Recalculating All Similarities

1. Navigate to the "Recalculate Similarities" section
2. Click "Recalculate All Similarities"
3. Confirm the operation in the modal that appears
4. Wait for the calculation to complete (may take several minutes)
5. Review the results and download the updated data if needed

## API Reference

The application provides a RESTful API for programmatic access and integration with other tools:

### Field Management

#### Add Field
- **Endpoint**: `POST /api/add_field`
- **Description**: Add a new research field and calculate its similarities with existing fields
- **Form Parameters**:
  - `name`: Field name (required)
  - `group`: Group name (required)
  - `subgroup`: Subgroup name (required)
  - `definition`: Field definition (optional)
  - `methodologies`: Field methodologies (optional)
  - `applications`: Field applications (optional)
- **Response**: JSON with success status and operation result

#### Delete Field
- **Endpoint**: `POST /api/delete_field`
- **Description**: Delete a field and update similarity records
- **JSON Body**: `{"fieldName": "Field Name"}`
- **Response**: JSON with success status and operation result

#### Delete Field and Recalculate
- **Endpoint**: `POST /api/delete_field_all`
- **Description**: Delete a field and recalculate all similarities from scratch
- **JSON Body**: `{"fieldName": "Field Name"}`
- **Response**: JSON with success status and operation result

### Taxonomy Navigation

#### Get Subgroups
- **Endpoint**: `GET /api/get_subgroups?group={group_name}`
- **Description**: Get all subgroups for a specific group
- **Parameters**: `group` - The name of the group
- **Response**: JSON with list of subgroup names

### Similarity Analysis

#### Get Similarity Between Two Fields
- **Endpoint**: `GET /api/get_similarity?field1={field1}&field2={field2}`
- **Description**: Get similarity between two specific fields
- **Parameters**:
  - `field1`: Name of the first field
  - `field2`: Name of the second field
- **Response**: JSON with similarity score and field data

#### Get All Similarities for a Field
- **Endpoint**: `GET /api/get_all_similarities_for_field?field={field_name}`
- **Description**: Get all similarities for a specific field
- **Parameters**: `field` - The name of the field
- **Response**: JSON with sorted list of all fields and their similarity scores

#### Recalculate All Similarities
- **Endpoint**: `POST /api/recalculate_similarities`
- **Description**: Recalculate all pairwise similarities between fields
- **Response**: JSON with success status and operation result

### Data Export

#### Download Similarities
- **Endpoint**: `GET /api/download_similarities`
- **Description**: Download the complete similarity data in JSON format
- **Response**: JSON file with tags array and similarities array

### System Status

#### Test API
- **Endpoint**: `GET /api/test`
- **Description**: Check system status and data availability
- **Response**: JSON with system status and debug information

## Configuration Guide

The `config.py` file contains all configurable parameters for the application:

### File Paths
```python
DATA_DIR = 'data'
NESTED_DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'nested_descriptions_research_groups.json')
NESTED_DATA_FILE = os.path.join(DATA_DIR, 'nested_descriptions_research_groups.json')
SIMILARITY_FILE = os.path.join(DATA_DIR, 'final_outputs_enhanced_multi/field_similarities.json')
```

### Group-based Similarity Parameters
```python
SAME_GROUP_BASELINE = 0.7        # Baseline similarity for fields in same general group
SAME_SUBGROUP_BASELINE = 0.75    # Baseline similarity for fields in same subgroup
SIMILARITY_WEIGHT_SUB = 0.2      # Weight of calculated similarity to add to baseline for subgroup
SIMILARITY_WEIGHT_GENERAL = 0.15 # Weight of calculated similarity to add to baseline for general group
MAX_CROSS_GROUP_SIMILARITY = 0.7 # Maximum similarity for fields not in same group/subgroup
```

### Description Facet Weights
```python
DESCRIPTION_WEIGHTS = {
    "definition": 0.6,     # Weight for definition comparison
    "methodologies": 0.2,  # Weight for methodologies comparison
    "applications": 0.2,   # Weight for applications comparison
}
```

### Similarity Component Weights
```python
COMPONENT_WEIGHTS = {
    "embedding": 0.4,  # Weight for neural embedding similarity
    "domain": 0.4,     # Weight for domain-specific similarity
    "facet": 0.2       # Weight for faceted comparison
}
```

### Domain Boosting Configuration
```python
ENABLE_DOMAIN_BOOSTING = True    # Whether to apply domain-based score boosting
MAX_BOOST_FACTOR = 0.15          # Maximum boost to apply (0.15 = up to 15% boost)
DOMAIN_BOOST_THRESHOLD = 0.7     # Minimum domain similarity to trigger boost
```

## Technical Requirements

- **Python 3.8+**: Core programming language
- **Flask 2.2+**: Web framework for API and interface
- **Sentence Transformers 2.2+**: Neural embeddings for semantic similarity
- **scikit-learn 1.2+**: Machine learning utilities and vector operations
- **spaCy 3.5+**: Natural language processing toolkit
- **NumPy 1.24+**: Numerical computing library
- **Bootstrap 5.2+**: Frontend framework for responsive design
- **jQuery 3.6+**: JavaScript library for DOM manipulation

## Data Model

### Hierarchical Research Field Structure

The application organizes research fields in a three-level hierarchy:

```
Categories (high-level research domains)
└── Subgroups (specialized areas within domains)
    └── Fields (specific research disciplines)
```

Example:
```
Computer Science
└── Artificial Intelligence
    ├── Machine Learning
    ├── Deep Learning
    └── Reinforcement Learning
```

### Field Data Structure

Each field is represented as a JSON object with the following structure:

```json
{
  "name": "Field Name",
  "description": {
    "definition": "Comprehensive definition of the field...",
    "methodologies": "Research approaches and methods used in this field...",
    "applications": "Real-world applications and implementations..."
  }
}
```

### Similarity Data Structure

Similarities are stored as a JSON object with two main components:

1. `tags`: Array of all unique field names
2. `similarities`: Array of field pairs with calculated scores

```json
{
  "tags": ["Machine Learning", "Deep Learning", "Robotics", "..."],
  "similarities": [
    {
      "field1": "Machine Learning",
      "field2": "Deep Learning",
      "similarity_score": 0.85
    },
    {
      "field1": "Machine Learning",
      "field2": "Robotics",
      "similarity_score": 0.42
    },
    // Additional pairs...
  ]
}
```
