# Research Field Similarity Web Application

This Flask application allows users to manage research fields and calculate similarities between them.

## Project Structure

```
/research_field_similarity/
├── app.py                           # Main Flask application
├── requirements.txt                 # Python dependencies
├── data/                            # Data directory
│   ├── nested_descriptions_research_groups.json  # Research fields data
│   └── field_similarities.json      # Precomputed similarities
├── templates/                       # Flask templates
│   └── index.html                   # Main application template
└── static/                          # Static files (if needed)
```

## Setup Instructions

1. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare the data directory**

```bash
mkdir -p data
```

4. **Place your JSON files in the data directory**

- Copy your `nested_descriptions_research_groups.json` to the `data` directory
- Copy your `field_similarities.json` to the `data` directory (if you have it)

5. **Setup the templates directory**

```bash
mkdir templates
```

6. **Copy the provided index.html to the templates directory**

7. **Run the application**

```bash
python app.py
```

8. **Access the application**

Open your browser and navigate to `http://127.0.0.1:5000`

## Requirements

Create a `requirements.txt` file with the following content:

```
flask==2.3.3
numpy==1.24.3
sentence-transformers==2.2.2
spacy==3.6.1
scikit-learn==1.3.0
werkzeug==2.3.7
```

You'll also need to download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Features

1. **Add a New Field**
   - Fill out the form with field details
   - The new field is added to the dataset
   - Similarities with existing fields are calculated
   - Updated similarities file is available for download

2. **View Field Similarities**
   - Select two fields from dropdown menus
   - View calculated similarity score
   - Explore field details in expandable sections

## Implementation Details

- The application uses `sentence-transformers` for high-quality text embeddings
- Similarity calculation follows a multi-faceted approach with group adjustments
- The code is optimized to only calculate new similarities, not recompute existing ones
- Group-based adjustments use baseline similarities for same group/subgroup fields