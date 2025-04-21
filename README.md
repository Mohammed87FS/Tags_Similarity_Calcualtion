# Setup Instructions for Research Field Similarity App

Follow these steps to set up and run the Research Field Similarity application:

## 1. Create Project Structure

```
research_field_similarity/
├── final_app.py
├── templates/
│   └── final_app.html
├── data/
    └── /final_outputs_enhanced_multi
        └── field_similarities.json

```

## 2. Create Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```
flask==2.3.3
numpy==1.24.3
sentence-transformers==2.2.2
spacy==3.6.1
scikit-learn==1.3.0
werkzeug==2.3.7
```

Then install the dependencies:

```bash
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

## 4. Copy Code Files

1. Copy the Python code into `app.py`
2. Copy the HTML template into `templates/index.html`
3. Create the `data` directory for storing JSON files:

```bash
mkdir -p data
```

## 5. Prepare Data Files

If you already have the following files, place them in the correct locations:

- Copy `nested_descriptions_research_groups.json` to the `data` directory
- If you have a `field_similarities.json` file, copy it to the `data` directory

If you don't have these files, the application will create sample data when you first run it.

## 6. Run the Application

```bash
# Make sure your virtual environment is activated
python app.py
```

The application will run at http://127.0.0.1:5000/

## 7. Troubleshooting

If you encounter any issues:

1. Check the console for error messages
2. Verify file paths in `app.py` (adjust if needed)
3. Visit the `/test` endpoint (http://127.0.0.1:5000/test) to check if data is loading correctly
4. Ensure your templates directory is named `templates` (with an 's') and contains `index.html`

## 8. Using the Application

Once the application is running:

1. **View Field Similarities**:
   - Select two fields from the dropdowns
   - Click "View Similarity" to see the similarity score and field details

2. **Add a New Field**:
   - Fill in the field name, group, and subgroup
   - Enter description information in the text areas
   - Click "Add Field & Calculate Similarities"
   - After processing, you can download the updated similarities file

## 9. Notes About the Algorithm

The similarity calculation uses multiple components:

- **Embedding Similarity**: Based on semantic understanding using sentence transformers
- **TF-IDF Similarity**: Term frequency analysis with emphasis on technical terminology
- **Domain Similarity**: Measures relatedness based on domain-specific concepts
- **Group-Based Adjustments**: Applies baselines based on organizational structure

These are combined to create a comprehensive similarity measure between research fields.