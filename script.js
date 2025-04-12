// Store our tags with descriptions
let fields = [];

// DOM elements
const tagForm = document.getElementById('tag-form');
const tagInput = document.getElementById('tag-input');
const tagDescription = document.getElementById('tag-description');
const tagsContainer = document.getElementById('tags-container');
const calculateBtn = document.getElementById('calculate-btn');
const similarityResults = document.getElementById('similarity-results');

// Add event listeners
tagForm.addEventListener('submit', addTag);
calculateBtn.addEventListener('click', calculateSimilarity);

// Function to add a new research field
function addTag(event) {
    event.preventDefault();
    
    const name = tagInput.value.trim();
    const description = tagDescription.value.trim();
    
    // Validate inputs
    if (!name || !description) {
        alert('Please provide both a name and description');
        return;
    }
    
    // Check if field already exists
    if (fields.some(field => field.name.toLowerCase() === name.toLowerCase())) {
        alert('This research field already exists');
        return;
    }
    
    // Add field
    fields.push({
        name: name,
        description: description
    });
    
    // Update display
    displayTags();
    
    // Clear the form
    tagInput.value = '';
    tagDescription.value = '';
    
    // Enable calculate button if we have at least 2 fields
    calculateBtn.disabled = fields.length < 2;
}

// Function to display all research fields
function displayTags() {
    tagsContainer.innerHTML = '';
    
    fields.forEach((field, index) => {
        const fieldElement = document.createElement('div');
        fieldElement.className = 'field-item';
        
        const header = document.createElement('div');
        header.className = 'field-header';
        
        const title = document.createElement('h3');
        title.textContent = field.name;
        header.appendChild(title);
        
        const removeBtn = document.createElement('button');
        removeBtn.textContent = 'Remove';
        removeBtn.className = 'remove-btn';
        removeBtn.addEventListener('click', () => removeTag(index));
        header.appendChild(removeBtn);
        
        fieldElement.appendChild(header);
        
        const preview = document.createElement('p');
        preview.className = 'field-description';
        // Show a preview of the description
        preview.textContent = field.description.substring(0, 100) + 
                            (field.description.length > 100 ? '...' : '');
        fieldElement.appendChild(preview);
        
        const expandBtn = document.createElement('button');
        expandBtn.textContent = 'Show More';
        expandBtn.className = 'expand-btn';
        expandBtn.addEventListener('click', function() {
            if (preview.classList.contains('expanded')) {
                preview.textContent = field.description.substring(0, 100) + 
                                    (field.description.length > 100 ? '...' : '');
                expandBtn.textContent = 'Show More';
                preview.classList.remove('expanded');
            } else {
                preview.textContent = field.description;
                expandBtn.textContent = 'Show Less';
                preview.classList.add('expanded');
            }
        });
        fieldElement.appendChild(expandBtn);
        
        tagsContainer.appendChild(fieldElement);
    });
}

// Function to remove a research field
function removeTag(index) {
    fields.splice(index, 1);
    displayTags();
    calculateBtn.disabled = fields.length < 2;
    
    // Clear results if we now have fewer than 2 fields
    if (fields.length < 2) {
        similarityResults.innerHTML = '';
    }
}

// Function to calculate similarity between all research fields
async function calculateSimilarity() {
    // Show loading state
    similarityResults.innerHTML = '<p>Calculating similarities...</p>';
    
    try {
        // Make API call to backend
        const response = await fetch('http://localhost:5000/calculate-similarities', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                fields: fields
            })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        displaySimilarityResults(data.pairs);
        
        // Save the results as JSON
        const jsonData = JSON.stringify(data.pairs, null, 2);
        saveAsFile(jsonData, 'research_field_similarities.json');
        
    } catch (error) {
        console.error('Error calculating similarities:', error);
        similarityResults.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

// Function to display similarity results
function displaySimilarityResults(pairs) {
    similarityResults.innerHTML = '';
    
    const resultsTitle = document.createElement('h3');
    resultsTitle.textContent = 'Similarity Results (sorted by highest similarity)';
    similarityResults.appendChild(resultsTitle);
    
    // Create a table for the results
    const table = document.createElement('table');
    table.className = 'similarity-table';
    
    // Add table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Field 1', 'Field 2', 'Similarity Score'].forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Add table body
    const tbody = document.createElement('tbody');
    pairs.forEach(pair => {
        const row = document.createElement('tr');
        
        const field1Cell = document.createElement('td');
        field1Cell.textContent = pair.field1;
        
        const field2Cell = document.createElement('td');
        field2Cell.textContent = pair.field2;
        
        const scoreCell = document.createElement('td');
        scoreCell.textContent = (pair.similarity_score * 100).toFixed(1) + '%';
        
        // Color-code based on similarity score
        if (pair.similarity_score >= 0.7) {
            row.className = 'high-similarity';
        } else if (pair.similarity_score >= 0.4) {
            row.className = 'medium-similarity';
        } else {
            row.className = 'low-similarity';
        }
        
        row.appendChild(field1Cell);
        row.appendChild(field2Cell);
        row.appendChild(scoreCell);
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    similarityResults.appendChild(table);
    
    // Add download button
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = 'Download Results as JSON';
    downloadBtn.className = 'download-btn';
    downloadBtn.addEventListener('click', () => {
        const jsonData = JSON.stringify(pairs, null, 2);
        saveAsFile(jsonData, 'research_field_similarities.json');
    });
    similarityResults.appendChild(downloadBtn);
}

// Helper function to save data as a file
function saveAsFile(data, filename) {
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
}