// Store our fields and existing fields from JSON
let fields = [];
let existingFields = [];

// DOM elements
const tagForm = document.getElementById('tag-form');
const tagInput = document.getElementById('tag-input');
const tagDescription = document.getElementById('tag-description');
const tagsContainer = document.getElementById('tags-container');
const calculateBtn = document.getElementById('calculate-btn');
const similarityResults = document.getElementById('similarity-results');
const field1Select = document.getElementById('field1-select');
const field2Select = document.getElementById('field2-select');
const compareBtn = document.getElementById('compare-btn');

// Add event listeners
tagForm.addEventListener('submit', addTag);
calculateBtn.addEventListener('click', calculateAllSimilarities);
compareBtn.addEventListener('click', compareSelectedFields);
field1Select.addEventListener('change', checkCompareButtonState);
field2Select.addEventListener('change', checkCompareButtonState);

// Load existing fields when page loads
document.addEventListener('DOMContentLoaded', loadExistingFields);

// Function to load existing fields from the server
async function loadExistingFields() {
    try {
        const response = await fetch('http://localhost:5000/get-fields');
        if (!response.ok) {
            throw new Error('Error loading existing fields');
        }
        
        const data = await response.json();
        existingFields = data.fields || [];
        
        // Populate select dropdowns
        populateFieldSelects();
        
    } catch (error) {
        console.error('Error loading fields:', error);
    }
}

// Function to populate field select dropdowns
function populateFieldSelects() {
    // Clear existing options except the first "Select a field" option
    while (field1Select.options.length > 1) {
        field1Select.remove(1);
    }
    
    while (field2Select.options.length > 1) {
        field2Select.remove(1);
    }
    
    // Add options for each field
    existingFields.forEach(field => {
        const option1 = document.createElement('option');
        option1.value = field;
        option1.textContent = field;
        field1Select.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = field;
        option2.textContent = field;
        field2Select.appendChild(option2);
    });
}

// Check if compare button should be enabled
function checkCompareButtonState() {
    const field1 = field1Select.value;
    const field2 = field2Select.value;
    compareBtn.disabled = !field1 || !field2 ;
}

// Function to compare two selected fields
async function compareSelectedFields() {
    const field1 = field1Select.value;
    const field2 = field2Select.value;
    
    if (!field1 || !field2 ) { // this should be changed.. since we need to compare the same field
        return;
    }
    if (field1 === field2) {
        similarityResults.innerHTML = `
            <div class="comparison-result high-similarity">
                <h3>Similarity Result</h3>
                <div class="comparison-fields">
                    <span class="field-name">${field1}</span>
                    <span class="similarity-arrow">↔</span>
                    <span class="field-name">${field2}</span>
                </div>
                <div class="similarity-score">
                    <span class="score-value">100.0%</span>
                    <div class="score-bar">
                        <div class="score-fill" style="width: 100%"></div>
                    </div>
                </div>
            </div>
        `;
        return;
    }

    // Show loading state
    similarityResults.innerHTML = '<p>Calculating similarity...</p>';
    
    try {
        const response = await fetch('http://localhost:5000/compare-fields', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                field1: field1,
                field2: field2
            })
        });
        
        if (!response.ok) {
            throw new Error('Error comparing fields');
        }
        
        const result = await response.json();
        
        // Display the single comparison result
        displaySingleComparisonResult(result);
        
    } catch (error) {
        console.error('Error comparing fields:', error);
        similarityResults.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

// Function to display a single comparison result
function displaySingleComparisonResult(result) {
    similarityResults.innerHTML = '';
    
    const resultCard = document.createElement('div');
    resultCard.className = 'comparison-result';
    
    // Determine similarity class
    let similarityClass = 'low-similarity';
    if (result.similarity_score >= 0.7) {
        similarityClass = 'high-similarity';
    } else if (result.similarity_score >= 0.4) {
        similarityClass = 'medium-similarity';
    }
    
    resultCard.classList.add(similarityClass);
    
    resultCard.innerHTML = `
        <h3>Similarity Result</h3>
        <div class="comparison-fields">
            <span class="field-name">${result.field1}</span>
            <span class="similarity-arrow">↔</span>
            <span class="field-name">${result.field2}</span>
        </div>
        <div class="similarity-score">
            <span class="score-value">${(result.similarity_score * 100).toFixed(1)}%</span>
            <div class="score-bar">
                <div class="score-fill" style="width: ${result.similarity_score * 100}%"></div>
            </div>
        </div>
    `;
    
    similarityResults.appendChild(resultCard);
}

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
    
    // Check if field already exists in our local array
    if (fields.some(field => field.name.toLowerCase() === name.toLowerCase())) {
        alert('This research field already exists in your current session');
        return;
    }
    
    // Check if field already exists in existing fields
    if (existingFields.some(field => field.toLowerCase() === name.toLowerCase())) {
        alert('This research field already exists in the database');
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
    
    // Enable calculate button if we have at least 1 field (can combine with existing)
    calculateBtn.disabled = fields.length < 1 && existingFields.length === 0;
}

// Function to display all locally added research fields
function displayTags() {
    tagsContainer.innerHTML = '';
    
    if (fields.length === 0) {
        tagsContainer.innerHTML = '<p>No fields added in current session. You can add new fields above.</p>';
        return;
    }
    
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
    
    // Enable calculate button if we have at least 1 field (can combine with existing)
    calculateBtn.disabled = fields.length < 1 && existingFields.length === 0;
}

// Function to calculate similarity between all fields (existing + new)
async function calculateAllSimilarities() {
    // Show loading state
    similarityResults.innerHTML = '<p>Calculating similarities for all fields...</p>';
    
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
        
        // Update existing fields list and dropdowns
        loadExistingFields();
        
        // Clear fields after successful calculation as they're now saved
        fields = [];
        displayTags();
        
    } catch (error) {
        console.error('Error calculating similarities:', error);
        similarityResults.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

// Function to display similarity results for all pairs
function displaySimilarityResults(pairs) {
    similarityResults.innerHTML = '';
    
    const resultsTitle = document.createElement('h3');
    resultsTitle.textContent = 'All Similarity Results (sorted by highest similarity)';
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
    
    });
    similarityResults.appendChild(downloadBtn);
}