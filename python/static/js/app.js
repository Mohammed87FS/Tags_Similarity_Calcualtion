/**
 * Research Field Similarity Tool - Main JavaScript
 * Wrapped in IIFE to prevent global namespace pollution
 */

(function() {
    // Wait for DOM to be fully loaded
    $(document).ready(function() {
        // Define colors as RGB values for CSS variables
        document.documentElement.style.setProperty('--primary-color-rgb', '48, 80, 224');
        
        // Enable Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                boundary: document.body
            });
        });
        
        // Theme toggle functionality
        initThemeToggle();
        
        // Setup event handlers
        setupEventHandlers();
    });
    
    // Variables to store source field data for later use
    let currentSourceFieldData = null;
    let currentSourceFieldGroup = '';
    let currentSourceFieldSubgroup = '';
    
    function initThemeToggle() {
        const themeToggleBtn = document.getElementById('theme-toggle');
        if (!themeToggleBtn) return;
        
        const themeIcon = themeToggleBtn.querySelector('i');
        
        // Check if user has a saved preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.documentElement.setAttribute('data-bs-theme', 'dark');
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }
        
        themeToggleBtn.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-bs-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            document.documentElement.setAttribute('data-bs-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            if (newTheme === 'dark') {
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
            } else {
                themeIcon.classList.remove('fa-sun');
                themeIcon.classList.add('fa-moon');
            }
        });
    }
    
    function setupEventHandlers() {
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Only process if Alt key is pressed
            if (e.altKey) {
                switch (e.key.toLowerCase()) {
                    case 'a':
                        e.preventDefault();
                        document.querySelector('a[href="#add-field-section"]')?.click();
                        break;
                    case 'v':
                        e.preventDefault();
                        document.querySelector('a[href="#view-similarity-section"]')?.click();
                        break;
                    case 'h':
                        e.preventDefault();
                        const helpModal = document.getElementById('helpModal');
                        if (helpModal) new bootstrap.Modal(helpModal).show();
                        break;
                    case 'd':
                        e.preventDefault();
                        document.getElementById('theme-toggle')?.click();
                        break;
                    case 's':
                        e.preventDefault();
                        // Submit the active form
                        if (document.activeElement.closest('form')) {
                            document.activeElement.closest('form').requestSubmit();
                        }
                        break;
                }
            }
        });
        
        // Handle group selection change
        $('#field-group').change(handleGroupChange);
        
        // Handle subgroup selection change
        $('#field-subgroup').change(handleSubgroupChange);
        
        // Handle form submissions
        $('#add-field-form').submit(handleAddFieldSubmit);
        $('#view-similarity-form').submit(handleViewSimilaritySubmit);
    }
    
    function validateForm(formElement) {
        let isValid = true;
        
        // Remove all existing validation classes
        formElement.querySelectorAll('.is-invalid').forEach(el => {
            el.classList.remove('is-invalid');
        });
        
        // Check required fields
        formElement.querySelectorAll('[required]').forEach(el => {
            if (!el.value.trim()) {
                el.classList.add('is-invalid');
                isValid = false;
            } else if (el.id === 'field2' && el.value === formElement.querySelector('#field1').value) {
                el.classList.add('is-invalid');
                isValid = false;
            }
        });
        
        // Check new group/subgroup fields if they're visible
        if (formElement.querySelector('#field-group') && 
            formElement.querySelector('#field-group').value === 'new' &&
            (!formElement.querySelector('#new-group').value.trim())) {
            formElement.querySelector('#new-group').classList.add('is-invalid');
            isValid = false;
        }
        
        if (formElement.querySelector('#field-subgroup') && 
            formElement.querySelector('#field-subgroup').value === 'new' &&
            (!formElement.querySelector('#new-subgroup').value.trim())) {
            formElement.querySelector('#new-subgroup').classList.add('is-invalid');
            isValid = false;
        }
        
        return isValid;
    }
    
    function handleGroupChange() {
        const selectedGroup = $(this).val();
        
        if (selectedGroup === 'new') {
            $('#new-group').show().focus();
            $('#field-subgroup').html('<option value="new">+ Add New Subgroup</option>');
            $('#new-subgroup').show();
        } else if (selectedGroup) {
            $('#new-group').hide();
            $('#field-subgroup').html('<option value="">Loading subgroups...</option>');
            
            // Fetch subgroups for selected group
            $.getJSON('/get_subgroups', { group: selectedGroup })
                .done(function(data) {
                    if (data.success) {
                        let options = '<option value="">Select a subgroup</option>';
                        data.subgroups.forEach(function(subgroup) {
                            options += `<option value="${subgroup}">${subgroup}</option>`;
                        });
                        options += '<option value="new">+ Add New Subgroup</option>';
                        $('#field-subgroup').html(options);
                    } else {
                        showAlert('error', 'Error loading subgroups');
                    }
                })
                .fail(function() {
                    showAlert('error', 'Failed to load subgroups');
                    $('#field-subgroup').html('<option value="">Select a subgroup</option><option value="new">+ Add New Subgroup</option>');
                });
        } else {
            $('#new-group').hide();
            $('#field-subgroup').html('<option value="">Select a group first</option>');
        }
    }
    
    function handleSubgroupChange() {
        if ($(this).val() === 'new') {
            $('#new-subgroup').show().focus();
        } else {
            $('#new-subgroup').hide();
        }
    }
    
    function handleAddFieldSubmit(e) {
        e.preventDefault();
        
        // Validate form
        if (!validateForm(this)) {
            showAlert('error', 'Please fill in all required fields');
            return;
        }
        
        // Get form data
        const formData = new FormData();
        formData.append('name', $('#field-name').val());
        
        // Handle group (new or existing)
        if ($('#field-group').val() === 'new') {
            formData.append('group', $('#new-group').val());
        } else {
            formData.append('group', $('#field-group').val());
        }
        
        // Handle subgroup (new or existing)
        if ($('#field-subgroup').val() === 'new') {
            formData.append('subgroup', $('#new-subgroup').val());
        } else {
            formData.append('subgroup', $('#field-subgroup').val());
        }
        
        // Add description sections
        formData.append('definition', $('#field-definition').val());
        formData.append('methodologies', $('#field-methodologies').val());
        formData.append('applications', $('#field-applications').val());
        formData.append('technologies', $('#field-technologies').val());
        formData.append('challenges', $('#field-challenges').val());
        formData.append('future_directions', $('#field-future').val());
        
        // Show loading indicator
        $('#add-field-form').hide();
        $('#add-field-loading').show();
        
        // Submit form data
        $.ajax({
            url: '/add_field',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Hide loading indicator
                $('#add-field-loading').hide();
                
                if (response.success) {
                    // Show download section
                    $('#download-section').show();
                    
                    // Show success message
                    showAlert('success', 'Field added successfully!');
                    
                    // Reset form
                    $('#add-field-form')[0].reset();
                    $('#new-group').hide();
                    $('#new-subgroup').hide();
                    
                    // Refresh page after 5 seconds to update dropdowns
                    setTimeout(function() {
                        location.reload();
                    }, 5000);
                } else {
                    // Show form again
                    $('#add-field-form').show();
                    
                    // Show error message
                    showAlert('error', response.error || 'Error adding field');
                }
            },
            error: function(xhr) {
                // Hide loading indicator
                $('#add-field-loading').hide();
                
                // Show form again
                $('#add-field-form').show();
                
                // Show error message
                let errorMessage = 'Error adding field';
                
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }
                
                showAlert('error', errorMessage);
            }
        });
    }
    
    function handleViewSimilaritySubmit(e) {
        e.preventDefault();
        
        // Validate form
        if (!validateForm(this)) {
            return;
        }
        
        const selectedField = $('#field1').val();
        
        if (!selectedField) {
            showAlert('error', 'Please select a field');
            return;
        }
        
        // Show loading indicator
        $('#view-similarity-form').hide();
        $('#similarity-results').hide();
        $('#view-similarity-loading').show();
        
        // Set selected field name in the results section
        $('#selected-field-name').text(selectedField);
        $('#accordion-field1-name').text(selectedField);
        
        // Get all similarities for this field in a single request
        $.getJSON('/get_all_similarities_for_field', { field: selectedField })
            .done(function(data) {
                if (data.success) {
                    // Store source field data for later use in modal
                    currentSourceFieldData = data.source_field_data;
                    
                    // IMPORTANT: Save the group info from the response
                    if (data.source_field_data && typeof data.source_field_data === 'object') {
                        currentSourceFieldGroup = data.source_field_data.group || '';
                        currentSourceFieldSubgroup = data.source_field_data.subgroup || '';
                        
                        // Double-check if group might be in different location
                        if (!currentSourceFieldGroup && data.source_field_group) {
                            currentSourceFieldGroup = data.source_field_group;
                        }
                        if (!currentSourceFieldSubgroup && data.source_field_subgroup) {
                            currentSourceFieldSubgroup = data.source_field_subgroup;
                        }
                    }
                    
                    // Populate field details for the selected field
                    let fieldDetails = '<dl class="row">';
                    if (currentSourceFieldData.description) {
                        Object.entries(currentSourceFieldData.description).forEach(([key, value]) => {
                            if (value) {
                                fieldDetails += `<dt class="col-sm-3 text-capitalize">${key}:</dt><dd class="col-sm-9">${value}</dd>`;
                            }
                        });
                    }
                    fieldDetails += '</dl>';
                    $('#field1-details-content').html(fieldDetails);
                    
                    // Show similarities
                    displaySimilarityResults(selectedField, data.similarities);
                } else {
                    // Show form again
                    $('#view-similarity-loading').hide();
                    $('#view-similarity-form').show();
                    
                    // Show error message
                    showAlert('error', data.error || 'Error retrieving field data');
                }
            })
            .fail(function(xhr) {
                // Hide loading indicator
                $('#view-similarity-loading').hide();
                
                // Show form again
                $('#view-similarity-form').show();
                
                // Show error message
                let errorMessage = 'Error retrieving field data';
                
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }
                
                showAlert('error', errorMessage);
            });
    }

    function displaySimilarityResults(selectedField, similarities) {
        // Set up sorting toggle
        $('#sort-by-similarity').change(function() {
            if ($(this).is(':checked')) {
                // Sort by similarity (highest to lowest)
                similarities.sort((a, b) => b.similarity - a.similarity);
            } else {
                // Sort alphabetically
                similarities.sort((a, b) => a.field.localeCompare(b.field));
            }
            
            // Update table
            populateSimilarityTable(similarities);
        });
        
        // Initial population (sorted by similarity by default)
        similarities.sort((a, b) => b.similarity - a.similarity);
        populateSimilarityTable(similarities);
        
        // Hide loading indicator
        $('#view-similarity-loading').hide();
        
        // Show results and form
        $('#similarity-results').show();
        $('#view-similarity-form').show();
    }

    function populateSimilarityTable(similarities) {
        let tableHtml = '';
        
        if (similarities.length === 0) {
            tableHtml = '<tr><td colspan="4" class="text-center">No other fields available for comparison</td></tr>';
        } else {
            similarities.forEach(function(item) {
                // Get group/subgroup if available
                let groupText = '';
                if (item.group) {
                    groupText = item.group + (item.subgroup ? ' › ' + item.subgroup : '');
                } else {
                    groupText = 'N/A';
                }
                
                // Format similarity score
                const similarityScore = item.similarity;
                const formattedScore = similarityScore.toFixed(4);
                
                // Determine color class based on similarity
                let colorClass = '';
                if (similarityScore >= 0.7) {
                    colorClass = 'text-success';
                } else if (similarityScore >= 0.5) {
                    colorClass = 'text-primary';
                } else if (similarityScore >= 0.3) {
                    colorClass = 'text-secondary';
                } else {
                    colorClass = 'text-muted';
                }
                
                // Add table row
                tableHtml += `
                    <tr>
                        <td>${item.field}</td>
                        <td><small>${groupText}</small></td>
                        <td class="text-end ${colorClass} fw-bold">${formattedScore}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary view-details-btn" 
                                    data-field="${item.field}" data-similarity="${similarityScore}">
                                <i class="fas fa-eye" aria-hidden="true"></i>
                            </button>
                        </td>
                    </tr>
                `;
            });
        }
        
        // Update table
        $('#similarity-table-body').html(tableHtml);
        
        // Attach click handlers to view details buttons
        $('.view-details-btn').click(function() {
            const comparedField = $(this).data('field');
            const similarityScore = $(this).data('similarity');
            openComparisonModal($('#field1').val(), comparedField, similarityScore, similarities);
        });
    }

    function openComparisonModal(field1, field2, similarityScore, allSimilarities) {
        // Find the details for field2 from the similarities array
        const field2Data = allSimilarities.find(item => item.field === field2);
        
        if (!field2Data) {
            showAlert('error', 'Field data not found');
            return;
        }
        
        // Set field names
        $('#modal-field1-name').text(field1);
        $('#modal-field2-name').text(field2);
        $('#modal-accordion-field2-name').text(field2);
        
        // Set group/subgroup for both fields
        let field1Group = '';
        let field2Group = '';
        
        // Field1 group info from the stored source field data
        if (currentSourceFieldGroup) {
            field1Group = currentSourceFieldGroup;
            if (currentSourceFieldSubgroup) {
                field1Group += ' › ' + currentSourceFieldSubgroup;
            }
        }
        
        // Field2 group info comes from the API response
        if (field2Data.group) {
            field2Group = field2Data.group + (field2Data.subgroup ? ' › ' + field2Data.subgroup : '');
        }
        
        // Set modal content
        $('#modal-field1-group').text(field1Group);
        $('#modal-field2-group').text(field2Group);
        
        // Set similarity score
        const formattedScore = similarityScore.toFixed(4);
        $('#modal-similarity-score').text(formattedScore);
        
        // Animate gauge
        setTimeout(() => {
            const gaugePercent = (similarityScore * 100) + '%';
            document.documentElement.style.setProperty('--gauge-percent', gaugePercent);
            $('#modal-similarity-progress-bar').css('width', gaugePercent);
            $('#modal-similarity-progress-bar').attr('aria-valuenow', Math.round(similarityScore * 100));
        }, 100);
        
        // Set interpretation text
        const interpretationText = getInterpretationText(similarityScore);
        $('#modal-interpretation-text').html(interpretationText);
        
        // Set field2 details
        let field2Details = '<dl class="row">';
        if (field2Data.field_data && field2Data.field_data.description) {
            Object.entries(field2Data.field_data.description).forEach(([key, value]) => {
                if (value) {
                    field2Details += `<dt class="col-sm-3 text-capitalize">${key}:</dt><dd class="col-sm-9">${value}</dd>`;
                }
            });
        }
        field2Details += '</dl>';
        $('#modal-field2-details-content').html(field2Details);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('comparisonModal'));
        modal.show();
    }
    
    function getInterpretationText(similarityScore) {
        let interpretation = '';
        let interpretationClass = '';
        
        if (similarityScore >= 0.9) {
            interpretation = '<strong>Very High Similarity:</strong> These fields are extremely closely related, likely with significant overlap in their core concepts, methodologies, and applications.';
            interpretationClass = 'text-success';
        } else if (similarityScore >= 0.7) {
            interpretation = '<strong>High Similarity:</strong> These fields are closely related with substantial overlap in their domain concepts, approaches, and applications.';
            interpretationClass = 'text-success';
        } else if (similarityScore >= 0.5) {
            interpretation = '<strong>Moderate Similarity:</strong> These fields have noticeable connections and share some important concepts or methodological approaches.';
            interpretationClass = 'text-primary';
        } else if (similarityScore >= 0.3) {
            interpretation = '<strong>Low Similarity:</strong> These fields have some limited connections but are generally distinct in their approaches and focus areas.';
            interpretationClass = 'text-secondary';
        } else {
            interpretation = '<strong>Very Low Similarity:</strong> These fields appear to be substantially different with minimal overlap in concepts, methodologies, or applications.';
            interpretationClass = 'text-muted';
        }
        
        return `<p class="${interpretationClass} mb-0">${interpretation}</p>`;
    }
    
    function showAlert(type, message) {
        const alertClass = type === 'error' ? 'alert-danger' : 'alert-success';
        const icon = type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle';
        const alertHtml = `
            <div class="alert ${alertClass} alert-dismissible fade show">
                <i class="fas ${icon} me-2" aria-hidden="true"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        $('#alert-container').append(alertHtml);
        
        // Auto-dismiss alert after 5 seconds
        setTimeout(function() {
            const firstAlert = $('#alert-container .alert').first();
            if (firstAlert.length) {
                const bsAlert = bootstrap.Alert.getInstance(firstAlert[0]) || new bootstrap.Alert(firstAlert[0]);
                bsAlert.close();
            }
        }, 5000);
    }
})();


  document.addEventListener('DOMContentLoaded', function() {
    // Get modal elements
    const modal = document.getElementById('recalculateModal');
    const modalInstance = new bootstrap.Modal(modal);
    const confirmationSection = document.getElementById('modal-confirmation');
    const confirmationButtons = document.getElementById('modal-confirmation-buttons');
    const progressSection = document.getElementById('modal-progress');
    const progressButtons = document.getElementById('modal-progress-buttons');
    const resultSection = document.getElementById('modal-result');
    const resultContent = document.getElementById('result-content');
    const resultButtons = document.getElementById('modal-result-buttons');
    const calculationStatus = document.getElementById('calculation-status');
    const statusElement = document.getElementById('recalculate-status');
    
    // Button event handlers
    document.getElementById('confirm-recalculate').addEventListener('click', function() {
      // Show progress UI
      confirmationSection.classList.add('d-none');
      confirmationButtons.classList.add('d-none');
      progressSection.classList.remove('d-none');
      progressButtons.classList.remove('d-none');
      
      // Start recalculation
      fetch('/api/recalculate_similarities', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Hide progress UI
        progressSection.classList.add('d-none');
        progressButtons.classList.add('d-none');
        
        // Show result UI
        resultSection.classList.remove('d-none');
        resultButtons.classList.remove('d-none');
        
        if (data.success) {
          const timestamp = new Date().toLocaleString();
          resultContent.innerHTML = `
            <div class="text-center mb-3">
              <div class="bg-success text-white p-3 rounded-circle d-inline-block">
                <i class="fas fa-check fa-3x"></i>
              </div>
            </div>
            <div class="alert alert-success">
              <h6 class="alert-heading"><strong>Success!</strong></h6>
              <p>All field similarities have been recalculated and saved.</p>
            </div>
            <div class="card bg-light">
              <div class="card-body">
                <p class="mb-1"><strong>Recalculation Summary:</strong></p>
                <ul class="mb-0">
                  <li>${data.count} similarity pairs calculated</li>
                  <li>Completed at: ${timestamp}</li>
                </ul>
              </div>
            </div>
          `;
          
          // Also update the status outside the modal
          statusElement.innerHTML = `
            <div class="alert alert-success d-flex align-items-center">
              <i class="fas fa-check-circle me-3"></i>
              <div>
                <strong>Similarities Updated:</strong> Successfully recalculated ${data.count} field similarity pairs.
                <a href="/api/download_similarities" class="btn btn-sm btn-outline-success ms-2" download>
                  <i class="fas fa-download me-1"></i>Download
                </a>
              </div>
            </div>
          `;
        } else {
          resultContent.innerHTML = `
            <div class="text-center mb-3">
              <div class="bg-danger text-white p-3 rounded-circle d-inline-block">
                <i class="fas fa-exclamation-triangle fa-3x"></i>
              </div>
            </div>
            <div class="alert alert-danger">
              <h6 class="alert-heading"><strong>Error Occurred</strong></h6>
              <p class="mb-0">${data.error || 'An unknown error occurred during the recalculation process.'}</p>
            </div>
          `;
          
          // Update status outside modal
          statusElement.innerHTML = `
            <div class="alert alert-danger d-flex align-items-center">
              <i class="fas fa-exclamation-triangle me-3"></i>
              <div>
                <strong>Recalculation Failed:</strong> ${data.error || 'An unknown error occurred.'}
              </div>
            </div>
          `;
        }
      })
      .catch(error => {
        // Hide progress UI
        progressSection.classList.add('d-none');
        progressButtons.classList.add('d-none');
        
        // Show error result
        resultSection.classList.remove('d-none');
        resultButtons.classList.remove('d-none');
        
        resultContent.innerHTML = `
          <div class="text-center mb-3">
            <div class="bg-danger text-white p-3 rounded-circle d-inline-block">
              <i class="fas fa-times fa-3x"></i>
            </div>
          </div>
          <div class="alert alert-danger">
            <h6 class="alert-heading"><strong>Communication Error</strong></h6>
            <p class="mb-0">Failed to communicate with the server: ${error.message}</p>
          </div>
        `;
        
        // Update status outside modal
        statusElement.innerHTML = `
          <div class="alert alert-danger d-flex align-items-center">
            <i class="fas fa-times-circle me-3"></i>
            <div>
              <strong>Connection Error:</strong> Unable to complete recalculation.
            </div>
          </div>
        `;
      });
    });
    
    // Reset modal when hidden
    modal.addEventListener('hidden.bs.modal', function () {
      // Reset to confirmation view
      resultSection.classList.add('d-none');
      resultButtons.classList.add('d-none');
      progressSection.classList.add('d-none');
      progressButtons.classList.add('d-none');
      confirmationSection.classList.remove('d-none');
      confirmationButtons.classList.remove('d-none');
    });
  });