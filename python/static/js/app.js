/**
 * Research Field Similarity Tool - Main JavaScript
 * Wrapped in IIFE to prevent global namespace pollution
 */

(function () {
    // -----------------------------------------
    // VARIABLES AND INITIALIZATION
    // -----------------------------------------

    // Variables to store source field data for later use
    let currentSourceFieldData = null;
    let currentSourceFieldGroup = '';
    let currentSourceFieldSubgroup = '';

    // Wait for DOM to be fully loaded
    $(document).ready(function () {
        // Define colors as RGB values for CSS variables
        document.documentElement.style.setProperty('--primary-color-rgb', '48, 80, 224');

        // Initialize components
        initTooltips();
        initThemeToggle();
        initModals();
        initFieldDeletion();
        setupEventHandlers();
    });

    // -----------------------------------------
    // CORE UI INITIALIZATION
    // -----------------------------------------

    /**
     * Initialize Bootstrap tooltips
     */
    function initTooltips() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                boundary: document.body
            });
        });
    }

    /**
     * Initialize theme toggle functionality
     */
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

        // Toggle theme on click
        themeToggleBtn.addEventListener('click', function () {
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

    /**
     * Initialize modals with their handlers
     */
    function initModals() {
        // Initialize Add Field Modal
        initAddFieldModal();

        // Initialize Recalculate Modal
        initRecalculateModal();

        initDeleteFieldModal();

    }

    /**
     * Set up common event handlers
     */
    function setupEventHandlers() {
        // Keyboard shortcuts
        setupKeyboardShortcuts();

        // Form event handlers
        $('#field-group').change(handleGroupChange);
        $('#field-subgroup').change(handleSubgroupChange);

        // Form submissions
        $('#add-field-form').submit(handleAddFieldSubmit);
        $('#view-similarity-form').submit(handleViewSimilaritySubmit);
    }

    /**
     * Set up keyboard shortcuts
     */
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function (e) {
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
                    case 'r':
                        e.preventDefault();
                        document.querySelector('a[href="#recalculate-similarity-section"]')?.click();
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
    }

    // -----------------------------------------
    // MODAL INITIALIZATION AND HANDLERS
    // -----------------------------------------

    /**
     * Initialize Add Field Modal
     */
    function initAddFieldModal() {
        const modal = document.getElementById('addFieldModal');
        if (!modal) return;

        const modalInstance = new bootstrap.Modal(modal);
        const confirmationSection = document.getElementById('add-modal-confirmation');
        const confirmationButtons = document.getElementById('add-modal-confirmation-buttons');
        const progressSection = document.getElementById('add-modal-progress');
        const progressButtons = document.getElementById('add-modal-progress-buttons');
        const resultSection = document.getElementById('add-modal-result');
        const resultContent = document.getElementById('add-result-content');
        const resultButtons = document.getElementById('add-modal-result-buttons');

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

        // Add Field button click handler
        const addFieldBtn = document.getElementById('add-field-btn');
        if (addFieldBtn) {
            addFieldBtn.addEventListener('click', function (e) {
                e.preventDefault();

                // Validate form first
                if (!validateForm(document.getElementById('add-field-form'))) {
                    showAlert('error', 'Please fill in all required fields');
                    return;
                }

                // Show the modal
                modalInstance.show();
            });
        }

        // Confirm Add Field button click handler
        const confirmAddBtn = document.getElementById('confirm-add-field');
        if (confirmAddBtn) {
            confirmAddBtn.addEventListener('click', function () {
                // Show progress UI
                confirmationSection.classList.add('d-none');
                confirmationButtons.classList.add('d-none');
                progressSection.classList.remove('d-none');
                progressButtons.classList.remove('d-none');

                // Submit the form data
                submitAddFieldForm(resultSection, resultButtons, progressSection, progressButtons, resultContent);
            });
        }

        // Update the original form submission handler
        $('#add-field-form').off('submit').on('submit', function (e) {
            e.preventDefault();
            // Just trigger the add-field-btn click, which will handle validation and show the modal
            $('#add-field-btn').click();
        });
    }

    /**
     * Initialize Recalculate Modal
     */
    function initRecalculateModal() {
        const modal = document.getElementById('recalculateModal');
        if (!modal) return;

        const confirmationSection = document.getElementById('modal-confirmation');
        const confirmationButtons = document.getElementById('modal-confirmation-buttons');
        const progressSection = document.getElementById('modal-progress');
        const progressButtons = document.getElementById('modal-progress-buttons');
        const resultSection = document.getElementById('modal-result');
        const resultContent = document.getElementById('result-content');
        const resultButtons = document.getElementById('modal-result-buttons');
        const statusElement = document.getElementById('recalculate-status');

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

        // Confirm Recalculate button click handler
        const confirmRecalcBtn = document.getElementById('confirm-recalculate');
        if (confirmRecalcBtn) {
            confirmRecalcBtn.addEventListener('click', function () {
                // Show progress UI
                confirmationSection.classList.add('d-none');
                confirmationButtons.classList.add('d-none');
                progressSection.classList.remove('d-none');
                progressButtons.classList.remove('d-none');

                // Start recalculation
                recalculateSimilarities(resultSection, resultButtons, progressSection, progressButtons, resultContent, statusElement);
            });
        }
    }

    // -----------------------------------------
    // FORM HANDLING
    // -----------------------------------------

    /**
     * Validate form data
     * @param {HTMLFormElement} formElement - The form to validate
     * @returns {boolean} - Whether the form is valid
     */
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

    /**
     * Handle group selection change
     */
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
                .done(function (data) {
                    if (data.success) {
                        let options = '<option value="">Select a subgroup</option>';
                        data.subgroups.forEach(function (subgroup) {
                            options += `<option value="${subgroup}">${subgroup}</option>`;
                        });
                        options += '<option value="new">+ Add New Subgroup</option>';
                        $('#field-subgroup').html(options);
                    } else {
                        showAlert('error', 'Error loading subgroups');
                    }
                })
                .fail(function () {
                    showAlert('error', 'Failed to load subgroups');
                    $('#field-subgroup').html('<option value="">Select a subgroup</option><option value="new">+ Add New Subgroup</option>');
                });
        } else {
            $('#new-group').hide();
            $('#field-subgroup').html('<option value="">Select a group first</option>');
        }
    }

    /**
     * Handle subgroup selection change
     */
    function handleSubgroupChange() {
        if ($(this).val() === 'new') {
            $('#new-subgroup').show().focus();
        } else {
            $('#new-subgroup').hide();
        }
    }

    /**
     * Handle Add Field form submission
     * @param {Event} e - The submit event
     */
    function handleAddFieldSubmit(e) {
        e.preventDefault();

        // Validate form
        if (!validateForm(this)) {
            showAlert('error', 'Please fill in all required fields');
            return;
        }

        // Trigger the add field button click (will show modal)
        $('#add-field-btn').click();
    }

    /**
     * Submit the Add Field form data to the server
     */
    function submitAddFieldForm(resultSection, resultButtons, progressSection, progressButtons, resultContent) {
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
     


        // Submit form data
        $.ajax({
            url: '/add_field',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                // Hide progress UI
                progressSection.classList.add('d-none');
                progressButtons.classList.add('d-none');

                // Show result UI
                resultSection.classList.remove('d-none');
                resultButtons.classList.remove('d-none');

                if (response.success) {
                    const timestamp = new Date().toLocaleString();
                    resultContent.innerHTML = `
                        <div class="text-center mb-3">
                            <div class="bg-success text-white p-3 rounded-circle d-inline-block">
                                <i class="fas fa-check fa-3x"></i>
                            </div>
                        </div>
                        <div class="alert alert-success">
                            <h6 class="alert-heading"><strong>Success!</strong></h6>
                            <p>The field has been added and similarities calculated.</p>
                        </div>
                        <div class="card bg-light">
                            <div class="card-body">
                                <p class="mb-1"><strong>Field Added:</strong></p>
                                <ul class="mb-0">
                                    <li>${$('#field-name').val()}</li>
                                    <li>Added at: ${timestamp}</li>
                                </ul>
                            </div>
                        </div>
                    `;

                    // Reset form
                    document.getElementById('add-field-form').reset();
                    $('#new-group').hide();
                    $('#new-subgroup').hide();

                    // Add close button event handler to refresh the page
                    document.getElementById('add-modal-close').addEventListener('click', function () {
                        location.reload();
                    });
                } else {
                    resultContent.innerHTML = `
                        <div class="text-center mb-3">
                            <div class="bg-danger text-white p-3 rounded-circle d-inline-block">
                                <i class="fas fa-exclamation-triangle fa-3x"></i>
                            </div>
                        </div>
                        <div class="alert alert-danger">
                            <h6 class="alert-heading"><strong>Error Occurred</strong></h6>
                            <p class="mb-0">${response.error || 'An unknown error occurred while adding the field.'}</p>
                        </div>
                    `;
                }
            },
            error: function (xhr) {
                // Hide progress UI
                progressSection.classList.add('d-none');
                progressButtons.classList.add('d-none');

                // Show error result
                resultSection.classList.remove('d-none');
                resultButtons.classList.remove('d-none');

                let errorMessage = 'Error adding field';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }

                resultContent.innerHTML = `
                    <div class="text-center mb-3">
                        <div class="bg-danger text-white p-3 rounded-circle d-inline-block">
                            <i class="fas fa-times fa-3x"></i>
                        </div>
                    </div>
                    <div class="alert alert-danger">
                        <h6 class="alert-heading"><strong>Communication Error</strong></h6>
                        <p class="mb-0">Failed to add field: ${errorMessage}</p>
                    </div>
                `;
            }
        });
    }

    /**
     * Recalculate similarities between all fields
     */
    function recalculateSimilarities(resultSection, resultButtons, progressSection, progressButtons, resultContent, statusElement) {
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
                    if (statusElement) {
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
                    }
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
                    if (statusElement) {
                        statusElement.innerHTML = `
                        <div class="alert alert-danger d-flex align-items-center">
                            <i class="fas fa-exclamation-triangle me-3"></i>
                            <div>
                                <strong>Recalculation Failed:</strong> ${data.error || 'An unknown error occurred.'}
                            </div>
                        </div>
                    `;
                    }
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
                if (statusElement) {
                    statusElement.innerHTML = `
                    <div class="alert alert-danger d-flex align-items-center">
                        <i class="fas fa-times-circle me-3"></i>
                        <div>
                            <strong>Connection Error:</strong> Unable to complete recalculation.
                        </div>
                    </div>
                `;
                }
            });
    }

    // -----------------------------------------
    // SIMILARITY COMPARISON
    // -----------------------------------------

    /**
     * Handle View Similarity form submission
     * @param {Event} e - The submit event
     */
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
            .done(function (data) {
                if (data.success) {
                    // Store source field data for later use in modal
                    currentSourceFieldData = data.source_field_data;

                    // Save the group info from the response
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
            .fail(function (xhr) {
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

    /**
     * Display similarity results in the UI
     * @param {string} selectedField - The selected field name
     * @param {Array} similarities - Array of similarity data
     */
    function displaySimilarityResults(selectedField, similarities) {
        // Set up sorting toggle
        $('#sort-by-similarity').change(function () {
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

    /**
 * Populate the similarity table with data
 * @param {Array} similarities - Array of similarity data
 */
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
                     
                        <button class="btn btn-sm btn-outline-danger delete-field-btn" 
                                data-field="${item.field}" title="Delete Field">
                            <i class="fas fa-trash" aria-hidden="true"></i>
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
    
    // Attach click handlers to delete buttons
    $('.delete-field-btn').click(function() {
        const fieldToDelete = $(this).data('field');
        openDeleteFieldModal(fieldToDelete);
    });
}

    /**
     * Open the comparison modal to show details between two fields
     * @param {string} field1 - First field name
     * @param {string} field2 - Second field name
     * @param {number} similarityScore - Similarity score
     * @param {Array} allSimilarities - All similarity data
     */
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

    /**
     * Generate interpretation text based on similarity score
     * @param {number} similarityScore - Similarity score
     * @returns {string} - HTML for interpretation text
     */
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

    // -----------------------------------------
    // UTILITY FUNCTIONS
    // -----------------------------------------

    /**
     * Show an alert message
     * @param {string} type - 'error' or 'success'
     * @param {string} message - Message to display
     */
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
        setTimeout(function () {
            const firstAlert = $('#alert-container .alert').first();
            if (firstAlert.length) {
                const bsAlert = bootstrap.Alert.getInstance(firstAlert[0]) || new bootstrap.Alert(firstAlert[0]);
                bsAlert.close();
            }
        }, 5000);
    }
})();

/**
 * Initialize Delete Field Modal
 */
function initDeleteFieldModal() {
    const modal = document.getElementById('deleteFieldModal');
    if (!modal) return;
    
    const modalInstance = new bootstrap.Modal(modal);
    const confirmationSection = document.getElementById('delete-modal-confirmation');
    const confirmationButtons = document.getElementById('delete-modal-confirmation-buttons');
    const progressSection = document.getElementById('delete-modal-progress');
    const progressButtons = document.getElementById('delete-modal-progress-buttons');
    const resultSection = document.getElementById('delete-modal-result');
    const resultContent = document.getElementById('delete-result-content');
    const resultButtons = document.getElementById('delete-modal-result-buttons');
    
    // Store the field name to delete
    let fieldToDelete = '';
    
    // Expose the function to open the modal
    window.openDeleteFieldModal = function(fieldName) {
        fieldToDelete = fieldName;
        document.getElementById('delete-field-name').textContent = fieldName;
        modalInstance.show();
    };
    
    // Reset modal when hidden
    modal.addEventListener('hidden.bs.modal', function () {
        // Reset to confirmation view
        resultSection.classList.add('d-none');
        resultButtons.classList.add('d-none');
        progressSection.classList.add('d-none');
        progressButtons.classList.add('d-none');
        confirmationSection.classList.remove('d-none');
        confirmationButtons.classList.remove('d-none');
        
        // Reset field name
        fieldToDelete = '';
    });
    
    // Confirm Delete button click handler
    const confirmDeleteBtn = document.getElementById('confirm-delete-field');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', function() {
            if (!fieldToDelete) {
                showAlert('error', 'No field selected for deletion');
                return;
            }
            
            // Show progress UI
            confirmationSection.classList.add('d-none');
            confirmationButtons.classList.add('d-none');
            progressSection.classList.remove('d-none');
            progressButtons.classList.remove('d-none');
            
            // Submit the delete request
            deleteField(fieldToDelete, resultSection, resultButtons, progressSection, progressButtons, resultContent);
        });
    }
    
    // Add close button event handler
    const closeBtn = document.getElementById('delete-modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            location.reload();
        });
    }
}

/**
 * Delete a field and update similarities
 * @param {string} fieldName - Name of the field to delete
 * @param {HTMLElement} resultSection - Result section element
 * @param {HTMLElement} resultButtons - Result buttons element
 * @param {HTMLElement} progressSection - Progress section element
 * @param {HTMLElement} progressButtons - Progress buttons element
 * @param {HTMLElement} resultContent - Result content element
 */
function deleteField(fieldName, resultSection, resultButtons, progressSection, progressButtons, resultContent) {
    // Send deletion request to the server
    fetch('/api/delete_field', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fieldName: fieldName })
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
                    <p>The field has been deleted and similarities updated.</p>
                </div>
                <div class="card bg-light">
                    <div class="card-body">
                        <p class="mb-1"><strong>Deletion Summary:</strong></p>
                        <ul class="mb-0">
                            <li>Field deleted: ${fieldName}</li>
                            <li>Updated similarity pairs: ${data.updatedCount || 'N/A'}</li>
                            <li>Completed at: ${timestamp}</li>
                        </ul>
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
                    <p class="mb-0">${data.error || 'An unknown error occurred while deleting the field.'}</p>
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
                <p class="mb-0">Failed to delete field: ${error.message}</p>
            </div>
        `;
    });
}
/**
 * Initialize the field deletion functionality
 */
function initFieldDeletion() {
    const deleteForm = document.getElementById('delete-field-form');
    const fieldSelect = document.getElementById('field-to-delete');
    const deleteBtn = document.getElementById('submit-delete-btn');
    const statusDiv = document.getElementById('delete-status');
    
    if (!deleteForm || !fieldSelect || !deleteBtn) return;
    
    // Enable/disable delete button based on selection
    fieldSelect.addEventListener('change', function() {
        deleteBtn.disabled = !this.value;
    });
    
    // Handle form submission
    deleteForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedField = fieldSelect.value;
        if (!selectedField) return;
        
        // Confirm deletion
        if (confirm(`Are you sure you want to delete "${selectedField}"? This action cannot be undone.`)) {
            // Show loading indicator
            statusDiv.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="spinner-border text-danger me-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div>
                        <h6 class="mb-0">Deleting field and recalculating similarities...</h6>
                        <p class="small text-muted mb-0">This may take a few moments.</p>
                    </div>
                </div>
            `;
            statusDiv.style.display = 'block';
            
            // Disable form while processing
            fieldSelect.disabled = true;
            deleteBtn.disabled = true;
            
            // Send delete request
            deleteFieldAndRecalculate(selectedField);
        }
    });
}

/**
 * Delete a field and recalculate all similarities
 * @param {string} fieldName - The name of the field to delete
 */
function deleteFieldAndRecalculate(fieldName) {
    const statusDiv = document.getElementById('delete-status');
    const fieldSelect = document.getElementById('field-to-delete');
    const deleteBtn = document.getElementById('submit-delete-btn');
    
    // Send deletion request to the server
    fetch('/api/delete_field_all', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fieldName: fieldName })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Re-enable form elements
        fieldSelect.disabled = false;
        deleteBtn.disabled = true;
        
        if (data.success) {
            // Show success message
            const timestamp = new Date().toLocaleString();
            statusDiv.innerHTML = `
                <div class="alert alert-success">
                    <div class="d-flex">
                        <div class="me-3">
                            <i class="fas fa-check-circle fa-2x"></i>
                        </div>
                        <div>
                            <h6 class="alert-heading">Field Deleted Successfully</h6>
                            <p class="mb-0">The field "${fieldName}" has been deleted and all similarities recalculated.</p>
                            <hr>
                            <p class="mb-0"><strong>Fields remaining:</strong> ${data.fieldCount || 'N/A'}</p>
                            <p class="mb-0"><strong>Similarities calculated:</strong> ${data.comparisonCount || 'N/A'}</p>
                            <p class="mb-0"><strong>Completed at:</strong> ${timestamp}</p>
                            <div class="mt-2">
                                <a href="/api/download_similarities" class="btn btn-sm btn-outline-success" download>
                                    <i class="fas fa-download me-1"></i> Download Updated Data
                                </a>
                                <button type="button" class="btn btn-sm btn-outline-primary ms-2" onclick="location.reload()">
                                    <i class="fas fa-sync me-1"></i> Refresh Page
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Reset the select (remove the deleted option)
            fieldSelect.querySelector(`option[value="${fieldName}"]`).remove();
            fieldSelect.value = '';
        } else {
            // Show error message
            statusDiv.innerHTML = `
                <div class="alert alert-danger">
                    <div class="d-flex">
                        <div class="me-3">
                            <i class="fas fa-exclamation-circle fa-2x"></i>
                        </div>
                        <div>
                            <h6 class="alert-heading">Error Deleting Field</h6>
                            <p class="mb-0">${data.error || 'An unknown error occurred while deleting the field.'}</p>
                            <button type="button" class="btn btn-sm btn-outline-danger mt-2" onclick="document.getElementById('delete-status').style.display='none';">
                                <i class="fas fa-times me-1"></i> Close
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
    })
    .catch(error => {
        // Re-enable form elements
        fieldSelect.disabled = false;
        deleteBtn.disabled = false;
        
        // Show error message
        statusDiv.innerHTML = `
            <div class="alert alert-danger">
                <div class="d-flex">
                    <div class="me-3">
                        <i class="fas fa-exclamation-circle fa-2x"></i>
                    </div>
                    <div>
                        <h6 class="alert-heading">Communication Error</h6>
                        <p class="mb-0">Failed to communicate with the server: ${error.message}</p>
                        <button type="button" class="btn btn-sm btn-outline-danger mt-2" onclick="document.getElementById('delete-status').style.display='none';">
                            <i class="fas fa-times me-1"></i> Close
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
}
/**
 * Initialize the field deletion functionality
 */
function initFieldDeletion() {
    const deleteForm = document.getElementById('delete-field-form');
    const fieldSelect = document.getElementById('field-to-delete');
    const deleteBtn = document.getElementById('submit-delete-btn');
    const statusDiv = document.getElementById('delete-status');
    
    // Initialize modals
    const confirmModal = new bootstrap.Modal(document.getElementById('deleteConfirmModal'));
    const progressModal = new bootstrap.Modal(document.getElementById('deleteProgressModal'));
    const resultModal = new bootstrap.Modal(document.getElementById('deleteResultModal'));
    
    // References to elements in modals
    const confirmFieldName = document.getElementById('confirm-field-name');
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
    const resultContent = document.getElementById('delete-result-content');
    const resultHeader = document.getElementById('delete-result-header');
    const resultClose = document.getElementById('delete-result-close');
    
    if (!deleteForm || !fieldSelect || !deleteBtn) return;
    
    // Enable/disable delete button based on selection
    fieldSelect.addEventListener('change', function() {
        deleteBtn.disabled = !this.value;
    });
    
    // Handle form submission
    deleteForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedField = fieldSelect.value;
        if (!selectedField) return;
        
        // Show confirmation modal
        confirmFieldName.textContent = selectedField;
        confirmModal.show();
    });
    
    // Handle confirmation modal button
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', function() {
            // Get the field name from the confirmation modal
            const fieldName = confirmFieldName.textContent;
            
            // Hide confirmation modal
            confirmModal.hide();
            
            // Show progress modal
            progressModal.show();
            
            // Execute deletion
            deleteFieldAndRecalculate(fieldName, resultModal, progressModal, resultContent, resultHeader, fieldSelect);
        });
    }
    
    // Reset UI on result modal close
    if (resultClose) {
        resultClose.addEventListener('click', function() {
            // Refresh the page to update all field lists
            location.reload();
        });
    }
}

/**
 * Delete a field and recalculate all similarities
 * @param {string} fieldName - The name of the field to delete
 * @param {bootstrap.Modal} resultModal - The result modal
 * @param {bootstrap.Modal} progressModal - The progress modal
 * @param {HTMLElement} resultContent - The result content element
 * @param {HTMLElement} resultHeader - The result header element
 * @param {HTMLElement} fieldSelect - The field select element
 */
function deleteFieldAndRecalculate(fieldName, resultModal, progressModal, resultContent, resultHeader, fieldSelect) {
    // Send deletion request to the server
    fetch('/api/delete_field', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fieldName: fieldName })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Hide progress modal
        progressModal.hide();
        
        if (data.success) {
            // Update header for success
            resultHeader.className = 'modal-header bg-success text-white';
            document.getElementById('deleteResultModalLabel').textContent = 'Deletion Successful';
            
            // Show success message
            const timestamp = new Date().toLocaleString();
            resultContent.innerHTML = `
                <div class="text-center mb-4">
                    <div class="bg-success text-white p-3 rounded-circle d-inline-block">
                        <i class="fas fa-check fa-3x"></i>
                    </div>
                </div>
                <div class="alert alert-success">
                    <h6 class="alert-heading">Field Deleted Successfully</h6>
                    <p>The field "${fieldName}" has been deleted and all similarities recalculated.</p>
                </div>
              
            `;
            
            // Remove the option from the select
            const option = fieldSelect.querySelector(`option[value="${fieldName}"]`);
            if (option) option.remove();
            
            // Reset select value
            fieldSelect.value = '';
            
            // Disable delete button until new selection
            document.getElementById('submit-delete-btn').disabled = true;
        } else {
            // Update header for error
            resultHeader.className = 'modal-header bg-danger text-white';
            document.getElementById('deleteResultModalLabel').textContent = 'Deletion Failed';
            
            // Show error message
            resultContent.innerHTML = `
                <div class="text-center mb-4">
                    <div class="bg-danger text-white p-3 rounded-circle d-inline-block">
                        <i class="fas fa-exclamation-circle fa-3x"></i>
                    </div>
                </div>
                <div class="alert alert-danger">
                    <h6 class="alert-heading">Error Deleting Field</h6>
                    <p class="mb-0">${data.error || 'An unknown error occurred while deleting the field.'}</p>
                </div>
            `;
        }
        
        // Show result modal
        resultModal.show();
    })
    .catch(error => {
        // Hide progress modal
        progressModal.hide();
        
        // Update header for error
        resultHeader.className = 'modal-header bg-danger text-white';
        document.getElementById('deleteResultModalLabel').textContent = 'Deletion Error';
        
        // Show error message
        resultContent.innerHTML = `
            <div class="text-center mb-4">
                <div class="bg-danger text-white p-3 rounded-circle d-inline-block">
                    <i class="fas fa-times fa-3x"></i>
                </div>
            </div>
            <div class="alert alert-danger">
                <h6 class="alert-heading">Communication Error</h6>
                <p class="mb-0">Failed to communicate with the server: ${error.message}</p>
            </div>
        `;
        
        // Show result modal
        resultModal.show();
    });
}

// Navigation tab functionality - Show only the clicked section
document.addEventListener('DOMContentLoaded', function() {
    // Get all the nav links that point to content sections
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link[href^="#"]');
    
    // Get all content sections
    const contentSections = [
        document.getElementById('add-field-section'),
        document.getElementById('delete-field-section'),
        document.getElementById('view-similarity-section'),
        document.getElementById('recalculate-similarity-section')
    ].filter(section => section); // Filter out any null values
    
    // Function to show only the target section
    function showOnlySection(sectionId) {
        // Hide all sections
        contentSections.forEach(section => {
            section.closest('.row').style.display = 'none';
        });
        
        // Show only the target section
        const targetSection = document.getElementById(sectionId.substring(1)); // Remove the # from the ID
        if (targetSection) {
            targetSection.closest('.row').style.display = '';
        }
        
        // Add active class to current nav item and remove from others
        navLinks.forEach(link => {
            if (link.getAttribute('href') === sectionId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
    
    // Add click event to each nav link
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Skip links that open modals
            if(this.getAttribute('data-bs-toggle')) return;
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            e.preventDefault();
            showOnlySection(targetId);
            
            // Update URL fragment without scrolling
            history.pushState(null, null, targetId);
        });
    });
    
    // Show default section or section from URL fragment on page load
    const hash = window.location.hash;
    if (hash && document.querySelector(hash)) {
        showOnlySection(hash);
    } else {
        // Default to showing the add field section
        showOnlySection('#add-field-section');
    }
});