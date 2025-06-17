
(function () {
    
    
    

    
    let currentSourceFieldData = null;
    let currentSourceFieldGroup = '';
    let currentSourceFieldSubgroup = '';

    
    $(document).ready(function () {
        
        document.documentElement.style.setProperty('--primary-color-rgb', '48, 80, 224');

        
        initTooltips();
        initThemeToggle();
        initModals();
        initFieldDeletion();
        setupEventHandlers();
    });

    
    
    

    function initTooltips() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                boundary: document.body
            });
        });
    }

    function initThemeToggle() {
        const themeToggleBtn = document.getElementById('theme-toggle');
        if (!themeToggleBtn) return;

        const themeIcon = themeToggleBtn.querySelector('i');

        
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.documentElement.setAttribute('data-bs-theme', 'dark');
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }

        
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

 
    function initModals() {
        
        initAddFieldModal();

        
        initRecalculateModal();

        initDeleteFieldModal();

    }

 
    function setupEventHandlers() {
        
        setupKeyboardShortcuts();

        
        $('#field-group').change(handleGroupChange);
        $('#field-subgroup').change(handleSubgroupChange);

        
        $('#add-field-form').submit(handleAddFieldSubmit);
        $('#view-similarity-form').submit(handleViewSimilaritySubmit);
    }

  
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function (e) {
            
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
                        
                        if (document.activeElement.closest('form')) {
                            document.activeElement.closest('form').requestSubmit();
                        }
                        break;
                }
            }
        });
    }

    

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

        
        modal.addEventListener('hidden.bs.modal', function () {
            
            resultSection.classList.add('d-none');
            resultButtons.classList.add('d-none');
            progressSection.classList.add('d-none');
            progressButtons.classList.add('d-none');
            confirmationSection.classList.remove('d-none');
            confirmationButtons.classList.remove('d-none');
        });

        
        const addFieldBtn = document.getElementById('add-field-btn');
        if (addFieldBtn) {
            addFieldBtn.addEventListener('click', function (e) {
                e.preventDefault();

                
                if (!validateForm(document.getElementById('add-field-form'))) {
                    showAlert('error', 'Please fill in all required fields');
                    return;
                }

                
                modalInstance.show();
            });
        }

        
        const confirmAddBtn = document.getElementById('confirm-add-field');
        if (confirmAddBtn) {
            confirmAddBtn.addEventListener('click', function () {
                
                confirmationSection.classList.add('d-none');
                confirmationButtons.classList.add('d-none');
                progressSection.classList.remove('d-none');
                progressButtons.classList.remove('d-none');

                
                submitAddFieldForm(resultSection, resultButtons, progressSection, progressButtons, resultContent);
            });
        }

        
        $('#add-field-form').off('submit').on('submit', function (e) {
            e.preventDefault();
            
            $('#add-field-btn').click();
        });
    }

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

        
        modal.addEventListener('hidden.bs.modal', function () {
            
            resultSection.classList.add('d-none');
            resultButtons.classList.add('d-none');
            progressSection.classList.add('d-none');
            progressButtons.classList.add('d-none');
            confirmationSection.classList.remove('d-none');
            confirmationButtons.classList.remove('d-none');
        });

        
        const confirmRecalcBtn = document.getElementById('confirm-recalculate');
        if (confirmRecalcBtn) {
            confirmRecalcBtn.addEventListener('click', function () {
                
                confirmationSection.classList.add('d-none');
                confirmationButtons.classList.add('d-none');
                progressSection.classList.remove('d-none');
                progressButtons.classList.remove('d-none');

                
                recalculateSimilarities(resultSection, resultButtons, progressSection, progressButtons, resultContent, statusElement);
            });
        }
    }

    
    
    function validateForm(formElement) {
        let isValid = true;

        
        formElement.querySelectorAll('.is-invalid').forEach(el => {
            el.classList.remove('is-invalid');
        });

        
        formElement.querySelectorAll('[required]').forEach(el => {
            if (!el.value.trim()) {
                el.classList.add('is-invalid');
                isValid = false;
            } else if (el.id === 'field2' && el.value === formElement.querySelector('#field1').value) {
                el.classList.add('is-invalid');
                isValid = false;
            }
        });

        
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

    
    function handleSubgroupChange() {
        if ($(this).val() === 'new') {
            $('#new-subgroup').show().focus();
        } else {
            $('#new-subgroup').hide();
        }
    }

   
    function handleAddFieldSubmit(e) {
        e.preventDefault();

        
        if (!validateForm(this)) {
            showAlert('error', 'Please fill in all required fields');
            return;
        }

        
        $('#add-field-btn').click();
    }

   
    function submitAddFieldForm(resultSection, resultButtons, progressSection, progressButtons, resultContent) {
        
        const formData = new FormData();
        formData.append('name', $('#field-name').val());

        
        if ($('#field-group').val() === 'new') {
            formData.append('group', $('#new-group').val());
        } else {
            formData.append('group', $('#field-group').val());
        }

        
        if ($('#field-subgroup').val() === 'new') {
            formData.append('subgroup', $('#new-subgroup').val());
        } else {
            formData.append('subgroup', $('#field-subgroup').val());
        }

        
        formData.append('definition', $('#field-definition').val());
        formData.append('methodologies', $('#field-methodologies').val());
        formData.append('applications', $('#field-applications').val());
     


        
        $.ajax({
            url: '/add_field',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                
                progressSection.classList.add('d-none');
                progressButtons.classList.add('d-none');

                
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

                    
                    document.getElementById('add-field-form').reset();
                    $('#new-group').hide();
                    $('#new-subgroup').hide();

                    
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
                
                progressSection.classList.add('d-none');
                progressButtons.classList.add('d-none');

                
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
                
                progressSection.classList.add('d-none');
                progressButtons.classList.add('d-none');

                
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
                
                progressSection.classList.add('d-none');
                progressButtons.classList.add('d-none');

                
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

    
    
    

    function handleViewSimilaritySubmit(e) {
        e.preventDefault();

        
        if (!validateForm(this)) {
            return;
        }

        const selectedField = $('#field1').val();

        if (!selectedField) {
            showAlert('error', 'Please select a field');
            return;
        }

        
        $('#view-similarity-form').hide();
        $('#similarity-results').hide();
        $('#view-similarity-loading').show();

        
        $('#selected-field-name').text(selectedField);
        $('#accordion-field1-name').text(selectedField);

        
        $.getJSON('/get_all_similarities_for_field', { field: selectedField })
            .done(function (data) {
                if (data.success) {
                    
                    currentSourceFieldData = data.source_field_data;

                    
                    if (data.source_field_data && typeof data.source_field_data === 'object') {
                        currentSourceFieldGroup = data.source_field_data.group || '';
                        currentSourceFieldSubgroup = data.source_field_data.subgroup || '';

                        
                        if (!currentSourceFieldGroup && data.source_field_group) {
                            currentSourceFieldGroup = data.source_field_group;
                        }
                        if (!currentSourceFieldSubgroup && data.source_field_subgroup) {
                            currentSourceFieldSubgroup = data.source_field_subgroup;
                        }
                    }

                    
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

                    
                    displaySimilarityResults(selectedField, data.similarities);
                } else {
                    
                    $('#view-similarity-loading').hide();
                    $('#view-similarity-form').show();

                    
                    showAlert('error', data.error || 'Error retrieving field data');
                }
            })
            .fail(function (xhr) {
                
                $('#view-similarity-loading').hide();

                
                $('#view-similarity-form').show();

                
                let errorMessage = 'Error retrieving field data';

                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }

                showAlert('error', errorMessage);
            });
    }


    function displaySimilarityResults(selectedField, similarities) {
        
        $('#sort-by-similarity').change(function () {
            if ($(this).is(':checked')) {
                
                similarities.sort((a, b) => b.similarity - a.similarity);
            } else {
                
                similarities.sort((a, b) => a.field.localeCompare(b.field));
            }

            
            populateSimilarityTable(similarities);
        });

        
        similarities.sort((a, b) => b.similarity - a.similarity);
        populateSimilarityTable(similarities);

        
        $('#view-similarity-loading').hide();

        
        $('#similarity-results').show();
        $('#view-similarity-form').show();
    }

function populateSimilarityTable(similarities) {
    let tableHtml = '';
    
    if (similarities.length === 0) {
        tableHtml = '<tr><td colspan="4" class="text-center">No other fields available for comparison</td></tr>';
    } else {
        similarities.forEach(function(item) {
            
            let groupText = '';
            if (item.group) {
                groupText = item.group + (item.subgroup ? ' › ' + item.subgroup : '');
            } else {
                groupText = 'N/A';
            }
            
            
            const similarityScore = item.similarity;
            const formattedScore = similarityScore.toFixed(4);
            
            
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
    
    
    $('#similarity-table-body').html(tableHtml);
    
    
    $('.view-details-btn').click(function() {
        const comparedField = $(this).data('field');
        const similarityScore = $(this).data('similarity');
        openComparisonModal($('#field1').val(), comparedField, similarityScore, similarities);
    });
    
    
    $('.delete-field-btn').click(function() {
        const fieldToDelete = $(this).data('field');
        openDeleteFieldModal(fieldToDelete);
    });
}

  
    function openComparisonModal(field1, field2, similarityScore, allSimilarities) {
        
        const field2Data = allSimilarities.find(item => item.field === field2);

        if (!field2Data) {
            showAlert('error', 'Field data not found');
            return;
        }

        
        $('#modal-field1-name').text(field1);
        $('#modal-field2-name').text(field2);
        $('#modal-accordion-field2-name').text(field2);

        
        let field1Group = '';
        let field2Group = '';

        
        if (currentSourceFieldGroup) {
            field1Group = currentSourceFieldGroup;
            if (currentSourceFieldSubgroup) {
                field1Group += ' › ' + currentSourceFieldSubgroup;
            }
        }

        
        if (field2Data.group) {
            field2Group = field2Data.group + (field2Data.subgroup ? ' › ' + field2Data.subgroup : '');
        }

        
        $('#modal-field1-group').text(field1Group);
        $('#modal-field2-group').text(field2Group);

        
        const formattedScore = similarityScore.toFixed(4);
        $('#modal-similarity-score').text(formattedScore);

        
        setTimeout(() => {
            const gaugePercent = (similarityScore * 100) + '%';
            document.documentElement.style.setProperty('--gauge-percent', gaugePercent);
            $('#modal-similarity-progress-bar').css('width', gaugePercent);
            $('#modal-similarity-progress-bar').attr('aria-valuenow', Math.round(similarityScore * 100));
        }, 100);

        
        const interpretationText = getInterpretationText(similarityScore);
        $('#modal-interpretation-text').html(interpretationText);

        
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

        
        setTimeout(function () {
            const firstAlert = $('#alert-container .alert').first();
            if (firstAlert.length) {
                const bsAlert = bootstrap.Alert.getInstance(firstAlert[0]) || new bootstrap.Alert(firstAlert[0]);
                bsAlert.close();
            }
        }, 5000);
    }
})();


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
    
    
    let fieldToDelete = '';
    
    
    window.openDeleteFieldModal = function(fieldName) {
        fieldToDelete = fieldName;
        document.getElementById('delete-field-name').textContent = fieldName;
        modalInstance.show();
    };
    
    
    modal.addEventListener('hidden.bs.modal', function () {
        
        resultSection.classList.add('d-none');
        resultButtons.classList.add('d-none');
        progressSection.classList.add('d-none');
        progressButtons.classList.add('d-none');
        confirmationSection.classList.remove('d-none');
        confirmationButtons.classList.remove('d-none');
        
        
        fieldToDelete = '';
    });
    
    
    const confirmDeleteBtn = document.getElementById('confirm-delete-field');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', function() {
            if (!fieldToDelete) {
                showAlert('error', 'No field selected for deletion');
                return;
            }
            
            
            confirmationSection.classList.add('d-none');
            confirmationButtons.classList.add('d-none');
            progressSection.classList.remove('d-none');
            progressButtons.classList.remove('d-none');
            
            
            deleteField(fieldToDelete, resultSection, resultButtons, progressSection, progressButtons, resultContent);
        });
    }
    
    
    const closeBtn = document.getElementById('delete-modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            location.reload();
        });
    }
}

function deleteField(fieldName, resultSection, resultButtons, progressSection, progressButtons, resultContent) {
    
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
        
        progressSection.classList.add('d-none');
        progressButtons.classList.add('d-none');
        
        
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
        
        progressSection.classList.add('d-none');
        progressButtons.classList.add('d-none');
        
        
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

function initFieldDeletion() {
    const deleteForm = document.getElementById('delete-field-form');
    const fieldSelect = document.getElementById('field-to-delete');
    const deleteBtn = document.getElementById('submit-delete-btn');
    const statusDiv = document.getElementById('delete-status');
    
    if (!deleteForm || !fieldSelect || !deleteBtn) return;
    
    
    fieldSelect.addEventListener('change', function() {
        deleteBtn.disabled = !this.value;
    });
    
    
    deleteForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedField = fieldSelect.value;
        if (!selectedField) return;
        
        
        if (confirm(`Are you sure you want to delete "${selectedField}"? This action cannot be undone.`)) {
            
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
            
            
            fieldSelect.disabled = true;
            deleteBtn.disabled = true;
            
            
            deleteFieldAndRecalculate(selectedField);
        }
    });
}


function deleteFieldAndRecalculate(fieldName) {
    const statusDiv = document.getElementById('delete-status');
    const fieldSelect = document.getElementById('field-to-delete');
    const deleteBtn = document.getElementById('submit-delete-btn');
    
    
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
        
        fieldSelect.disabled = false;
        deleteBtn.disabled = true;
        
        if (data.success) {
            
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
            
            
            fieldSelect.querySelector(`option[value="${fieldName}"]`).remove();
            fieldSelect.value = '';
        } else {
            
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
        
        fieldSelect.disabled = false;
        deleteBtn.disabled = false;
        
        
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

function initFieldDeletion() {
    const deleteForm = document.getElementById('delete-field-form');
    const fieldSelect = document.getElementById('field-to-delete');
    const deleteBtn = document.getElementById('submit-delete-btn');
    const statusDiv = document.getElementById('delete-status');
    
    
    const confirmModal = new bootstrap.Modal(document.getElementById('deleteConfirmModal'));
    const progressModal = new bootstrap.Modal(document.getElementById('deleteProgressModal'));
    const resultModal = new bootstrap.Modal(document.getElementById('deleteResultModal'));
    
    
    const confirmFieldName = document.getElementById('confirm-field-name');
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
    const resultContent = document.getElementById('delete-result-content');
    const resultHeader = document.getElementById('delete-result-header');
    const resultClose = document.getElementById('delete-result-close');
    
    if (!deleteForm || !fieldSelect || !deleteBtn) return;
    
    
    fieldSelect.addEventListener('change', function() {
        deleteBtn.disabled = !this.value;
    });
    
    
    deleteForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedField = fieldSelect.value;
        if (!selectedField) return;
        
        
        confirmFieldName.textContent = selectedField;
        confirmModal.show();
    });
    
    
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', function() {
            
            const fieldName = confirmFieldName.textContent;
            
            
            confirmModal.hide();
            
            
            progressModal.show();
            
            
            deleteFieldAndRecalculate(fieldName, resultModal, progressModal, resultContent, resultHeader, fieldSelect);
        });
    }
    
    
    if (resultClose) {
        resultClose.addEventListener('click', function() {
            
            location.reload();
        });
    }
}


function deleteFieldAndRecalculate(fieldName, resultModal, progressModal, resultContent, resultHeader, fieldSelect) {
    
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
        
        progressModal.hide();
        
        if (data.success) {
            
            resultHeader.className = 'modal-header bg-success text-white';
            document.getElementById('deleteResultModalLabel').textContent = 'Deletion Successful';
            
            
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
            
            
            const option = fieldSelect.querySelector(`option[value="${fieldName}"]`);
            if (option) option.remove();
            
            
            fieldSelect.value = '';
            
            
            document.getElementById('submit-delete-btn').disabled = true;
        } else {
            
            resultHeader.className = 'modal-header bg-danger text-white';
            document.getElementById('deleteResultModalLabel').textContent = 'Deletion Failed';
            
            
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
        
        
        resultModal.show();
    })
    .catch(error => {
        
        progressModal.hide();
        
        
        resultHeader.className = 'modal-header bg-danger text-white';
        document.getElementById('deleteResultModalLabel').textContent = 'Deletion Error';
        
        
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
        
        
        resultModal.show();
    });
}


document.addEventListener('DOMContentLoaded', function() {
    
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link[href^="#"]');
    
    
    const contentSections = [
        document.getElementById('add-field-section'),
        document.getElementById('delete-field-section'),
        document.getElementById('view-similarity-section'),
        document.getElementById('recalculate-similarity-section')
    ].filter(section => section); 
    
    
    function showOnlySection(sectionId) {
        
        contentSections.forEach(section => {
            section.closest('.row').style.display = 'none';
        });
        
        
        const targetSection = document.getElementById(sectionId.substring(1)); 
        if (targetSection) {
            targetSection.closest('.row').style.display = '';
        }
        
        
        navLinks.forEach(link => {
            if (link.getAttribute('href') === sectionId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
    
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            
            if(this.getAttribute('data-bs-toggle')) return;
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            e.preventDefault();
            showOnlySection(targetId);
            
            
            history.pushState(null, null, targetId);
        });
    });
    
    
    const hash = window.location.hash;
    if (hash && document.querySelector(hash)) {
        showOnlySection(hash);
    } else {
        
        showOnlySection('#add-field-section');
    }
});