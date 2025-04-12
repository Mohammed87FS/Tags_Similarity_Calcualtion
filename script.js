let tags = [];
const tagForm = document.getElementById('tag-form'); 
const tagInput = document.getElementById('tag-input');
const tagsContainer = document.getElementById('tags-container'); 
const calculateBtn = document.getElementById('calculate-btn');
const similarityResults = document.getElementById('similarity-results'); 


tagForm.addEventListener('submit', function(event) {
    event.preventDefault(); 
    const tagValue = tagInput.value.trim(); 

    if (tagValue && !tags.includes(tagValue)) {
        tags.push(tagValue); 
        updateTagsDisplay(); 
        tagInput.value = ''; 
    }
});