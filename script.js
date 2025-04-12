let tags = [];
const tagForm = document.getElementById('tag-form'); 
const tagInput = document.getElementById('tag-input');
const tagsContainer = document.getElementById('tags-container'); 
const calculateBtn = document.getElementById('calculate-btn');
const similarityResults = document.getElementById('similarity-results'); 


tagForm.addEventListener('submit',addTag);
calculateBtn.addEventListener('click',calculateSimilarity);

function addTag(event){

    event.preventDefault(); // Prevent form submission
    const tag = tagInput.value.trim(); // Get the value of the input field and trim whitespace
    if(tag && !tags.includes(tag)){ // Check if the tag is not empty and not already in the array
        tags.push(tag); // Add the tag to the array
        displayTags(); // Update the displayed tags
        tagInput.value = ''; // Clear the input field
    }
}

function displayTags(){ 
    tagsContainer.innerHTML = ''; // Clear the container
    tags.forEach((tag, index) => { // Loop through the tags array
        const tagElement = document.createElement('div'); // Create a new div for each tag
        tagElement.className = 'tag'; // Add a class to the div
        tagElement.textContent = tag; // Set the text content to the tag
        const removeBtn = document.createElement('button'); // Create a remove button
        removeBtn.textContent = 'Remove'; // Set button text
        removeBtn.addEventListener('click', () => removeTag(index)); // Add click event to remove the tag
        tagElement.appendChild(removeBtn); // Append the button to the tag element
        tagsContainer.appendChild(tagElement); // Append the tag element to the container
    });


 }

 function removeTag(index){ // Function to remove a tag
    tags.splice(index, 1); // Remove the tag from the array
    displayTags(); // Update the displayed tags
    }