document.addEventListener('DOMContentLoaded', function() {
    const queryImage = document.getElementById('queryImage');
    const pickButton = document.getElementById('pickButton');
    const retrieveButton = document.getElementById('retrieveButton');
    const resultsContainer = document.getElementById('resultsContainer');

    fetch('/api/initialize', { method: 'POST' })
        .then(response => response.json())
        .then(data => console.log('Service initialized'))
        .catch(error => console.error('Error initializing service:', error));

    pickButton.addEventListener('click', function() {
        fetch('/api/random-image')
            .then(response => response.json())
            .then(data => {
                queryImage.src = `data:image/png;base64,${data.image}`;
                queryImage.dataset.imageId = data.id;
                resultsContainer.innerHTML = '<p>Image selected. Click "Retrieve Similar Images" to see results.</p>';
            })
            .catch(error => {
                console.error('Error getting random image:', error);
                resultsContainer.innerHTML = '<p>Error getting image. Please try again.</p>';
            });
    });

    retrieveButton.addEventListener('click', function() {
        if (!queryImage.dataset.imageId) {
            alert('Please pick an image first!');
            return;
        }

        resultsContainer.innerHTML = '<p>Loading similar images...</p>';

        fetch('/api/retrieve-similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageId: queryImage.dataset.imageId })
        })
        .then(response => response.json())
        .then(data => {
            resultsContainer.innerHTML = '';
            
            if (data.similar_images && data.similar_images.length > 0) {
                data.similar_images.forEach((image, index) => {
                    const imgElement = document.createElement('img');
                    imgElement.src = `data:image/png;base64,${image}`;
                    imgElement.alt = 'Similar image';
                    imgElement.className = 'result-image';
                    resultsContainer.appendChild(imgElement);
                });
            } else {
                resultsContainer.textContent = 'No similar images found.';
            }
        })
        .catch(error => {
            console.error('Error retrieving similar images:', error);
            resultsContainer.innerHTML = '<p>Error retrieving images. Please try again.</p>';
        });
    });
});
