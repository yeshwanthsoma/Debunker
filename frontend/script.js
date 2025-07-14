document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const audioInput = document.getElementById('audio-input');
    const textClaimInput = document.getElementById('text-claim-input');
    const transcriptionOutput = document.getElementById('transcription-output');
    const confidenceOutput = document.getElementById('confidence-output');
    const claimOutput = document.getElementById('claim-output');
    const analysisOutput = document.getElementById('analysis-output');
    const timelineOutput = document.getElementById('timeline-output');
    const credibilityOutput = document.getElementById('credibility-output');
    const prosodyOutput = document.getElementById('prosody-output');
    const sourcesOutput = document.getElementById('sources-output');
    
    // Loading indicator
    const loadingIndicator = document.createElement('div');
    loadingIndicator.id = 'loading-indicator';
    loadingIndicator.innerHTML = `
        <div class="spinner"></div>
        <p>Analyzing claim...</p>
    `;
    loadingIndicator.style.display = 'none';
    document.body.appendChild(loadingIndicator);
    
    // Add spinner styles
    const style = document.createElement('style');
    style.textContent = `
        #loading-indicator {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            color: white;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);

    analyzeBtn.addEventListener('click', function() {
        const audioFile = audioInput.files[0];
        const textClaim = textClaimInput.value.trim();

        if (!audioFile && !textClaim) {
            alert('Please provide an audio file or enter a text claim.');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        
        // Prepare data for API request
        const formData = new FormData();
        if (audioFile) {
            formData.append('audio_file', audioFile);
        }
        
        // Create request data
        const requestData = {
            text_claim: textClaim,
            enable_prosody: true
        };
        
        // If we have audio, we need to handle it
        if (audioFile) {
            // Convert audio file to base64
            const reader = new FileReader();
            reader.readAsDataURL(audioFile);
            reader.onload = function() {
                const base64Audio = reader.result.split(',')[1]; // Remove data URL prefix
                requestData.audio_data = base64Audio;
                sendAnalysisRequest(requestData);
            };
        } else {
            // No audio, just send the text claim
            sendAnalysisRequest(requestData);
        }
    });
    
    function sendAnalysisRequest(requestData) {
        console.log('Sending analysis request:', requestData);
        
        // Make API request to backend
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Update UI with results
            transcriptionOutput.textContent = data.transcription || 'No transcription available';
            confidenceOutput.textContent = data.confidence || 'Confidence: Unknown';
            claimOutput.textContent = data.claim || 'No claim extracted';
            
            // Handle HTML content - use innerHTML for markdown/HTML content
            if (data.analysis) {
                analysisOutput.innerHTML = data.analysis;
            } else {
                analysisOutput.textContent = 'No analysis available';
            }
            
            timelineOutput.innerHTML = data.timeline || '<p>No timeline available</p>';
            credibilityOutput.innerHTML = data.credibility || '<p>No credibility data available</p>';
            prosodyOutput.innerHTML = data.prosody || 'No prosody analysis available';
            
            // Update sources if available
            if (data.sources) {
                sourcesOutput.innerHTML = data.sources;
            } else {
                sourcesOutput.textContent = 'No sources available';
            }
            
            // Scroll to results
            document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Show error message
            console.error('Error analyzing claim:', error);
            alert('Error analyzing claim: ' + error.message);
        });
    }
    
    // Add example claim buttons functionality
    const exampleButtons = document.querySelectorAll('.example-btn');
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            textClaimInput.value = this.textContent;
        });
    });
});
