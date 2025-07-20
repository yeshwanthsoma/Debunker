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
            
            // Update Summary Tab
            updateSummaryTab(data);
            
            // Update Evidence Tab  
            updateEvidenceTab(data);
            
            // Update Debate Tab
            updateDebateTab(data);
            
            // Update Credibility Tab
            updateCredibilityTab(data);
            
            // Update Audio Analysis Tab
            updateAudioAnalysisTab(data);
            
            // Update Sources Tab
            updateSourcesTab(data);
            
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
    
    // Functions to update different tabs with new backend structure
    function updateSummaryTab(data) {
        // Update verdict with styling
        const verdictElement = document.getElementById('verdict-output');
        const verdictText = data.verdict || 'Unknown';
        verdictElement.textContent = verdictText;
        verdictElement.className = `verdict-badge verdict-${verdictText.toLowerCase()}`;
        
        // Update confidence
        const confidence = data.confidence || 0;
        document.getElementById('confidence-output').textContent = `${(confidence * 100).toFixed(1)}%`;
        
        // Update other summary fields
        document.getElementById('claim-output').textContent = data.claim || 'No claim extracted';
        document.getElementById('explanation-output').textContent = data.explanation || 'No explanation available';
        document.getElementById('transcription-output').textContent = data.transcription || 'No transcription available';
    }
    
    function updateEvidenceTab(data) {
        const evidence = data.evidence || {};
        
        // Primary claims
        const primaryClaims = evidence.primary_claims || [];
        const claimsHtml = primaryClaims.length > 0 
            ? `<ul>${primaryClaims.map(claim => `<li>${claim}</li>`).join('')}</ul>`
            : '<p>No primary claims identified</p>';
        document.getElementById('primary-claims-output').innerHTML = claimsHtml;
        
        // Claim evaluations
        const evaluations = evidence.claim_evaluations || {};
        const evaluationsHtml = Object.keys(evaluations).length > 0
            ? Object.entries(evaluations).map(([claim, status]) => 
                `<div class="evaluation-item">
                    <strong>${claim}:</strong> <span class="status-${status.toLowerCase()}">${status}</span>
                </div>`).join('')
            : '<p>No claim evaluations available</p>';
        document.getElementById('claim-evaluations-output').innerHTML = evaluationsHtml;
        
        // Web sources consulted
        const webSources = evidence.web_sources_consulted || [];
        const sourcesHtml = webSources.length > 0
            ? `<ul>${webSources.map(source => `<li>${source}</li>`).join('')}</ul>`
            : '<p>No web sources consulted</p>';
        document.getElementById('web-sources-output').innerHTML = sourcesHtml;
        
        // Reasoning
        document.getElementById('reasoning-output').textContent = evidence.reasoning || 'No reasoning provided';
    }
    
    function updateDebateTab(data) {
        const debate = data.debate_content;
        
        if (!debate) {
            document.getElementById('supporting-args-output').innerHTML = '<p>No debate content available</p>';
            document.getElementById('opposing-args-output').innerHTML = '<p>No debate content available</p>';
            document.getElementById('expert-opinions-output').innerHTML = '<p>No expert opinions available</p>';
            document.getElementById('historical-context-output').textContent = 'No historical context available';
            document.getElementById('scientific-consensus-output').textContent = 'No scientific consensus available';
            return;
        }
        
        // Supporting arguments
        const supportingHtml = debate.supporting_arguments && debate.supporting_arguments.length > 0
            ? `<ul>${debate.supporting_arguments.map(arg => `<li>${arg}</li>`).join('')}</ul>`
            : '<p>No supporting arguments available</p>';
        document.getElementById('supporting-args-output').innerHTML = supportingHtml;
        
        // Opposing arguments
        const opposingHtml = debate.opposing_arguments && debate.opposing_arguments.length > 0
            ? `<ul>${debate.opposing_arguments.map(arg => `<li>${arg}</li>`).join('')}</ul>`
            : '<p>No opposing arguments available</p>';
        document.getElementById('opposing-args-output').innerHTML = opposingHtml;
        
        // Expert opinions
        const expertsHtml = debate.expert_opinions && debate.expert_opinions.length > 0
            ? debate.expert_opinions.map(expert => 
                `<div class="expert-opinion">
                    <strong>${expert.expert}:</strong> ${expert.opinion}
                </div>`).join('')
            : '<p>No expert opinions available</p>';
        document.getElementById('expert-opinions-output').innerHTML = expertsHtml;
        
        // Context and consensus
        document.getElementById('historical-context-output').textContent = debate.historical_context || 'No historical context available';
        document.getElementById('scientific-consensus-output').textContent = debate.scientific_consensus || 'No scientific consensus available';
    }
    
    function updateCredibilityTab(data) {
        const metrics = data.credibility_metrics || {};
        
        // Update metric bars
        updateMetricBar('language-quality', metrics.language_quality || 0);
        updateMetricBar('audio-authenticity', metrics.audio_authenticity || 0);
        updateMetricBar('source-reliability', metrics.source_reliability || 0);
        updateMetricBar('factual-accuracy', metrics.factual_accuracy || 0);
        
        // Update warning flags
        const flags = metrics.flags || [];
        const flagsHtml = flags.length > 0
            ? `<ul>${flags.map(flag => `<li class="warning-flag">${flag}</li>`).join('')}</ul>`
            : '<p class="no-flags">No warning flags detected</p>';
        document.getElementById('warning-flags-output').innerHTML = flagsHtml;
    }
    
    function updateMetricBar(metricName, value) {
        const percentage = Math.round(value * 100);
        const barElement = document.getElementById(`${metricName}-bar`);
        const textElement = document.getElementById(`${metricName}-text`);
        
        if (barElement && textElement) {
            barElement.style.width = `${percentage}%`;
            textElement.textContent = `${percentage}%`;
            
            // Add color coding
            if (percentage >= 70) {
                barElement.className = 'metric-fill metric-good';
            } else if (percentage >= 40) {
                barElement.className = 'metric-fill metric-medium';
            } else {
                barElement.className = 'metric-fill metric-poor';
            }
        }
    }
    
    function updateAudioAnalysisTab(data) {
        const prosody = data.prosody;
        
        if (!prosody) {
            document.getElementById('prosody-container').style.display = 'none';
            document.getElementById('no-audio-message').style.display = 'block';
            return;
        }
        
        document.getElementById('prosody-container').style.display = 'block';
        document.getElementById('no-audio-message').style.display = 'none';
        
        // Update sarcasm probability
        const sarcasmPercent = Math.round(prosody.sarcasm_probability * 100);
        document.getElementById('sarcasm-probability').textContent = `${sarcasmPercent}%`;
        
        // Update voice metrics
        document.getElementById('pitch-mean').textContent = prosody.pitch_mean.toFixed(1);
        document.getElementById('pitch-std').textContent = prosody.pitch_std.toFixed(1);
        document.getElementById('energy-mean').textContent = prosody.energy_mean.toFixed(3);
        document.getElementById('speaking-rate').textContent = prosody.speaking_rate.toFixed(1);
        document.getElementById('tempo').textContent = prosody.tempo.toFixed(1);
    }
    
    function updateSourcesTab(data) {
        const sources = data.sources || [];
        
        if (sources.length === 0) {
            document.getElementById('sources-output').innerHTML = '<p>No sources available</p>';
            return;
        }
        
        const sourcesHtml = sources.map(source => `
            <div class="source-item">
                <h5>${source.name}</h5>
                <p class="source-type">${source.type}</p>
                ${source.url ? `<a href="${source.url}" target="_blank" rel="noopener">View Source</a>` : ''}
                ${source.rating ? `<span class="source-rating">${source.rating}</span>` : ''}
            </div>
        `).join('');
        
        document.getElementById('sources-output').innerHTML = sourcesHtml;
    }
});
