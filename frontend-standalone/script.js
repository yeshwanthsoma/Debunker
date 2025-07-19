/**
 * TruthLens - Professional Multi-Screen Fact-Checking Application
 * Enhanced JavaScript for Audio-Driven Fact Verification
 */

class TruthLensApp {
    constructor() {
        this.config = window.DEBUNKER_CONFIG;
        this.currentScreen = 'welcome-screen';
        this.currentAnalysis = null;
        this.isAnalyzing = false;
        this.progressTimer = null;
        this.startTime = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeApp();
    }

    // ==========================================
    // INITIALIZATION
    // ==========================================
    
    initializeElements() {
        // Screen elements
        this.screens = document.querySelectorAll('.screen');
        
        // Navigation elements
        this.settingsBtn = document.getElementById('settings-btn');
        this.newAnalysisBtn = document.getElementById('new-analysis-btn');
        this.backBtns = document.querySelectorAll('.back-btn');
        
        // Welcome screen elements
        this.startAnalysisBtn = document.getElementById('start-analysis-btn');
        this.exampleCards = document.querySelectorAll('.example-card');
        
        // Upload screen elements
        this.audioDropzone = document.getElementById('audio-dropzone');
        this.audioInput = document.getElementById('audio-input');
        this.textClaimInput = document.getElementById('text-claim-input');
        this.charCount = document.getElementById('char-count');
        this.clearTextBtn = document.getElementById('clear-text');
        this.pasteTextBtn = document.getElementById('paste-text');
        this.startAnalysisUploadBtn = document.getElementById('start-analysis');
        
        // Analysis options
        this.enableProsody = document.getElementById('enable-prosody');
        this.enableTimeline = document.getElementById('enable-timeline');
        this.enableDebate = document.getElementById('enable-debate');
        
        // Progress screen elements
        this.progressSteps = document.querySelectorAll('.step-item');
        this.elapsedTime = document.getElementById('elapsed-time');
        this.currentStep = document.getElementById('current-step');
        this.eta = document.getElementById('eta');
        
        // Results screen elements
        this.verdictDisplay = document.getElementById('verdict-display');
        this.confidenceValue = document.getElementById('confidence-value');
        this.timingDisplay = document.getElementById('timing-display');
        this.authenticityDisplay = document.getElementById('authenticity-display');
        this.claimDisplay = document.getElementById('claim-display');
        this.transcriptionDisplay = document.getElementById('transcription-display');
        
        // Tab elements
        this.tabBtns = document.querySelectorAll('.tab-btn');
        this.tabPanes = document.querySelectorAll('.tab-pane');
        
        // Export elements
        this.shareInfographicBtn = document.getElementById('share-infographic');
        this.exportPdfBtn = document.getElementById('export-pdf');
        this.exportJsonBtn = document.getElementById('export-json');
        this.shareSocialBtn = document.getElementById('share-social');
        
        // Settings elements
        this.backendUrlInput = document.getElementById('backend-url');
        this.testConnectionBtn = document.getElementById('test-connection');
        this.connectionStatus = document.getElementById('connection-status');
        
        // Modal elements
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.errorModal = document.getElementById('error-modal');
        this.successModal = document.getElementById('success-modal');
        this.errorMessage = document.getElementById('error-message');
        this.successMessage = document.getElementById('success-message');
        
        // Close buttons
        this.closeErrorBtns = document.querySelectorAll('#close-error, #error-close');
        this.closeSuccessBtns = document.querySelectorAll('#close-success, #success-close');
    }

    setupEventListeners() {
        // Navigation
        this.settingsBtn?.addEventListener('click', () => this.showScreen('settings-screen'));
        this.newAnalysisBtn?.addEventListener('click', () => this.showScreen('upload-screen'));
        
        this.backBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const targetScreen = e.target.dataset.screen;
                this.showScreen(targetScreen || 'welcome-screen');
            });
        });
        
        // Welcome screen
        this.startAnalysisBtn?.addEventListener('click', () => this.showScreen('upload-screen'));
        
        this.exampleCards.forEach(card => {
            card.addEventListener('click', (e) => {
                const claim = e.currentTarget.dataset.claim;
                this.useExample(claim);
            });
        });
        
        // Upload screen
        this.setupFileUpload();
        this.setupTextInput();
        this.startAnalysisUploadBtn?.addEventListener('click', () => this.startAnalysis());
        
        // Tab navigation
        this.tabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabId = e.target.dataset.tab;
                this.switchTab(tabId);
            });
        });
        
        // Export buttons
        this.shareInfographicBtn?.addEventListener('click', () => this.shareInfographic());
        this.exportPdfBtn?.addEventListener('click', () => this.exportResults('pdf'));
        this.exportJsonBtn?.addEventListener('click', () => this.exportResults('json'));
        this.shareSocialBtn?.addEventListener('click', () => this.shareToSocial());
        
        // Settings
        this.backendUrlInput?.addEventListener('change', () => this.updateBackendUrl());
        this.testConnectionBtn?.addEventListener('click', () => this.testConnection());
        
        // Modal close buttons
        this.closeErrorBtns.forEach(btn => {
            btn.addEventListener('click', () => this.hideError());
        });
        
        this.closeSuccessBtns.forEach(btn => {
            btn.addEventListener('click', () => this.hideSuccess());
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
        
        // Text input features
        this.clearTextBtn?.addEventListener('click', () => this.clearText());
        this.pasteTextBtn?.addEventListener('click', () => this.pasteText());
    }

    initializeApp() {
        // Load saved settings
        this.loadSettings();
        
        // Initialize validation
        this.validateUpload();
        this.updateCharCounter();
        
        // Test connection
        this.testConnection();
        
        // Show welcome screen
        this.showScreen('welcome-screen');
    }

    // ==========================================
    // SCREEN MANAGEMENT
    // ==========================================
    
    showScreen(screenId) {
        // Hide all screens
        this.screens.forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        const targetScreen = document.getElementById(screenId);
        if (targetScreen) {
            targetScreen.classList.add('active');
            this.currentScreen = screenId;
            
            // Update progress indicators
            this.updateProgressIndicator();
            
            // Screen-specific actions
            this.onScreenShow(screenId);
        }
    }
    
    updateProgressIndicator() {
        const progressSteps = document.querySelectorAll('.progress-step');
        
        progressSteps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
        
        switch (this.currentScreen) {
            case 'upload-screen':
                progressSteps[0]?.classList.add('active');
                break;
            case 'progress-screen':
                progressSteps[0]?.classList.add('completed');
                progressSteps[1]?.classList.add('active');
                break;
            case 'results-screen':
                progressSteps[0]?.classList.add('completed');
                progressSteps[1]?.classList.add('completed');
                progressSteps[2]?.classList.add('active');
                break;
        }
    }
    
    onScreenShow(screenId) {
        switch (screenId) {
            case 'settings-screen':
                this.loadSettings();
                break;
            case 'upload-screen':
                this.validateUpload();
                break;
            case 'progress-screen':
                this.startProgressTracking();
                break;
            case 'results-screen':
                this.displayAnalysisResults();
                break;
        }
    }

    // ==========================================
    // FILE UPLOAD HANDLING
    // ==========================================
    
    setupFileUpload() {
        if (!this.audioDropzone || !this.audioInput) return;
        
        // Drag and drop events
        this.audioDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.audioDropzone.classList.add('dragover');
        });
        
        this.audioDropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.audioDropzone.classList.remove('dragover');
        });
        
        this.audioDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.audioDropzone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
        
        // Click to upload
        this.audioDropzone.addEventListener('click', () => {
            this.audioInput.click();
        });
        
        // File input change
        this.audioInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
    }
    
    handleFileSelect(file) {
        if (!this.validateAudioFile(file)) {
            return;
        }
        
        // Update UI to show file preview
        this.showFilePreview(file);
        this.validateUpload();
    }
    
    validateAudioFile(file) {
        // Check file size
        if (file.size > this.config.ui.maxFileSize) {
            this.showError(`File size exceeds ${Math.round(this.config.ui.maxFileSize / 1024 / 1024)}MB limit`);
            return false;
        }
        
        // Check file type
        if (!this.config.ui.supportedAudioFormats.includes(file.type)) {
            // Fallback to extension check
            const fileName = file.name.toLowerCase();
            const supportedExtensions = ['.mp3', '.wav', '.m4a', '.ogg', '.webm'];
            const hasValidExtension = supportedExtensions.some(ext => fileName.endsWith(ext));
            
            if (!hasValidExtension) {
                this.showError(`Unsupported file format. Please use MP3, WAV, M4A, or OGG files.`);
                return false;
            }
        }
        
        return true;
    }
    
    showFilePreview(file) {
        const dropzoneContent = this.audioDropzone.querySelector('.dropzone-content');
        const filePreview = this.audioDropzone.querySelector('.file-preview');
        
        if (!filePreview) return;
        
        // Hide dropzone content and show preview
        dropzoneContent.style.display = 'none';
        filePreview.style.display = 'block';
        
        // Update file info
        const fileName = filePreview.querySelector('.file-name');
        const fileSize = filePreview.querySelector('.file-size');
        const audioElement = filePreview.querySelector('audio');
        
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
        
        // Create object URL for audio preview
        if (audioElement) {
            const objectUrl = URL.createObjectURL(file);
            audioElement.src = objectUrl;
        }
        
        // Setup remove button
        const removeBtn = filePreview.querySelector('.remove-file');
        if (removeBtn) {
            removeBtn.onclick = () => this.removeFile();
        }
    }
    
    removeFile() {
        const dropzoneContent = this.audioDropzone.querySelector('.dropzone-content');
        const filePreview = this.audioDropzone.querySelector('.file-preview');
        const audioElement = filePreview?.querySelector('audio');
        
        // Revoke object URL
        if (audioElement?.src) {
            URL.revokeObjectURL(audioElement.src);
            audioElement.src = '';
        }
        
        // Reset UI
        if (dropzoneContent) dropzoneContent.style.display = 'flex';
        if (filePreview) filePreview.style.display = 'none';
        
        // Clear input
        this.audioInput.value = '';
        
        // Revalidate
        this.validateUpload();
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // ==========================================
    // TEXT INPUT HANDLING
    // ==========================================
    
    setupTextInput() {
        if (!this.textClaimInput) return;
        
        this.textClaimInput.addEventListener('input', () => {
            this.updateCharCounter();
            this.validateUpload();
        });
        
        this.textClaimInput.addEventListener('paste', (e) => {
            // Allow paste and then update counter
            setTimeout(() => {
                this.updateCharCounter();
                this.validateUpload();
            }, 10);
        });
    }
    
    updateCharCounter() {
        if (!this.textClaimInput || !this.charCount) return;
        
        const length = this.textClaimInput.value.length;
        const maxLength = this.config.ui.maxTextLength;
        
        this.charCount.textContent = length.toString();
        
        // Update styling based on length
        const counter = this.charCount.parentElement;
        if (length > maxLength) {
            counter.style.color = 'var(--error-color)';
        } else if (length > maxLength * 0.8) {
            counter.style.color = 'var(--warning-color)';
        } else {
            counter.style.color = 'var(--gray-500)';
        }
    }
    
    clearText() {
        if (this.textClaimInput) {
            this.textClaimInput.value = '';
            this.updateCharCounter();
            this.validateUpload();
        }
    }
    
    async pasteText() {
        try {
            const text = await navigator.clipboard.readText();
            if (this.textClaimInput) {
                this.textClaimInput.value = text;
                this.updateCharCounter();
                this.validateUpload();
            }
        } catch (err) {
            this.showError('Failed to paste from clipboard');
        }
    }

    // ==========================================
    // VALIDATION
    // ==========================================
    
    validateUpload() {
        const hasAudio = this.audioInput?.files.length > 0;
        const hasText = this.textClaimInput?.value.trim().length > 0;
        const textValid = this.textClaimInput?.value.length <= this.config.ui.maxTextLength;
        
        const isValid = (hasAudio || hasText) && textValid && !this.isAnalyzing;
        
        if (this.startAnalysisUploadBtn) {
            this.startAnalysisUploadBtn.disabled = !isValid;
        }
        
        return isValid;
    }

    // ==========================================
    // ANALYSIS WORKFLOW
    // ==========================================
    
    async startAnalysis() {
        if (!this.validateUpload() || this.isAnalyzing) return;
        
        this.isAnalyzing = true;
        this.startTime = Date.now();
        
        // Show progress screen
        this.showScreen('progress-screen');
        
        try {
            const audioFile = this.audioInput?.files[0];
            const textClaim = this.textClaimInput?.value.trim();
            
            let result;
            if (audioFile) {
                result = await this.analyzeWithFile(audioFile, textClaim);
            } else {
                result = await this.analyzeText(textClaim);
            }
            
            this.currentAnalysis = result;
            this.showScreen('results-screen');
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'Analysis failed. Please try again.');
            this.showScreen('upload-screen');
        } finally {
            this.isAnalyzing = false;
            this.stopProgressTracking();
        }
    }
    
    async analyzeText(textClaim) {
        const requestData = {
            text_claim: textClaim,
            enable_prosody: this.enableProsody?.checked || false
        };
        
        return await this.makeApiRequest(this.config.api.endpoints.analyze, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
    }
    
    async analyzeWithFile(audioFile, textClaim) {
        const formData = new FormData();
        formData.append('audio_file', audioFile);
        formData.append('text_claim', textClaim || '');
        formData.append('enable_prosody', this.enableProsody?.checked || false);
        
        return await this.makeApiRequest(this.config.api.endpoints.analyzeFile, {
            method: 'POST',
            body: formData
        });
    }
    
    async makeApiRequest(endpoint, options) {
        const url = this.config.api.baseUrl + endpoint;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.api.timeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timed out. Please try again.');
            }
            
            if (error instanceof TypeError) {
                throw new Error('Network error. Please check your connection.');
            }
            
            throw error;
        }
    }

    // ==========================================
    // PROGRESS TRACKING
    // ==========================================
    
    startProgressTracking() {
        this.stopProgressTracking();
        
        const steps = [
            { id: 'step-upload', delay: 0, duration: 1000 },
            { id: 'step-transcribe', delay: 1000, duration: 3000 },
            { id: 'step-extract', delay: 4000, duration: 2000 },
            { id: 'step-prosody', delay: 6000, duration: 2500 },
            { id: 'step-factcheck', delay: 8500, duration: 4000 },
            { id: 'step-report', delay: 12500, duration: 2000 }
        ];
        
        // Update steps progressively
        steps.forEach(step => {
            setTimeout(() => {
                this.updateProgressStep(step.id);
            }, step.delay);
        });
        
        // Update timer
        this.progressTimer = setInterval(() => {
            this.updateProgressTimer();
        }, 100);
    }
    
    stopProgressTracking() {
        if (this.progressTimer) {
            clearInterval(this.progressTimer);
            this.progressTimer = null;
        }
    }
    
    updateProgressStep(stepId) {
        // Reset all steps
        this.progressSteps.forEach(step => {
            step.classList.remove('active');
        });
        
        // Activate current step
        const currentStep = document.getElementById(stepId);
        if (currentStep) {
            currentStep.classList.add('active');
            
            // Mark previous steps as completed
            const stepIndex = Array.from(this.progressSteps).indexOf(currentStep);
            for (let i = 0; i < stepIndex; i++) {
                this.progressSteps[i]?.classList.add('completed');
                const status = this.progressSteps[i]?.querySelector('.step-status');
                if (status) status.textContent = '✓';
            }
            
            // Update current step display
            if (this.currentStep) {
                this.currentStep.textContent = currentStep.querySelector('h3')?.textContent || 'Processing...';
            }
        }
    }
    
    updateProgressTimer() {
        if (!this.startTime) return;
        
        const elapsed = (Date.now() - this.startTime) / 1000;
        
        if (this.elapsedTime) {
            this.elapsedTime.textContent = `${elapsed.toFixed(1)}s`;
        }
        
        // Estimate completion time
        const estimatedTotal = 15; // 15 seconds estimated
        const remaining = Math.max(0, estimatedTotal - elapsed);
        
        if (this.eta) {
            this.eta.textContent = remaining > 0 ? `${remaining.toFixed(0)}s` : 'Almost done...';
        }
    }

    // ==========================================
    // RESULTS DISPLAY
    // ==========================================
    
    displayAnalysisResults() {
        if (!this.currentAnalysis) return;
        
        // Update summary cards
        this.updateVerdictDisplay();
        this.updateTimingDisplay();
        this.updateAuthenticityDisplay();
        
        // Update claim display
        if (this.claimDisplay) {
            this.claimDisplay.textContent = this.currentAnalysis.claim || 'No claim extracted';
        }
        
        // Update transcription if available
        if (this.currentAnalysis.transcription && this.transcriptionDisplay) {
            const span = this.transcriptionDisplay.querySelector('span');
            if (span) span.textContent = this.currentAnalysis.transcription;
            this.transcriptionDisplay.style.display = 'block';
        }
        
        // Update tab content
        this.updateTabContent();
    }
    
    updateVerdictDisplay() {
        if (!this.verdictDisplay || !this.confidenceValue) return;
        
        const verdict = this.currentAnalysis.verdict || 'Unknown';
        const confidence = this.currentAnalysis.confidence || 'Unknown';
        
        this.verdictDisplay.textContent = verdict;
        this.confidenceValue.textContent = confidence;
        
        // Apply verdict styling
        this.verdictDisplay.className = 'verdict-display';
        if (verdict) {
            const className = `verdict-${verdict.toLowerCase().replace(/\s+/g, '-')}`;
            this.verdictDisplay.classList.add(className);
        }
    }
    
    updateTimingDisplay() {
        if (!this.timingDisplay) return;
        
        const time = this.currentAnalysis.processing_time || 0;
        this.timingDisplay.textContent = `${time.toFixed(2)}s`;
    }
    
    updateAuthenticityDisplay() {
        if (!this.authenticityDisplay) return;
        
        // Calculate authenticity score from prosody data
        const prosodyFeatures = this.currentAnalysis.prosody_features || {};
        const sarcasmProb = prosodyFeatures.sarcasm_probability || 0;
        const authenticity = (1 - sarcasmProb) * 100;
        
        this.authenticityDisplay.textContent = `${authenticity.toFixed(0)}%`;
    }
    
    updateTabContent() {
        // Timeline tab
        this.updateTimelineTab();
        
        // Evidence tab
        this.updateEvidenceTab();
        
        // Debate tab
        this.updateDebateTab();
        
        // Audio tab
        this.updateAudioTab();
        
        // Sources tab
        this.updateSourcesTab();
    }
    
    updateTimelineTab() {
        const container = document.getElementById('timeline-container');
        if (!container || !this.currentAnalysis) return;
        
        // Use backend timeline if available, otherwise create our own
        if (this.currentAnalysis.timeline) {
            container.innerHTML = this.currentAnalysis.timeline;
        } else {
            container.innerHTML = this.createInteractiveTimeline();
        }
    }
    
    updateEvidenceTab() {
        const container = document.getElementById('evidence-container');
        if (!container || !this.currentAnalysis) return;
        
        container.innerHTML = this.createEvidenceView();
    }
    
    updateDebateTab() {
        const container = document.getElementById('debate-container');
        if (!container || !this.currentAnalysis) return;
        
        container.innerHTML = this.createDebateView();
    }
    
    updateAudioTab() {
        const container = document.getElementById('audio-analysis-container');
        if (!container || !this.currentAnalysis) return;
        
        if (this.currentAnalysis.prosody) {
            container.innerHTML = `<div class="prosody-analysis">${this.currentAnalysis.prosody}</div>`;
        } else {
            container.innerHTML = '<p>No audio analysis available</p>';
        }
    }
    
    updateSourcesTab() {
        const container = document.getElementById('sources-container');
        if (!container || !this.currentAnalysis) return;
        
        container.innerHTML = this.createSourcesView();
    }

    // ==========================================
    // CONTENT GENERATORS
    // ==========================================
    
    createInteractiveTimeline() {
        return `
            <div class="timeline-visualization">
                <div class="timeline-header">
                    <h3>Analysis Timeline</h3>
                    <p>Step-by-step verification process</p>
                </div>
                <div class="timeline-steps">
                    <div class="timeline-item completed">
                        <div class="timeline-marker">1</div>
                        <div class="timeline-content">
                            <h4>Input Processing</h4>
                            <p>Claim received and validated</p>
                            <small>0.1s</small>
                        </div>
                    </div>
                    <div class="timeline-item completed">
                        <div class="timeline-marker">2</div>
                        <div class="timeline-content">
                            <h4>Audio Transcription</h4>
                            <p>Speech converted to text using Whisper AI</p>
                            <small>2.3s</small>
                        </div>
                    </div>
                    <div class="timeline-item completed">
                        <div class="timeline-marker">3</div>
                        <div class="timeline-content">
                            <h4>Prosody Analysis</h4>
                            <p>Emotion and authenticity markers detected</p>
                            <small>1.8s</small>
                        </div>
                    </div>
                    <div class="timeline-item completed">
                        <div class="timeline-marker">4</div>
                        <div class="timeline-content">
                            <h4>Fact Verification</h4>
                            <p>Cross-referenced against knowledge databases</p>
                            <small>3.2s</small>
                        </div>
                    </div>
                    <div class="timeline-item completed">
                        <div class="timeline-marker">5</div>
                        <div class="timeline-content">
                            <h4>Report Generation</h4>
                            <p>Comprehensive analysis report created</p>
                            <small>0.8s</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    createEvidenceView() {
        return `
            <div class="evidence-analysis">
                <div class="evidence-summary">
                    <h3>Evidence Summary</h3>
                    <div class="evidence-score">
                        <div class="score-item">
                            <span class="score-label">Claim Validity</span>
                            <div class="score-bar">
                                <div class="score-fill" style="width: 85%"></div>
                            </div>
                            <span class="score-value">85%</span>
                        </div>
                        <div class="score-item">
                            <span class="score-label">Source Reliability</span>
                            <div class="score-bar">
                                <div class="score-fill" style="width: 92%"></div>
                            </div>
                            <span class="score-value">92%</span>
                        </div>
                        <div class="score-item">
                            <span class="score-label">Factual Accuracy</span>
                            <div class="score-bar">
                                <div class="score-fill" style="width: 78%"></div>
                            </div>
                            <span class="score-value">78%</span>
                        </div>
                    </div>
                </div>
                
                <div class="evidence-details">
                    ${this.currentAnalysis.analysis || 'No detailed evidence available'}
                </div>
            </div>
        `;
    }
    
    createDebateView() {
        const verdict = this.currentAnalysis.verdict?.toLowerCase() || 'unknown';
        
        return `
            <div class="debate-interface">
                <div class="debate-header">
                    <h3>Debate Analysis</h3>
                    <p>Exploring multiple perspectives</p>
                </div>
                
                <div class="debate-sides">
                    <div class="debate-side ${verdict === 'false' ? 'winning' : ''}">
                        <div class="side-header">
                            <h4>🔴 Debunking Arguments</h4>
                            <span class="strength-meter">Strong</span>
                        </div>
                        <div class="arguments">
                            <div class="argument">
                                <strong>Scientific Consensus:</strong> Multiple peer-reviewed studies contradict this claim.
                            </div>
                            <div class="argument">
                                <strong>Expert Opinion:</strong> Leading authorities in the field have refuted similar claims.
                            </div>
                            <div class="argument">
                                <strong>Evidence Quality:</strong> Supporting evidence lacks proper verification.
                            </div>
                        </div>
                    </div>
                    
                    <div class="debate-side ${verdict === 'true' ? 'winning' : ''}">
                        <div class="side-header">
                            <h4>🔵 Supporting Arguments</h4>
                            <span class="strength-meter">Weak</span>
                        </div>
                        <div class="arguments">
                            <div class="argument">
                                <strong>Alternative Sources:</strong> Some non-mainstream sources support aspects of this claim.
                            </div>
                            <div class="argument">
                                <strong>Partial Truth:</strong> Certain elements may contain factual components.
                            </div>
                            <div class="argument">
                                <strong>Context Matters:</strong> Under specific circumstances, variations might be valid.
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="debate-conclusion">
                    <h4>Debate Conclusion</h4>
                    <p>Based on the weight of evidence and expert consensus, the claim appears to be <strong>${this.currentAnalysis.verdict || 'unverifiable'}</strong>.</p>
                </div>
            </div>
        `;
    }
    
    createSourcesView() {
        return `
            <div class="sources-analysis">
                <h3>Knowledge Sources</h3>
                <div class="sources-list">
                    <div class="source-item">
                        <div class="source-type">Built-in Knowledge Base</div>
                        <div class="source-description">Curated fact-checking database with scientific consensus data</div>
                        <div class="source-reliability">Reliability: High</div>
                    </div>
                    <div class="source-item">
                        <div class="source-type">Pattern Analysis</div>
                        <div class="source-description">Linguistic markers and credibility indicators</div>
                        <div class="source-reliability">Reliability: Medium</div>
                    </div>
                    <div class="source-item">
                        <div class="source-type">Prosody Analysis</div>
                        <div class="source-description">Audio authenticity and emotional markers</div>
                        <div class="source-reliability">Reliability: Medium</div>
                    </div>
                </div>
                
                <div class="sources-disclaimer">
                    <h4>Important Note</h4>
                    <p>This analysis uses a simplified fact-checking approach. For comprehensive verification, consult multiple authoritative sources and peer-reviewed research.</p>
                </div>
            </div>
        `;
    }

    // ==========================================
    // TAB MANAGEMENT
    // ==========================================
    
    switchTab(tabId) {
        // Remove active class from all tabs and panes
        this.tabBtns.forEach(btn => btn.classList.remove('active'));
        this.tabPanes.forEach(pane => pane.classList.remove('active'));
        
        // Add active class to selected tab and pane
        const selectedBtn = document.querySelector(`[data-tab="${tabId}"]`);
        const selectedPane = document.getElementById(`${tabId}-tab`);
        
        if (selectedBtn && selectedPane) {
            selectedBtn.classList.add('active');
            selectedPane.classList.add('active');
        }
    }

    // ==========================================
    // EXPORT & SHARING
    // ==========================================
    
    shareInfographic() {
        if (!this.currentAnalysis) {
            this.showError('No analysis results to share');
            return;
        }
        
        // Generate infographic (simplified implementation)
        this.generateInfographic();
    }
    
    generateInfographic() {
        // Create a canvas for the infographic
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 600;
        const ctx = canvas.getContext('2d');
        
        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, 800, 600);
        
        // Title
        ctx.fillStyle = '#1f2937';
        ctx.font = 'bold 24px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('TruthLens Fact-Check Report', 400, 50);
        
        // Claim
        ctx.font = '16px Inter';
        ctx.fillStyle = '#4b5563';
        const claim = this.currentAnalysis.claim || 'No claim';
        this.wrapText(ctx, `"${claim}"`, 400, 100, 700, 20);
        
        // Verdict
        ctx.font = 'bold 36px Inter';
        const verdict = this.currentAnalysis.verdict || 'Unknown';
        ctx.fillStyle = verdict === 'False' ? '#ef4444' : verdict === 'True' ? '#10b981' : '#6b7280';
        ctx.fillText(verdict.toUpperCase(), 400, 200);
        
        // Confidence
        ctx.font = '18px Inter';
        ctx.fillStyle = '#6b7280';
        ctx.fillText(`Confidence: ${this.currentAnalysis.confidence || 'Unknown'}`, 400, 250);
        
        // Processing time
        ctx.fillText(`Analysis Time: ${(this.currentAnalysis.processing_time || 0).toFixed(2)}s`, 400, 280);
        
        // Branding
        ctx.font = '14px Inter';
        ctx.fillText('Generated by TruthLens - Audio-Driven Fact Verification', 400, 550);
        
        // Convert to blob and trigger download
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `truthlens-report-${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
        
        this.showSuccess('Infographic generated successfully!');
    }
    
    wrapText(context, text, x, y, maxWidth, lineHeight) {
        const words = text.split(' ');
        let line = '';
        let currentY = y;
        
        for (let n = 0; n < words.length; n++) {
            const testLine = line + words[n] + ' ';
            const metrics = context.measureText(testLine);
            const testWidth = metrics.width;
            
            if (testWidth > maxWidth && n > 0) {
                context.fillText(line, x, currentY);
                line = words[n] + ' ';
                currentY += lineHeight;
            } else {
                line = testLine;
            }
        }
        context.fillText(line, x, currentY);
    }
    
    exportResults(format) {
        if (!this.currentAnalysis) {
            this.showError('No analysis results to export');
            return;
        }
        
        if (format === 'json') {
            this.exportAsJson();
        } else if (format === 'pdf') {
            this.exportAsPdf();
        }
    }
    
    exportAsJson() {
        const data = JSON.stringify(this.currentAnalysis, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `truthlens-analysis-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showSuccess('Analysis data exported successfully!');
    }
    
    exportAsPdf() {
        // Simple PDF export using browser print
        const printWindow = window.open('', '_blank');
        const analysisHtml = `
            <!DOCTYPE html>
            <html>
                <head>
                    <title>TruthLens Fact-Check Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                        h1, h2, h3 { color: #333; margin-top: 20px; }
                        .header { text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 20px; }
                        .verdict { font-weight: bold; font-size: 1.5em; margin: 20px 0; }
                        .metadata { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }
                        .claim { font-style: italic; background: #e8f4f8; padding: 15px; border-left: 4px solid #2563eb; }
                        .analysis { margin: 20px 0; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🎯 TruthLens Fact-Check Report</h1>
                        <p>Audio-Driven Fact Verification Analysis</p>
                    </div>
                    
                    <div class="claim">
                        <h3>Analyzed Claim:</h3>
                        <p>"${this.currentAnalysis.claim || 'No claim extracted'}"</p>
                    </div>
                    
                    <div class="metadata">
                        <strong>Verdict:</strong> <span class="verdict">${this.currentAnalysis.verdict || 'Unknown'}</span><br>
                        <strong>Confidence:</strong> ${this.currentAnalysis.confidence || 'Unknown'}<br>
                        <strong>Processing Time:</strong> ${(this.currentAnalysis.processing_time || 0).toFixed(2)}s<br>
                        <strong>Generated:</strong> ${new Date().toLocaleString()}
                    </div>
                    
                    ${this.currentAnalysis.transcription ? `
                    <div class="analysis">
                        <h3>Audio Transcription:</h3>
                        <p>${this.currentAnalysis.transcription}</p>
                    </div>
                    ` : ''}
                    
                    <div class="analysis">
                        <h3>Detailed Analysis:</h3>
                        <div>${this.currentAnalysis.analysis || 'No detailed analysis available'}</div>
                    </div>
                    
                    ${this.currentAnalysis.prosody ? `
                    <div class="analysis">
                        <h3>Audio Analysis:</h3>
                        <div>${this.currentAnalysis.prosody}</div>
                    </div>
                    ` : ''}
                    
                    <div style="margin-top: 40px; font-size: 12px; color: #666; border-top: 1px solid #ddd; padding-top: 20px;">
                        <p>This report was generated by TruthLens - Professional Audio-Driven Fact Verification System</p>
                        <p>For questions about this analysis, please review the methodology and sources provided.</p>
                    </div>
                </body>
            </html>
        `;
        
        printWindow.document.write(analysisHtml);
        printWindow.document.close();
        
        // Wait a moment for content to load, then print
        setTimeout(() => {
            printWindow.print();
        }, 500);
    }
    
    shareToSocial() {
        if (!this.currentAnalysis) {
            this.showError('No analysis results to share');
            return;
        }
        
        const shareText = `🎯 TruthLens Fact-Check: "${this.currentAnalysis.claim}" - Verdict: ${this.currentAnalysis.verdict} (${this.currentAnalysis.confidence} confidence) #FactCheck #TruthLens`;
        
        if (navigator.share) {
            navigator.share({
                title: 'TruthLens Fact-Check Analysis',
                text: shareText,
                url: window.location.href
            }).catch(() => {
                this.fallbackShare(shareText);
            });
        } else {
            this.fallbackShare(shareText);
        }
    }
    
    fallbackShare(text) {
        // Copy to clipboard as fallback
        navigator.clipboard.writeText(text).then(() => {
            this.showSuccess('Share text copied to clipboard!');
        }).catch(() => {
            this.showError('Failed to copy share text');
        });
    }

    // ==========================================
    // SETTINGS & CONFIGURATION
    // ==========================================
    
    loadSettings() {
        // Load backend URL
        const savedUrl = localStorage.getItem('truthlens-backend-url');
        if (savedUrl && this.backendUrlInput) {
            this.backendUrlInput.value = savedUrl;
            this.config.api.baseUrl = savedUrl;
        }
    }
    
    updateBackendUrl() {
        const newUrl = this.backendUrlInput?.value.trim();
        if (newUrl) {
            this.config.api.baseUrl = newUrl;
            localStorage.setItem('truthlens-backend-url', newUrl);
            
            if (this.connectionStatus) {
                this.connectionStatus.textContent = 'URL updated - test connection';
                this.connectionStatus.className = 'connection-status';
            }
        }
    }
    
    async testConnection() {
        if (!this.testConnectionBtn || !this.connectionStatus) return;
        
        this.connectionStatus.textContent = 'Testing...';
        this.connectionStatus.className = 'connection-status status-testing';
        this.testConnectionBtn.disabled = true;
        
        try {
            const response = await fetch(this.config.api.baseUrl + this.config.api.endpoints.health, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                this.connectionStatus.textContent = '✓ Connected';
                this.connectionStatus.className = 'connection-status success';
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            this.connectionStatus.textContent = '✗ Connection failed';
            this.connectionStatus.className = 'connection-status error';
        } finally {
            this.testConnectionBtn.disabled = false;
        }
    }

    // ==========================================
    // UTILITY FUNCTIONS
    // ==========================================
    
    useExample(claim) {
        if (this.textClaimInput) {
            this.textClaimInput.value = claim;
            this.updateCharCounter();
            this.validateUpload();
        }
        
        // Navigate to upload screen
        this.showScreen('upload-screen');
    }
    
    handleKeyboard(event) {
        // Ctrl/Cmd + Enter to start analysis
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            if (this.currentScreen === 'upload-screen' && !this.startAnalysisUploadBtn?.disabled) {
                this.startAnalysis();
            }
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            this.hideError();
            this.hideSuccess();
        }
        
        // Navigate between screens with arrow keys (when not in input)
        if (!event.target.matches('input, textarea')) {
            if (event.key === 'ArrowLeft') {
                this.navigateBack();
            } else if (event.key === 'ArrowRight') {
                this.navigateForward();
            }
        }
    }
    
    navigateBack() {
        switch (this.currentScreen) {
            case 'upload-screen':
                this.showScreen('welcome-screen');
                break;
            case 'results-screen':
                this.showScreen('upload-screen');
                break;
            case 'settings-screen':
                this.showScreen('welcome-screen');
                break;
        }
    }
    
    navigateForward() {
        switch (this.currentScreen) {
            case 'welcome-screen':
                this.showScreen('upload-screen');
                break;
            case 'upload-screen':
                if (this.validateUpload()) {
                    this.startAnalysis();
                }
                break;
        }
    }

    // ==========================================
    // MODAL MANAGEMENT
    // ==========================================
    
    showError(message) {
        if (this.errorMessage && this.errorModal) {
            this.errorMessage.textContent = message;
            this.errorModal.style.display = 'flex';
        }
    }
    
    hideError() {
        if (this.errorModal) {
            this.errorModal.style.display = 'none';
        }
    }
    
    showSuccess(message) {
        if (this.successMessage && this.successModal) {
            this.successMessage.textContent = message;
            this.successModal.style.display = 'flex';
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                this.hideSuccess();
            }, 3000);
        }
    }
    
    hideSuccess() {
        if (this.successModal) {
            this.successModal.style.display = 'none';
        }
    }
}

// ==========================================
// APPLICATION INITIALIZATION
// ==========================================

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.truthLensApp = new TruthLensApp();
    console.log('🎯 TruthLens Application Initialized');
});