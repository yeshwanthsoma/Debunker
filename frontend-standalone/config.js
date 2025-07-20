/**
 * Configuration file for the Enhanced Conspiracy Theory Debunker Frontend
 */

// Default configuration
const CONFIG = {
    // Backend API configuration
    api: {
        baseUrl: 'http://localhost:8080',
        endpoints: {
            analyze: '/api/analyze',
            analyzeFile: '/api/analyze-file',
            health: '/health',
            stats: '/api/stats'
        },
        timeout: 120000, // 2 minutes timeout for analysis
        retries: 3
    },
    
    // UI configuration
    ui: {
        maxFileSize: 50 * 1024 * 1024, // 50MB max file size
        supportedAudioFormats: [
            'audio/mp3', 'audio/mpeg', 'audio/mpeg3', 'audio/x-mpeg-3',  // MP3 variants
            'audio/wav', 'audio/x-wav', 'audio/wave',                     // WAV variants  
            'audio/m4a', 'audio/mp4', 'audio/x-m4a',                     // M4A variants
            'audio/ogg', 'audio/vorbis',                                  // OGG variants
            'audio/webm'                                                  // WebM
        ],
        maxTextLength: 2000,
        animationDuration: 300
    },
    
    // Feature flags
    features: {
        audioAnalysis: true,
        exportFunctionality: true,
        shareResults: true,
        apiConfiguration: true
    },
    
    // Default settings
    defaults: {
        enableProsody: true,
        autoSaveResults: false,
        darkMode: false
    },
    
    // Error messages
    messages: {
        noInput: 'Please provide an audio file or enter a text claim.',
        fileTooLarge: 'File size exceeds the maximum limit of 50MB.',
        unsupportedFormat: 'Unsupported audio format. Please use MP3, WAV, M4A, or OGG.',
        networkError: 'Network error. Please check your connection and try again.',
        serverError: 'Server error. Please try again later.',
        analysisError: 'Analysis failed. Please check your input and try again.',
        connectionFailed: 'Failed to connect to the backend API.',
        textTooLong: 'Text exceeds the maximum length of 2000 characters.'
    }
};

// Environment-specific overrides
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Development environment
    CONFIG.api.baseUrl = 'http://localhost:8080';
} else if (window.location.hostname.includes('staging')) {
    // Staging environment
    CONFIG.api.baseUrl = 'https://staging-api.debunker.example.com';
} else {
    // Production environment
    CONFIG.api.baseUrl = 'https://api.debunker.example.com';
}

// Load saved configuration from localStorage
function loadSavedConfig() {
    try {
        const savedConfig = localStorage.getItem('debunker-config');
        if (savedConfig) {
            const parsed = JSON.parse(savedConfig);
            // Merge with default config
            Object.assign(CONFIG, parsed);
        }
    } catch (error) {
        console.warn('Failed to load saved configuration:', error);
    }
}

// Save configuration to localStorage
function saveConfig() {
    try {
        localStorage.setItem('debunker-config', JSON.stringify(CONFIG));
    } catch (error) {
        console.warn('Failed to save configuration:', error);
    }
}

// Update configuration
function updateConfig(path, value) {
    const keys = path.split('.');
    let current = CONFIG;
    
    for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) {
            current[keys[i]] = {};
        }
        current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    saveConfig();
}

// Get configuration value
function getConfig(path, defaultValue = null) {
    const keys = path.split('.');
    let current = CONFIG;
    
    for (const key of keys) {
        if (current[key] === undefined) {
            return defaultValue;
        }
        current = current[key];
    }
    
    return current;
}

// Initialize configuration
loadSavedConfig();

// Export configuration for use in other scripts
window.DEBUNKER_CONFIG = CONFIG;
window.updateConfig = updateConfig;
window.getConfig = getConfig;
window.saveConfig = saveConfig;
