# Enhanced Conspiracy Theory Debunker

A modern web application for fact-checking claims with advanced AI-powered analysis, audio processing, and real-time verification.

## Features

- **Advanced Fact-Checking**: Uses multiple reliable sources to verify claims
- **Audio Analysis**: Detects sarcasm, emotion, and audio authenticity through prosody analysis
- **Real-Time Visualization**: Interactive timeline and credibility radar charts
- **Enhanced Knowledge Base**: Combines Wikipedia, fact-checking datasets, and scientific literature
- **Modern UI**: Responsive design with intuitive user experience
- **Performance Optimized**: Caching, concurrency, and efficient resource utilization

## Technical Stack

- **Backend**: Python with Flask, LangChain, Transformers, and HuggingFace models
- **Frontend**: HTML5, CSS3, JavaScript with modern responsive design
- **AI Models**: GPT-4, MPNet Embeddings, Whisper-Small, RoBERTa Sentiment
- **Data Visualization**: Plotly for interactive charts and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/debunk.git
cd debunk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
- Modern UI: http://localhost:5000
- Gradio UI (legacy): http://localhost:7860

## Usage

1. Enter a text claim or upload an audio file containing a claim
2. Enable or disable advanced audio analysis as needed
3. Click "Analyze Claim" to process the input
4. View the results across different tabs:
   - Summary
   - Full Analysis
   - Timeline
   - Credibility Radar
   - Audio Analysis
   - Sources

## Performance Optimizations

- **Request Caching**: Repeated claims are served from cache
- **Concurrent Processing**: Multiple requests handled simultaneously
- **Efficient Model Loading**: Models loaded once and reused
- **Asynchronous Processing**: Long-running tasks don't block the UI
- **Optimized Knowledge Retrieval**: Fast vector search for relevant context

## License

MIT License
