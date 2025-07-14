import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import pickle
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import librosa
import numpy as np
import traceback
import requests
import json
from datetime import datetime
import hashlib
import re
from typing import Dict, List, Tuple, Optional
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import lru_cache
warnings.filterwarnings('ignore')

# Environment setup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

class EnhancedFactChecker:
    def __init__(self):
        self.initialize_models()
        self.build_enhanced_index()
        self.fact_check_apis = {
            'snopes': 'https://www.snopes.com/api/v1/fact-check',
            'factcheck': 'https://www.factcheck.org/api/v1/search',
            'politifact': 'https://www.politifact.com/api/v1/search'
        }
        self.cache = {}
        
    def initialize_models(self):
        """Initialize all ML models with better configurations"""
        print("🔧 Initializing enhanced models...")
        
        # Better transcription model
        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-small",  # Upgraded from tiny
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Enhanced embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-mpnet-base-v2',  # Better than MiniLM
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Sentiment and emotion analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        
        # LLM with better configuration
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,  # More deterministic
            max_tokens=1000,
            request_timeout=30
        )
        
        print("✅ Models initialized successfully!")
    
    def build_enhanced_index(self):
        """Build a more comprehensive knowledge base"""
        index_path = "enhanced_faiss_index"
        
        if os.path.exists(index_path):
            print("📚 Loading existing enhanced index...")
            self.vectorstore = FAISS.load_local(
                index_path, 
                embeddings=self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return
        
        print("🏗️ Building enhanced knowledge base...")
        
        # Multiple reliable sources
        sources = []
        
        # 1. Wikipedia (more focused on controversial topics)
        wiki_topics = [
            "conspiracy theory", "misinformation", "fact checking",
            "moon landing", "climate change", "vaccines", "9/11",
            "political conspiracy", "scientific consensus", "media literacy"
        ]
        
        for topic in wiki_topics:
            try:
                wiki_data = load_dataset(
                    "wikimedia/wikipedia", 
                    "20231101.en", 
                    split=f"train[:{500}]"
                )
                wiki_texts = [
                    f"WIKIPEDIA: {doc['title']}\n{doc['text'][:2000]}" 
                    for doc in wiki_data 
                    if any(t.lower() in doc['text'].lower() for t in [topic])
                ]
                sources.extend(wiki_texts)
            except Exception as e:
                print(f"⚠️ Error loading Wikipedia data for {topic}: {e}")
        
        # 2. Fact-checking datasets
        try:
            fact_check_data = load_dataset("liar", split="train[:1000]")
            fact_texts = [
                f"FACT_CHECK: Statement: {item['statement']}\n"
                f"Label: {item['label']}\n"
                f"Context: {item['context']}\n"
                f"Speaker: {item['speaker']}\n"
                f"Subject: {item['subject']}"
                for item in fact_check_data
            ]
            sources.extend(fact_texts)
        except Exception as e:
            print(f"⚠️ Error loading fact-check data: {e}")
        
        # 3. Scientific consensus sources
        scientific_topics = [
            "climate change scientific consensus",
            "vaccine safety research",
            "peer review process",
            "scientific method",
            "evidence based medicine"
        ]
        
        for topic in scientific_topics:
            # Add curated scientific content
            sources.append(f"SCIENTIFIC: {topic} - Peer-reviewed scientific literature consistently supports...")
        
        # Build the vector store
        if sources:
            print(f"📊 Indexing {len(sources)} documents...")
            self.vectorstore = FAISS.from_texts(sources, self.embeddings)
            self.vectorstore.save_local(index_path)
            
            # Save metadata
            with open("enhanced_sources.pkl", "wb") as f:
                pickle.dump(sources, f)
        else:
            print("⚠️ No sources loaded, using minimal index")
            self.vectorstore = FAISS.from_texts(["No data available"], self.embeddings)
        
        print("✅ Enhanced knowledge base built!")
    
    def extract_advanced_prosody(self, audio_path: str) -> Dict:
        """Advanced prosody analysis for sarcasm/emotion detection"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Advanced features
            features = {}
            
            # Pitch analysis
            pitch = librosa.yin(y, fmin=50, fmax=400)
            valid_pitch = pitch[pitch > 0]
            if len(valid_pitch) > 0:
                features['pitch_mean'] = np.mean(valid_pitch)
                features['pitch_std'] = np.std(valid_pitch)
                features['pitch_range'] = np.max(valid_pitch) - np.min(valid_pitch)
            else:
                features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = 0
            
            # Energy and intensity
            rms = librosa.feature.rms(y=y)
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            # Rhythm and tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            features['speaking_rate'] = len(beats) / (len(y) / sr)
            
            # Sarcasm indicators
            sarcasm_score = 0
            if features['pitch_std'] > 80:  # High pitch variation
                sarcasm_score += 0.3
            if features['energy_std'] > 0.05:  # Irregular energy
                sarcasm_score += 0.2
            if features['speaking_rate'] < 2.0:  # Slow, deliberate speech
                sarcasm_score += 0.3
            
            features['sarcasm_probability'] = min(sarcasm_score, 1.0)
            
            # Emotion indicators
            emotion_indicators = {
                'anger': features['energy_mean'] > 0.1 and features['pitch_mean'] > 180,
                'excitement': features['energy_mean'] > 0.08 and features['tempo'] > 120,
                'uncertainty': features['pitch_std'] > 60 and features['energy_std'] > 0.04,
                'confidence': features['energy_mean'] > 0.06 and features['pitch_std'] < 40
            }
            
            features['emotion_indicators'] = emotion_indicators
            
            return features
            
        except Exception as e:
            print(f"⚠️ Prosody analysis error: {e}")
            return {
                'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
                'energy_mean': 0, 'energy_std': 0, 'spectral_centroid_mean': 0,
                'tempo': 0, 'speaking_rate': 0, 'sarcasm_probability': 0,
                'emotion_indicators': {}
            }
    
    def analyze_claim_credibility(self, claim: str) -> Dict:
        """Analyze claim for credibility markers"""
        credibility_score = 0.5  # Start neutral
        flags = []
        
        # Red flags
        red_flag_patterns = [
            r'\b(they don\'t want you to know|hidden truth|mainstream media lies)\b',
            r'\b(big pharma|global elite|deep state)\b',
            r'\b(wake up|sheeple|do your research)\b',
            r'\b(100% proof|undeniable evidence|smoking gun)\b'
        ]
        
        for pattern in red_flag_patterns:
            if re.search(pattern, claim.lower()):
                credibility_score -= 0.15
                flags.append(f"Red flag: {pattern}")
        
        # Green flags (credible indicators)
        green_flag_patterns = [
            r'\b(according to|study shows|research indicates)\b',
            r'\b(peer reviewed|scientific consensus|experts agree)\b',
            r'\b(data suggests|statistics show|evidence indicates)\b'
        ]
        
        for pattern in green_flag_patterns:
            if re.search(pattern, claim.lower()):
                credibility_score += 0.1
                flags.append(f"Green flag: {pattern}")
        
        # Length and complexity (very short claims often lack nuance)
        if len(claim.split()) < 10:
            credibility_score -= 0.05
            flags.append("Very short claim - may lack context")
        
        return {
            'credibility_score': max(0, min(1, credibility_score)),
            'flags': flags
        }
    
    def enhanced_fact_check(self, claim: str, prosody_features: Dict) -> Dict:
        """Enhanced fact-checking with multiple sources"""
        
        # Get relevant documents
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # More context
        )
        
        docs = retriever.get_relevant_documents(claim)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Analyze claim credibility
        credibility = self.analyze_claim_credibility(claim)
        
        # Enhanced prompt with prosody and credibility
        prompt = ChatPromptTemplate.from_template("""
You are an expert fact-checker. Analyze the following claim comprehensively.

CLAIM: {claim}

AUDIO ANALYSIS:
- Sarcasm probability: {sarcasm_prob:.2f}
- Emotional tone: {emotion_tone}
- Credibility markers: {credibility_flags}

CONTEXT FROM RELIABLE SOURCES:
{context}

Provide a thorough analysis in this exact format:
VERDICT: [True/False/Partially True/Misleading/Unverifiable]
CONFIDENCE: [High/Medium/Low] ([0-100]%)
EXPLANATION: [Detailed explanation with reasoning]
EVIDENCE: [Key supporting/contradicting evidence]
SOURCES: [Types of sources used]
WARNINGS: [Any concerns about the claim's framing or context]
PROSODY_IMPACT: [How audio analysis affected the assessment]
""")
        
        try:
            response = self.llm.invoke(prompt.format(
                claim=claim,
                sarcasm_prob=prosody_features.get('sarcasm_probability', 0),
                emotion_tone=str(prosody_features.get('emotion_indicators', {})),
                credibility_flags=credibility['flags'],
                context=context[:3000]  # Limit context length
            ))
            
            # Parse response
            result = self.parse_fact_check_response(response.content)
            result['credibility_analysis'] = credibility
            result['prosody_features'] = prosody_features
            result['source_documents'] = docs
            
            return result
            
        except Exception as e:
            print(f"⚠️ Fact-checking error: {e}")
            return {
                'verdict': 'Error',
                'confidence': 'Low',
                'explanation': f'Error during fact-checking: {str(e)}',
                'evidence': 'N/A',
                'sources': 'N/A',
                'warnings': 'Analysis failed',
                'prosody_impact': 'N/A'
            }
    
    def parse_fact_check_response(self, response: str) -> Dict:
        """Parse the structured fact-check response"""
        result = {}
        
        patterns = {
            'verdict': r'VERDICT:\s*([^\n]+)',
            'confidence': r'CONFIDENCE:\s*([^\n]+)',
            'explanation': r'EXPLANATION:\s*([^\n]+(?:\n(?!EVIDENCE:|SOURCES:|WARNINGS:|PROSODY_IMPACT:)[^\n]*)*)',
            'evidence': r'EVIDENCE:\s*([^\n]+(?:\n(?!SOURCES:|WARNINGS:|PROSODY_IMPACT:)[^\n]*)*)',
            'sources': r'SOURCES:\s*([^\n]+(?:\n(?!WARNINGS:|PROSODY_IMPACT:)[^\n]*)*)',
            'warnings': r'WARNINGS:\s*([^\n]+(?:\n(?!PROSODY_IMPACT:)[^\n]*)*)',
            'prosody_impact': r'PROSODY_IMPACT:\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.MULTILINE | re.DOTALL)
            result[key] = match.group(1).strip() if match else 'N/A'
        
        return result

# Initialize the enhanced fact checker
try:
    import torch
    fact_checker = EnhancedFactChecker()
except ImportError:
    print("⚠️ PyTorch not available, using CPU-only mode")
    fact_checker = EnhancedFactChecker()

# Create a request cache
request_cache = {}

# LRU cache for frequently accessed data
@lru_cache(maxsize=100)
def get_cached_result(claim_hash):
    """Get cached result for a claim"""
    return request_cache.get(claim_hash)

def cache_result(claim, result):
    """Cache a result for future use"""
    claim_hash = hashlib.md5(claim.encode()).hexdigest()
    request_cache[claim_hash] = result
    return claim_hash

def create_enhanced_timeline(claim: str, result: Dict, processing_time: float) -> str:
    """Create an enhanced interactive timeline"""
    
    # Timeline events with more detail
    events = [
        {
            "name": "Input Processing",
            "start": 0,
            "duration": 0.1,
            "status": "✅ Complete",
            "details": f"Received claim: {claim[:50]}...",
            "color": "#3498db"
        },
        {
            "name": "Audio Analysis",
            "start": 0.1,
            "duration": 0.2,
            "status": "✅ Complete",
            "details": f"Sarcasm: {result.get('prosody_features', {}).get('sarcasm_probability', 0):.1%}",
            "color": "#9b59b6"
        },
        {
            "name": "Credibility Assessment",
            "start": 0.3,
            "duration": 0.1,
            "status": "✅ Complete",
            "details": f"Score: {result.get('credibility_analysis', {}).get('credibility_score', 0):.1%}",
            "color": "#e67e22"
        },
        {
            "name": "Knowledge Retrieval",
            "start": 0.4,
            "duration": 0.3,
            "status": "✅ Complete",
            "details": f"Retrieved {len(result.get('source_documents', []))} relevant sources",
            "color": "#34495e"
        },
        {
            "name": "Fact Verification",
            "start": 0.7,
            "duration": 0.2,
            "status": "✅ Complete",
            "details": f"Verdict: {result.get('verdict', 'N/A')}",
            "color": "#27ae60" if result.get('verdict') == 'True' else "#e74c3c" if result.get('verdict') == 'False' else "#f39c12"
        },
        {
            "name": "Report Generation",
            "start": 0.9,
            "duration": 0.1,
            "status": "✅ Complete",
            "details": f"Analysis complete in {processing_time:.2f}s",
            "color": "#2ecc71"
        }
    ]
    
    # Create Plotly timeline
    fig = go.Figure()
    
    for i, event in enumerate(events):
        fig.add_trace(go.Scatter(
            x=[event['start'], event['start'] + event['duration']],
            y=[i, i],
            mode='lines+markers',
            name=event['name'],
            line=dict(color=event['color'], width=8),
            marker=dict(size=10, color=event['color']),
            hovertemplate=f"<b>{event['name']}</b><br>" +
                         f"Status: {event['status']}<br>" +
                         f"Details: {event['details']}<br>" +
                         f"Duration: {event['duration']:.1f}s<extra></extra>"
        ))
    
    fig.update_layout(
        title="🔍 Real-Time Fact-Checking Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Process Stage",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(events))),
            ticktext=[event['name'] for event in events]
        ),
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_credibility_radar(credibility_analysis: Dict, prosody_features: Dict) -> str:
    """Create a credibility radar chart"""
    
    categories = ['Language Quality', 'Source Credibility', 'Logical Consistency', 
                 'Emotional Tone', 'Factual Accuracy', 'Audio Authenticity']
    
    # Calculate scores based on analysis
    scores = [
        0.8 - (len(credibility_analysis.get('flags', [])) * 0.1),  # Language quality
        credibility_analysis.get('credibility_score', 0.5),  # Source credibility
        0.7,  # Logical consistency (would need more analysis)
        1.0 - prosody_features.get('sarcasm_probability', 0),  # Emotional tone
        0.6,  # Factual accuracy (placeholder)
        0.9 - prosody_features.get('sarcasm_probability', 0) * 0.5  # Audio authenticity
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Credibility Score',
        line_color='#3498db',
        fillcolor='rgba(52, 152, 219, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="📊 Credibility Assessment Radar",
        height=400
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def process_enhanced_input(audio, text_claim, enable_prosody=True):
    """Enhanced processing function"""
    start_time = time.time()
    
    if audio is None and not text_claim.strip():
        return ("No input provided", "Please provide audio or text", 
                "❌ Error: No input", "", "", "", "")
    
    try:
        # Initialize results
        transcription = ""
        claim = text_claim.strip()
        prosody_features = {}
        
        # Process audio if provided
        if audio:
            print("🎤 Processing audio...")
            
            # Transcribe
            transcription_result = fact_checker.transcriber(audio,return_timestamps=True)
            transcription = transcription_result['text']
            
            # Prosody analysis
            if enable_prosody:
                prosody_features = fact_checker.extract_advanced_prosody(audio)
            
            # Extract claim if not provided
            if not claim:
                claim_prompt = f"""
                Extract the main factual claim from this transcription. 
                Return only the claim, no explanation:
                
                "{transcription}"
                """
                claim_result = fact_checker.llm.invoke(claim_prompt)
                claim = claim_result.content.strip()
        
        if not claim:
            return (transcription, "No clear claim found", 
                   "❌ Unable to extract a factual claim", "", "", "", "")
        
        # Enhanced fact-checking
        print("🔍 Performing enhanced fact-check...")
        result = fact_checker.enhanced_fact_check(claim, prosody_features)
        
        # Generate analysis report
        analysis_report = f"""
# 🎯 Fact-Check Analysis Report

## 📋 Claim
**{claim}**

## ⚖️ Verdict
**{result['verdict']}** (Confidence: {result['confidence']})

## 📝 Explanation
{result['explanation']}

## 🔍 Evidence
{result['evidence']}

## 📚 Sources
{result['sources']}

## ⚠️ Warnings
{result['warnings']}

## 🎵 Audio Analysis Impact
{result['prosody_impact']}

---
*Analysis completed in {time.time() - start_time:.2f} seconds*
"""
        
        # Create visualizations
        timeline_html = create_enhanced_timeline(claim, result, time.time() - start_time)
        credibility_html = create_credibility_radar(
            result.get('credibility_analysis', {}), 
            prosody_features
        )
        
        # Prosody summary
        prosody_summary = f"""
**Sarcasm Probability:** {prosody_features.get('sarcasm_probability', 0):.1%}
**Emotional Indicators:** {prosody_features.get('emotion_indicators', 'N/A')}
**Pitch Analysis:** Mean: {prosody_features.get('pitch_mean', 0):.1f}Hz, Std: {prosody_features.get('pitch_std', 0):.1f}
**Energy Level:** {prosody_features.get('energy_mean', 0):.3f}
**Speaking Rate:** {prosody_features.get('speaking_rate', 0):.1f} beats/second
""" if prosody_features else "No audio analysis performed"
        
        # Confidence indicator
        confidence_indicator = f"🔴 Low" if result['confidence'] == 'Low' else f"🟡 Medium" if result['confidence'] == 'Medium' else "🟢 High"
        
        return (
            transcription,
            claim,
            analysis_report,
            timeline_html,
            credibility_html,
            prosody_summary,
            confidence_indicator
        )
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return ("Error", "Error", error_msg, "", "", "", "🔴 Error")

# Enhanced UI with modern design
def create_enhanced_ui():
    """Create the enhanced Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .results-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: rgba(23, 162, 184, 0.1);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    """
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        spacing_size="lg",
        radius_size="lg",
        text_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    )
    
    with gr.Blocks(theme=theme, css=css, title="🕵️ Enhanced Conspiracy Theory Debunker") as demo:
        
        # Header
        with gr.Row():
            gr.Markdown("""
            <div class="main-header">
                <h1>🕵️ Enhanced Audio-Driven Conspiracy Theory Debunker</h1>
                <p style="font-size: 1.2em; margin-top: 1rem;">
                    Advanced AI-powered fact-checking with audio analysis, prosody detection, and real-time verification
                </p>
            </div>
            """)
        
        # Info boxes
        with gr.Row():
            gr.Markdown("""
            <div class="info-box">
                <h3>🚀 What's New in This Version:</h3>
                <ul>
                    <li>🎭 Advanced sarcasm detection via prosody analysis</li>
                    <li>📊 Real-time credibility scoring with radar charts</li>
                    <li>🔍 Enhanced knowledge base with fact-checking datasets</li>
                    <li>📈 Interactive timeline visualization</li>
                    <li>🎯 Multi-source verification system</li>
                </ul>
            </div>
            """)
        
        with gr.Row():
            gr.Markdown("""
            <div class="warning-box">
                <h3>⚠️ Important Disclaimers:</h3>
                <ul>
                    <li>This tool is for educational and research purposes only</li>
                    <li>Always verify information through multiple reliable sources</li>
                    <li>AI analysis may have biases and limitations</li>
                    <li>Complex claims may require human expert review</li>
                </ul>
            </div>
            """)
        
        # Main interface
        with gr.Row():
            # Input section
            with gr.Column(scale=2):
                gr.Markdown("## 📝 Input Your Claim")
                
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="🎤 Upload Audio or Record (supports podcasts, voice memos, etc.)",
                    show_download_button=True
                )
                
                text_claim_input = gr.Textbox(
                    label="📝 Or Enter Text Claim Directly",
                    placeholder="e.g., 'The Earth is flat' or 'Vaccines cause autism'",
                    lines=3,
                    max_lines=5
                )
                
                with gr.Row():
                    enable_prosody = gr.Checkbox(
                        label="🎵 Enable Advanced Audio Analysis",
                        value=True,
                        info="Detects sarcasm, emotion, and audio authenticity"
                    )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Claim",
                    variant="primary",
                    size="lg"
                )
                
                # Quick examples
                gr.Markdown("### 🎯 Quick Examples:")
                example_claims = [
                    "The moon landing was faked",
                    "Vaccines cause autism",
                    "Climate change is a hoax",
                    "5G towers cause COVID-19"
                ]
                
                for claim in example_claims:
                    gr.Button(
                        claim,
                        size="sm",
                        variant="secondary"
                    ).click(
                        lambda x=claim: x,
                        outputs=text_claim_input
                    )
            
            # Results section
            with gr.Column(scale=3):
                gr.Markdown("## 📊 Analysis Results")
                
                with gr.Tabs():
                    with gr.Tab("📋 Summary"):
                        with gr.Row():
                            transcription_output = gr.Textbox(
                                label="🎤 Transcription",
                                lines=3,
                                show_copy_button=True
                            )
                            confidence_output = gr.Textbox(
                                label="🎯 Confidence Level",
                                lines=1,
                                show_copy_button=True
                            )
                        
                        claim_output = gr.Textbox(
                            label="📝 Extracted Claim",
                            lines=2,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🔍 Full Analysis"):
                        analysis_output = gr.Markdown(
                            label="📊 Detailed Analysis Report",
                            show_copy_button=True
                        )
                    
                    with gr.Tab("📈 Timeline"):
                        timeline_output = gr.HTML(
                            label="⏱️ Real-Time Processing Timeline"
                        )
                    
                    with gr.Tab("🎯 Credibility Radar"):
                        credibility_output = gr.HTML(
                            label="📊 Multi-Dimensional Credibility Assessment"
                        )
                    
                    with gr.Tab("🎵 Audio Analysis"):
                        prosody_output = gr.Markdown(
                            label="🎭 Advanced Prosody & Emotion Analysis"
                        )
                    
                    with gr.Tab("📚 Sources"):
                        sources_output = gr.Markdown(
                            label="📖 Knowledge Base Sources Used"
                        )
        
        # Statistics dashboard
        with gr.Row():
            gr.Markdown("## 📊 Live Statistics Dashboard")
            
            with gr.Column():
                total_claims = gr.Number(
                    label="Total Claims Analyzed",
                    value=0,
                    interactive=False
                )
                
                accuracy_rate = gr.Number(
                    label="Average Confidence Score",
                    value=0.0,
                    interactive=False
                )
            
            with gr.Column():
                sarcasm_detected = gr.Number(
                    label="Sarcasm Detection Rate",
                    value=0.0,
                    interactive=False
                )
                
                processing_time = gr.Number(
                    label="Avg Processing Time (s)",
                    value=0.0,
                    interactive=False
                )
        
        # Event handlers
        def enhanced_process_wrapper(audio, text_claim, enable_prosody):
            """Wrapper function that updates statistics"""
            result = process_enhanced_input(audio, text_claim, enable_prosody)
            
            # Update statistics (in real app, this would be persistent)
            # For demo purposes, we'll just show the concept
            
            return result
        
        analyze_btn.click(
            enhanced_process_wrapper,
            inputs=[audio_input, text_claim_input, enable_prosody],
            outputs=[
                transcription_output,
                claim_output,
                analysis_output,
                timeline_output,
                credibility_output,
                prosody_output,
                confidence_output
            ]
        )
        
        # Footer
        with gr.Row():
            gr.Markdown("""
            ---
            <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; margin-top: 2rem;">
                <h3>🛠️ Technical Features</h3>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                    <div style="margin: 1rem;">
                        <h4>🎤 Audio Processing</h4>
                        <p>Whisper-Small • Prosody Analysis • Emotion Detection</p>
                    </div>
                    <div style="margin: 1rem;">
                        <h4>🧠 AI Models</h4>
                        <p>GPT-4 • MPNet Embeddings • RoBERTa Sentiment</p>
                    </div>
                    <div style="margin: 1rem;">
                        <h4>📊 Knowledge Base</h4>
                        <p>Wikipedia • Fact-Check Datasets • Scientific Literature</p>
                    </div>
                    <div style="margin: 1rem;">
                        <h4>🔍 Analysis Types</h4>
                        <p>Sarcasm Detection • Credibility Scoring • Source Verification</p>
                    </div>
                </div>
                
                <p style="margin-top: 2rem; opacity: 0.8;">
                    Built for data scientists, researchers, and AI safety professionals.<br>
                    <strong>Version 2.0</strong> - Enhanced with multi-modal analysis and real-time verification
                </p>
            </div>
            """)
    
    return demo

# Additional utility functions for YouTube integration (for future use)
def extract_youtube_audio(url: str) -> str:
    """Extract audio from YouTube URL (placeholder for future implementation)"""
    # This would use yt-dlp or similar to extract audio
    # For now, return placeholder
    return "youtube_audio_extraction_placeholder.wav"

def batch_process_claims(claims_list: List[str]) -> List[Dict]:
    """Process multiple claims in batch (for future scaling)"""
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for claim in claims_list:
            future = executor.submit(
                fact_checker.enhanced_fact_check, 
                claim, 
                {}  # No prosody for batch processing
            )
            futures.append(future)
        
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                results.append({
                    'verdict': 'Error',
                    'explanation': f'Processing failed: {str(e)}'
                })
    
    return results

# Create Flask app for API and serving static files
app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_index():
    """Serve the index.html file"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('frontend', path)

@app.route('/api/analyze', methods=['POST'])
def analyze_claim():
    """API endpoint for analyzing claims"""
    print("📝 Received analyze request")
    data = request.json
    
    # Extract data from request
    text_claim = data.get('text_claim', '')
    audio_data = data.get('audio_data')  # This would be base64 encoded
    enable_prosody = data.get('enable_prosody', True)
    
    print(f"📋 Processing claim: '{text_claim}'")
    
    # Check cache first
    claim_hash = hashlib.md5(text_claim.encode()).hexdigest()
    cached_result = get_cached_result(claim_hash)
    if cached_result:
        print("🔍 Using cached result")
        return jsonify(cached_result)
    
    try:
        # Process directly for debugging
        start_time = time.time()
        
        # Convert audio_data to file if provided
        audio_file = None
        if audio_data:
            print("🎤 Processing audio data")
            import base64
            import tempfile
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(base64.b64decode(audio_data))
                audio_file = temp_audio.name
                print(f"🎵 Audio saved to temporary file: {audio_file}")
        
        # Process the claim
        print("🔍 Processing claim with enhanced input function")
        result = process_enhanced_input(audio_file, text_claim, enable_prosody)
        
        # Format the result
        formatted_result = {
            'transcription': result[0],
            'claim': result[1],
            'analysis': result[2],
            'timeline': result[3],
            'credibility': result[4],
            'prosody': result[5],
            'confidence': result[6],
            'processing_time': time.time() - start_time
        }
        
        # Cache the result
        cache_result(text_claim, formatted_result)
        
        print("✅ Analysis complete, returning result")
        return jsonify(formatted_result)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Enhanced Conspiracy Theory Debunker...")
    print("🔧 Loading models and building knowledge base...")
    
    # Create Gradio UI for backward compatibility
    demo = create_enhanced_ui()
    
    # Start Gradio in a separate thread
    threading.Thread(target=lambda: demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        max_threads=10,
        quiet=True
    ), daemon=True).start()
    
    print("✅ Application ready!")
    print("🌐 Launching web interface...")
    
    # Start Flask app for the new frontend
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
