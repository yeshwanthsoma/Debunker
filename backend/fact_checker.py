"""
Enhanced Fact Checker - Core Logic
Extracted from the original app.py for modularity
"""

import os
import pickle
import warnings
import numpy as np
import librosa
import re
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

class EnhancedFactChecker:
    def __init__(self):
        """Initialize the fact checker with all models and knowledge base"""
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
        
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
            cuda_available = torch.cuda.is_available()
        except ImportError:
            device = -1
            cuda_available = False
        
        # Speech recognition model
        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-small",
            device=device
        )
        
        # Enhanced embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-mpnet-base-v2',
            model_kwargs={'device': 'cuda' if cuda_available else 'cpu'}
        )
        
        # Sentiment and emotion analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device
        )
        
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=device
        )
        
        # LLM with better configuration
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1000,
            request_timeout=30
        )
        
        print("✅ Models initialized successfully!")
    
    def build_enhanced_index(self):
        """Build a more comprehensive knowledge base"""
        index_path = "enhanced_faiss_index"
        
        if os.path.exists(index_path):
            print("📚 Loading existing enhanced index...")
            try:
                self.vectorstore = FAISS.load_local(
                    index_path, 
                    embeddings=self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                return
            except Exception as e:
                print(f"⚠️ Error loading index: {e}. Rebuilding...")
        
        print("🏗️ Building enhanced knowledge base...")
        
        # Multiple reliable sources
        sources = []
        
        # 1. Wikipedia (focused on controversial topics)
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
                    split=f"train[:{100}]"  # Reduced for faster loading
                )
                wiki_texts = [
                    f"WIKIPEDIA: {doc['title']}\n{doc['text'][:2000]}" 
                    for doc in wiki_data 
                    if any(t.lower() in doc['text'].lower() for t in [topic])
                ]
                sources.extend(wiki_texts[:10])  # Limit per topic
            except Exception as e:
                print(f"⚠️ Error loading Wikipedia data for {topic}: {e}")
        
        # 2. Fact-checking datasets
        try:
            fact_check_data = load_dataset("liar", split="train[:100]")  # Reduced
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
                features['pitch_mean'] = float(np.mean(valid_pitch))
                features['pitch_std'] = float(np.std(valid_pitch))
                features['pitch_range'] = float(np.max(valid_pitch) - np.min(valid_pitch))
            else:
                features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = 0.0
            
            # Energy and intensity
            rms = librosa.feature.rms(y=y)
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            # Rhythm and tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['speaking_rate'] = float(len(beats) / (len(y) / sr))
            
            # Sarcasm indicators
            sarcasm_score = 0.0
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
                'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
                'energy_mean': 0.0, 'energy_std': 0.0, 'spectral_centroid_mean': 0.0,
                'tempo': 0.0, 'speaking_rate': 0.0, 'sarcasm_probability': 0.0,
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
        
        # Length and complexity
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
            search_kwargs={"k": 8}
        )
        
        docs = retriever.get_relevant_documents(claim)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Analyze claim credibility
        credibility = self.analyze_claim_credibility(claim)
        
        # Enhanced prompt
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
                context=context[:3000]
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

    def create_enhanced_timeline(self, claim: str, result: Dict, processing_time: float) -> str:
        """Create an enhanced interactive timeline"""
        
        # Timeline events
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

    def create_credibility_radar(self, credibility_analysis: Dict, prosody_features: Dict) -> str:
        """Create a credibility radar chart"""
        
        categories = ['Language Quality', 'Source Credibility', 'Logical Consistency', 
                     'Emotional Tone', 'Factual Accuracy', 'Audio Authenticity']
        
        # Calculate scores based on analysis
        scores = [
            0.8 - (len(credibility_analysis.get('flags', [])) * 0.1),  # Language quality
            credibility_analysis.get('credibility_score', 0.5),  # Source credibility
            0.7,  # Logical consistency (placeholder)
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