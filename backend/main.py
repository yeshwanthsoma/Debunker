"""
Enhanced Conspiracy Theory Debunker - Fixed Backend API
Working around LangChain dependency issues
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import asyncio
import logging
import traceback
import time
import hashlib
import tempfile
import os
import json
import numpy as np
import librosa
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Conspiracy Theory Debunker API",
    description="Advanced AI-powered fact-checking with audio analysis",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AnalysisRequest(BaseModel):
    text_claim: str = Field(..., description="The claim to analyze")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    enable_prosody: bool = Field(True, description="Enable prosody analysis for audio")

class AnalysisResponse(BaseModel):
    transcription: str = Field("", description="Audio transcription")
    claim: str = Field("", description="Extracted or provided claim")
    analysis: str = Field("", description="Detailed analysis report")
    timeline: str = Field("", description="HTML timeline visualization")
    credibility: str = Field("", description="HTML credibility radar chart")
    prosody: str = Field("", description="Prosody analysis summary")
    confidence: str = Field("", description="Confidence level")
    verdict: str = Field("", description="Fact-check verdict")
    processing_time: float = Field(0.0, description="Processing time in seconds")

# Global variables for models
fact_checker = None
request_cache = {}

class AlternativeFactChecker:
    """Alternative fact checker without LangChain dependencies"""
    
    def __init__(self):
        self.initialize_models()
        self.knowledge_base = self.build_simple_knowledge_base()
        
    def initialize_models(self):
        """Initialize AI models"""
        logger.info("🔧 Initializing models...")
        
        try:
            # Speech recognition
            self.transcriber = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                device=-1  # CPU only for now
            )
            
            # Sentence embeddings
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            
            logger.info("✅ Models initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            raise
    
    def build_simple_knowledge_base(self):
        """Build a simple knowledge base without FAISS"""
        knowledge_base = {
            "flat_earth": {
                "verdict": "False",
                "explanation": "The Earth is an oblate spheroid, as proven by numerous scientific observations including satellite imagery, physics experiments, and astronomical observations.",
                "confidence": "High"
            },
            "moon_landing_fake": {
                "verdict": "False", 
                "explanation": "The Apollo moon landings (1969-1972) are well-documented historical events supported by extensive evidence including rock samples, retroreflectors, and independent verification.",
                "confidence": "High"
            },
            "moon_landing": {
                "verdict": "True", 
                "explanation": "The Apollo moon landings (1969-1972) are well-documented historical events supported by extensive evidence including rock samples, retroreflectors, and independent verification.",
                "confidence": "High"
            },
            "vaccines_autism": {
                "verdict": "False",
                "explanation": "Multiple large-scale scientific studies have found no link between vaccines and autism. The original study claiming this link was fraudulent and has been retracted.",
                "confidence": "High"
            },
            "climate_change": {
                "verdict": "True",
                "explanation": "Climate change due to human activities is supported by overwhelming scientific consensus and evidence from multiple independent sources.",
                "confidence": "High"
            },
            "5g_covid": {
                "verdict": "False", 
                "explanation": "There is no scientific evidence linking 5G technology to COVID-19. The virus existed before 5G deployment and spreads in areas without 5G coverage.",
                "confidence": "High"
            }
        }
        return knowledge_base
    
    def extract_advanced_prosody(self, audio_path: str) -> Dict:
        """Extract prosody features from audio"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            
            features = {}
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = 0.0
            
            # Energy analysis
            rms = librosa.feature.rms(y=y)
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            
            # Tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['speaking_rate'] = float(len(beats) / (len(y) / sr))
            
            # Sarcasm probability (simple heuristic)
            sarcasm_score = 0.0
            if features['pitch_std'] > 80:
                sarcasm_score += 0.3
            if features['energy_std'] > 0.05:
                sarcasm_score += 0.2
            if features['speaking_rate'] < 2.0:
                sarcasm_score += 0.3
            
            features['sarcasm_probability'] = min(sarcasm_score, 1.0)
            
            return features
            
        except Exception as e:
            logger.error(f"⚠️ Prosody analysis error: {e}")
            return {
                'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
                'energy_mean': 0.0, 'energy_std': 0.0, 'tempo': 0.0,
                'speaking_rate': 0.0, 'sarcasm_probability': 0.0
            }
    
    def analyze_claim_credibility(self, claim: str) -> Dict:
        """Analyze claim credibility markers"""
        credibility_score = 0.5
        flags = []
        
        # Red flags
        red_flags = [
            r'\b(they don\'t want you to know|hidden truth|mainstream media lies)\b',
            r'\b(big pharma|global elite|deep state)\b',
            r'\b(wake up|sheeple|do your research)\b',
        ]
        
        for pattern in red_flags:
            if re.search(pattern, claim.lower()):
                credibility_score -= 0.15
                flags.append(f"Red flag detected")
        
        # Green flags
        green_flags = [
            r'\b(according to|study shows|research indicates)\b',
            r'\b(peer reviewed|scientific consensus)\b',
        ]
        
        for pattern in green_flags:
            if re.search(pattern, claim.lower()):
                credibility_score += 0.1
                flags.append(f"Credible language detected")
        
        return {
            'credibility_score': max(0, min(1, credibility_score)),
            'flags': flags
        }
    
    def enhanced_fact_check(self, claim: str, prosody_features: Dict) -> Dict:
        """Enhanced fact-checking"""
        claim_lower = claim.lower()
        
        # Check against knowledge base
        verdict = "Unverifiable"
        explanation = "This claim could not be verified against our knowledge base."
        confidence = "Low"
        
        # Special handling for moon landing conspiracy claims
        if "moon landing" in claim_lower and any(word in claim_lower for word in ["fake", "faked", "hoax", "staged", "conspiracy"]):
            key = "moon_landing_fake"
            if key in self.knowledge_base:
                verdict = self.knowledge_base[key]["verdict"]
                explanation = self.knowledge_base[key]["explanation"]
                confidence = self.knowledge_base[key]["confidence"]
        else:
            # Regular knowledge base lookup
            for key, info in self.knowledge_base.items():
                if key.replace("_", " ") in claim_lower or any(word in claim_lower for word in key.split("_")):
                    verdict = info["verdict"]
                    explanation = info["explanation"]
                    confidence = info["confidence"]
                    break
        
        # Analyze credibility
        credibility = self.analyze_claim_credibility(claim)
        
        # Sentiment analysis
        try:
            sentiment = self.sentiment_analyzer(claim)
            sentiment_label = sentiment[0]['label'] if sentiment else 'NEUTRAL'
        except:
            sentiment_label = 'NEUTRAL'
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'explanation': explanation,
            'evidence': f"Analysis based on pattern matching and knowledge base lookup. Sentiment: {sentiment_label}",
            'sources': "Built-in knowledge base with scientific consensus data",
            'warnings': "This analysis uses a simplified fact-checking approach",
            'prosody_impact': f"Sarcasm probability: {prosody_features.get('sarcasm_probability', 0):.2%}",
            'credibility_analysis': credibility,
            'prosody_features': prosody_features
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the fact checker on startup"""
    global fact_checker
    logger.info("🚀 Starting Enhanced Fact Checker...")
    try:
        fact_checker = AlternativeFactChecker()
        logger.info("✅ Fact checker initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize fact checker: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Conspiracy Theory Debunker API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "fact_checker_ready": fact_checker is not None,
        "timestamp": time.time()
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_claim(request: AnalysisRequest):
    """Analyze a claim for fact-checking"""
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info(f"📝 Received analysis request for claim: '{request.text_claim[:50]}...'")
    
    start_time = time.time()
    
    try:
        # Process audio if provided
        audio_file = None
        transcription = ""
        prosody_features = {}
        
        if request.audio_data:
            logger.info("🎤 Processing audio data")
            audio_file = await process_audio_data(request.audio_data)
            
            # Transcribe
            transcription_result = fact_checker.transcriber(audio_file, return_timestamps=True)
            transcription = transcription_result['text']
            
            # Prosody analysis
            if request.enable_prosody:
                prosody_features = fact_checker.extract_advanced_prosody(audio_file)
        
        # Use provided claim or extract from transcription
        claim = request.text_claim.strip()
        if not claim and transcription:
            claim = transcription.strip()
        
        if not claim:
            raise HTTPException(status_code=400, detail="No clear claim found")
        
        # Perform fact-checking
        logger.info("🔍 Performing fact-check...")
        fact_check_result = fact_checker.enhanced_fact_check(claim, prosody_features)
        
        # Generate visualizations
        timeline_html = create_enhanced_timeline(claim, fact_check_result, time.time() - start_time)
        credibility_html = create_credibility_radar(
            fact_check_result.get('credibility_analysis', {}),
            prosody_features
        )
        
        # Format responses
        analysis_report = format_analysis_report(fact_check_result, claim)
        prosody_summary = format_prosody_summary(prosody_features) if prosody_features else "No audio analysis performed"
        
        processing_time = time.time() - start_time
        
        result = AnalysisResponse(
            transcription=transcription,
            claim=claim,
            analysis=analysis_report,
            timeline=timeline_html,
            credibility=credibility_html,
            prosody=prosody_summary,
            confidence=fact_check_result.get('confidence', 'Unknown'),
            verdict=fact_check_result.get('verdict', 'Unknown'),
            processing_time=processing_time
        )
        
        logger.info(f"✅ Analysis completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-file")
async def analyze_claim_with_file(
    audio_file: Optional[UploadFile] = File(None),
    text_claim: str = Form(""),
    enable_prosody: bool = Form(True)
):
    """Analyze a claim with file upload support"""
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info(f"📝 Received file analysis request")
    
    try:
        # Save uploaded file temporarily if provided
        audio_file_path = None
        if audio_file:
            logger.info(f"🎤 Processing uploaded audio file: {audio_file.filename}")
            audio_file_path = await save_uploaded_file(audio_file)
        
        # Convert to AnalysisRequest format
        audio_data = None
        if audio_file_path:
            # Convert file to base64 for processing
            import base64
            with open(audio_file_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        request = AnalysisRequest(
            text_claim=text_claim,
            audio_data=audio_data,
            enable_prosody=enable_prosody
        )
        
        result = await analyze_claim(request)
        
        # Clean up temporary file
        if audio_file_path and os.path.exists(audio_file_path):
            os.unlink(audio_file_path)
        
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Error during file analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def process_audio_data(audio_data: str) -> str:
    """Process base64 audio data and return file path"""
    import base64
    
    try:
        audio_bytes = base64.b64decode(audio_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio data")

async def save_uploaded_file(audio_file: UploadFile) -> str:
    """Save uploaded file and return path"""
    try:
        # Get file extension from uploaded filename
        import os
        file_extension = os.path.splitext(audio_file.filename)[1] if audio_file.filename else '.wav'
        
        # Default to .wav if no extension found
        if not file_extension:
            file_extension = '.wav'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to save uploaded file")

def format_analysis_report(result: Dict, claim: str) -> str:
    """Format the analysis result into a readable report"""
    return f"""
# 🎯 Fact-Check Analysis Report

## 📋 Claim
**{claim}**

## ⚖️ Verdict
**{result.get('verdict', 'Unknown')}** (Confidence: {result.get('confidence', 'Unknown')})

## 📝 Explanation
{result.get('explanation', 'No explanation available')}

## 🔍 Evidence
{result.get('evidence', 'No evidence available')}

## 📚 Sources
{result.get('sources', 'No sources available')}

## ⚠️ Warnings
{result.get('warnings', 'No warnings')}

## 🎵 Audio Analysis Impact
{result.get('prosody_impact', 'No audio analysis')}
"""

def format_prosody_summary(prosody_features: Dict) -> str:
    """Format prosody features into a readable summary"""
    return f"""
**Sarcasm Probability:** {prosody_features.get('sarcasm_probability', 0):.1%}
**Pitch Analysis:** Mean: {prosody_features.get('pitch_mean', 0):.1f}Hz, Std: {prosody_features.get('pitch_std', 0):.1f}
**Energy Level:** {prosody_features.get('energy_mean', 0):.3f}
**Speaking Rate:** {prosody_features.get('speaking_rate', 0):.1f} beats/second
**Tempo:** {prosody_features.get('tempo', 0):.1f} BPM
"""

def create_enhanced_timeline(claim: str, result: Dict, processing_time: float) -> str:
    """Create an enhanced interactive timeline"""
    return f"""
<div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;">
    <h3>📈 Processing Timeline</h3>
    <div style="margin: 10px 0;">
        <div style="background: #e3f2fd; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #2196f3;">
            ✅ <strong>Input Processing</strong> - Claim received and validated
        </div>
        <div style="background: #f3e5f5; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #9c27b0;">
            ✅ <strong>Audio Analysis</strong> - Transcription and prosody analysis
        </div>
        <div style="background: #fff3e0; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #ff9800;">
            ✅ <strong>Credibility Assessment</strong> - Language and pattern analysis
        </div>
        <div style="background: #e8f5e8; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #4caf50;">
            ✅ <strong>Fact Verification</strong> - Knowledge base lookup: {result.get('verdict', 'Unknown')}
        </div>
        <div style="background: #fce4ec; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #e91e63;">
            ✅ <strong>Report Generation</strong> - Analysis complete in {processing_time:.2f}s
        </div>
    </div>
</div>
"""

def create_credibility_radar(credibility_analysis: Dict, prosody_features: Dict) -> str:
    """Create a credibility radar chart"""
    credibility_score = credibility_analysis.get('credibility_score', 0.5)
    sarcasm_prob = prosody_features.get('sarcasm_probability', 0)
    
    return f"""
<div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;">
    <h3>🎯 Credibility Assessment</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
        <div>
            <strong>Language Quality:</strong>
            <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div style="background: #4caf50; height: 20px; width: {credibility_score*100}%; display: flex; align-items: center; padding-left: 5px; color: white; font-size: 12px;">
                    {credibility_score:.1%}
                </div>
            </div>
        </div>
        <div>
            <strong>Audio Authenticity:</strong>
            <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div style="background: #2196f3; height: 20px; width: {(1-sarcasm_prob)*100}%; display: flex; align-items: center; padding-left: 5px; color: white; font-size: 12px;">
                    {(1-sarcasm_prob):.1%}
                </div>
            </div>
        </div>
    </div>
    <div style="margin-top: 10px; font-size: 14px; color: #666;">
        <p><strong>Flags:</strong> {', '.join(credibility_analysis.get('flags', ['None detected']))}</p>
    </div>
</div>
"""

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "total_requests": len(request_cache),
        "cache_size": len(request_cache),
        "uptime": time.time(),
        "version": "2.0.0-fixed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
