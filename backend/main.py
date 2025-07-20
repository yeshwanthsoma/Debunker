"""
TruthLens - Professional Fact-Checking Backend API
Enhanced with Google Fact Check Tools API and OpenAI integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
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
from config import get_settings, is_api_available
from fact_check_apis import MultiSourceFactChecker
from models import (
    AnalysisRequest, AnalysisResponse, ProsodyAnalysis, SourceInfo,
    EvidenceAssessment, DebateContent, CredibilityMetrics, ExpertOpinion,
    ErrorResponse, HealthResponse, StatsResponse
)

# Configure logging
import uuid
from datetime import datetime

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Request tracking
request_counter = 0

# Initialize FastAPI app
app = FastAPI(
    title="TruthLens - Professional Fact-Checking API",
    description="Advanced AI-powered fact-checking with Google Fact Check Tools API, OpenAI integration, and audio analysis",
    version="3.0.0"
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

class ProsodyAnalysis(BaseModel):
    sarcasm_probability: float = Field(0.0, description="Probability of sarcasm (0-1)")
    pitch_mean: float = Field(0.0, description="Average pitch in Hz")
    pitch_std: float = Field(0.0, description="Pitch standard deviation")
    energy_mean: float = Field(0.0, description="Average energy level")
    speaking_rate: float = Field(0.0, description="Speaking rate in beats/second")
    tempo: float = Field(0.0, description="Tempo in BPM")

class SourceInfo(BaseModel):
    name: str = Field("", description="Source name")
    type: str = Field("", description="Source type (Web Source, AI Analysis, etc.)")
    url: Optional[str] = Field(None, description="Source URL if available")
    rating: Optional[str] = Field(None, description="Source rating or credibility")

class EvidenceAssessment(BaseModel):
    primary_claims: List[str] = Field([], description="Primary factual claims extracted")
    claim_evaluations: Dict[str, str] = Field({}, description="Individual claim evaluations")
    web_sources_consulted: List[str] = Field([], description="Web sources used in research")
    reasoning: str = Field("", description="Logical reasoning for verdict")
    
class DebateContent(BaseModel):
    supporting_arguments: List[str] = Field([], description="Arguments supporting the claim")
    opposing_arguments: List[str] = Field([], description="Arguments opposing the claim") 
    expert_opinions: List[Dict[str, str]] = Field([], description="Expert opinions and quotes")
    historical_context: str = Field("", description="Historical context of the claim")
    scientific_consensus: str = Field("", description="Current scientific consensus")

class CredibilityMetrics(BaseModel):
    language_quality: float = Field(0.5, description="Quality of language used (0-1)")
    audio_authenticity: float = Field(0.5, description="Audio authenticity score (0-1)")
    source_reliability: float = Field(0.5, description="Reliability of sources (0-1)")
    factual_accuracy: float = Field(0.5, description="Factual accuracy score (0-1)")
    flags: List[str] = Field([], description="Credibility warning flags")

class AnalysisResponse(BaseModel):
    transcription: str = Field("", description="Audio transcription")
    claim: str = Field("", description="Extracted or provided claim")
    verdict: str = Field("", description="Fact-check verdict (True/False/Misleading/Unverifiable)")
    confidence: float = Field(0.0, description="Confidence level (0-1)")
    explanation: str = Field("", description="Detailed explanation of the verdict")
    evidence: EvidenceAssessment = Field(default_factory=EvidenceAssessment, description="Evidence assessment details")
    sources: List[SourceInfo] = Field([], description="Sources used in fact-checking")
    prosody: Optional[ProsodyAnalysis] = Field(None, description="Audio prosody analysis")
    credibility_metrics: CredibilityMetrics = Field(default_factory=CredibilityMetrics, description="Credibility assessment")
    debate_content: Optional[DebateContent] = Field(None, description="Debate and discussion content")
    provider: str = Field("", description="Fact-checking provider used")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    timestamp: str = Field("", description="Analysis timestamp")

# Global variables for models
fact_checker = None
multi_source_checker = None
request_cache = {}
settings = get_settings()

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
        """Enhanced fact-checking with fallback to simple knowledge base"""
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
            'warnings': "This analysis uses a simplified fact-checking approach - upgrade to professional APIs for enhanced verification",
            'prosody_impact': f"Sarcasm probability: {prosody_features.get('sarcasm_probability', 0):.2%}",
            'credibility_analysis': credibility,
            'prosody_features': prosody_features
        }
    
    async def professional_fact_check(self, claim: str, prosody_features: Dict, context: str = None) -> Dict:
        """Professional fact-checking using multiple external APIs"""
        global multi_source_checker
        
        try:
            if multi_source_checker:
                logger.info("🔍 Using professional fact-checking APIs...")
                
                # Get comprehensive fact-check results
                result = await multi_source_checker.comprehensive_fact_check(claim, context)
                
                # Enhance with prosody analysis
                result['prosody_impact'] = f"Sarcasm probability: {prosody_features.get('sarcasm_probability', 0):.2%}"
                result['prosody_features'] = prosody_features
                
                # Add credibility analysis
                credibility = self.analyze_claim_credibility(claim)
                result['credibility_analysis'] = credibility
                
                # Add professional warning
                result['warnings'] = "Analysis performed using professional fact-checking APIs and AI verification"
                
                return result
            else:
                logger.warning("Professional fact-checker not available, falling back to simple method")
                return self.enhanced_fact_check(claim, prosody_features)
                
        except Exception as e:
            logger.error(f"Professional fact-check failed: {e}")
            logger.info("Falling back to simple fact-checking method")
            return self.enhanced_fact_check(claim, prosody_features)

@app.on_event("startup")
async def startup_event():
    """Initialize the fact checker on startup"""
    global fact_checker, multi_source_checker
    logger.info("🚀 Starting TruthLens Professional Fact-Checking System...")
    logger.info("=" * 60)
    
    # Check API availability first
    from config import is_api_available, get_settings
    settings = get_settings()
    
    logger.info("📋 API Configuration Status:")
    logger.info(f"   Environment: {settings.environment}")
    logger.info(f"   Debug Mode: {settings.debug}")
    logger.info(f"   Log Level: {settings.log_level}")
    
    api_status = {
        "Google Fact Check": is_api_available("google_fact_check"),
        "OpenAI": is_api_available("openai"), 
        "Anthropic": is_api_available("anthropic"),
        "News API": is_api_available("news_api")
    }
    
    for api_name, available in api_status.items():
        status_icon = "✅" if available else "❌"
        logger.info(f"   {status_icon} {api_name}: {'Available' if available else 'Not configured'}")
    
    available_apis = [name for name, available in api_status.items() if available]
    logger.info(f"📊 Professional APIs Available: {len(available_apis)}/{len(api_status)}")
    
    try:
        logger.info("🔧 Initializing AI models...")
        fact_checker = AlternativeFactChecker()
        logger.info("✅ Basic fact checker initialized successfully!")
        
        # Initialize professional fact-checking if APIs are available
        if any(api_status.values()):
            try:
                logger.info("🔌 Initializing professional fact-checking APIs...")
                multi_source_checker = MultiSourceFactChecker()
                logger.info("✅ Professional fact-checking APIs initialized!")
                logger.info(f"🎯 Mode: PROFESSIONAL ({', '.join(available_apis)})")
            except Exception as e:
                logger.error(f"❌ Professional API initialization failed: {e}")
                logger.info("📝 Falling back to basic fact-checking mode.")
                multi_source_checker = None
        else:
            logger.warning("⚠️ No professional APIs configured")
            logger.info("📝 Running in BASIC mode. Add API keys for professional verification.")
            multi_source_checker = None
            
        logger.info("=" * 60)
        logger.info("🎉 TruthLens startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize fact checker: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global multi_source_checker
    if multi_source_checker:
        await multi_source_checker.close()
        logger.info("🔄 API sessions closed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TruthLens - Professional Fact-Checking API",
        "version": "3.0.0",
        "status": "running",
        "features": {
            "basic_fact_checking": True,
            "google_fact_check_api": multi_source_checker is not None,
            "openai_integration": multi_source_checker is not None,
            "audio_analysis": True,
            "prosody_detection": True
        }
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
    global request_counter
    request_counter += 1
    request_id = f"REQ-{request_counter:04d}"
    
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info("=" * 80)
    logger.info(f"🎯 NEW FACT-CHECK REQUEST [{request_id}]")
    logger.info(f"📝 Claim: '{request.text_claim[:100]}{'...' if len(request.text_claim) > 100 else ''}'")
    logger.info(f"🎤 Audio Data: {'Yes' if request.audio_data else 'No'}")
    logger.info(f"🎵 Prosody Analysis: {'Enabled' if request.enable_prosody else 'Disabled'}")
    logger.info(f"🔌 Professional APIs: {'Available' if multi_source_checker else 'Not Available'}")
    
    start_time = time.time()
    stage_times = {"start": start_time}
    
    try:
        # Stage 1: Audio Processing
        audio_file = None
        transcription = ""
        prosody_features = {}
        
        if request.audio_data:
            logger.info(f"🎤 [{request_id}] STAGE 1: Processing audio data...")
            audio_file = await process_audio_data(request.audio_data)
            stage_times["audio_processed"] = time.time()
            logger.debug(f"   Audio file saved: {audio_file}")
            
            # Transcribe
            logger.info(f"🗣️ [{request_id}] STAGE 2: Transcribing audio with Whisper...")
            transcription_result = fact_checker.transcriber(audio_file, return_timestamps=True)
            transcription = transcription_result['text']
            stage_times["transcription_done"] = time.time()
            logger.info(f"   Transcription: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            
            # Prosody analysis
            if request.enable_prosody:
                logger.info(f"🎵 [{request_id}] STAGE 3: Analyzing prosody features...")
                prosody_features = fact_checker.extract_advanced_prosody(audio_file)
                stage_times["prosody_done"] = time.time()
                logger.debug(f"   Prosody features: {prosody_features}")
                logger.info(f"   Sarcasm probability: {prosody_features.get('sarcasm_probability', 0):.2%}")
        else:
            logger.info(f"📝 [{request_id}] No audio data provided, using text claim only")
        
        # Stage 4: Claim Extraction
        claim = request.text_claim.strip()
        if not claim and transcription:
            claim = transcription.strip()
            logger.info(f"📋 [{request_id}] Using transcribed claim")
        
        if not claim:
            logger.error(f"❌ [{request_id}] No clear claim found")
            raise HTTPException(status_code=400, detail="No clear claim found")
        
        logger.info(f"📋 [{request_id}] Final claim: '{claim[:100]}{'...' if len(claim) > 100 else ''}'")
        
        # Stage 5: Fact-Checking
        logger.info(f"🔍 [{request_id}] STAGE 4: Starting fact-checking process...")
        stage_times["fact_check_start"] = time.time()
        
        if multi_source_checker:
            logger.info(f"🎯 [{request_id}] Using PROFESSIONAL fact-checking (Multi-API)")
            try:
                fact_check_result = await fact_checker.professional_fact_check(claim, prosody_features, transcription)
                stage_times["fact_check_done"] = time.time()
                logger.info(f"✅ [{request_id}] Professional fact-check completed")
                logger.info(f"   Verdict: {fact_check_result.get('verdict', 'Unknown')}")
                logger.info(f"   Confidence: {fact_check_result.get('confidence', 'Unknown')}")
                logger.info(f"   Provider: {fact_check_result.get('provider', 'Unknown')}")
            except Exception as e:
                logger.error(f"❌ [{request_id}] Professional fact-check failed: {e}")
                logger.info(f"🔄 [{request_id}] Falling back to basic fact-checking...")
                fact_check_result = fact_checker.enhanced_fact_check(claim, prosody_features)
                stage_times["fact_check_done"] = time.time()
                logger.info(f"✅ [{request_id}] Basic fact-check completed as fallback")
        else:
            logger.info(f"📝 [{request_id}] Using BASIC fact-checking (Built-in knowledge)")
            fact_check_result = fact_checker.enhanced_fact_check(claim, prosody_features)
            stage_times["fact_check_done"] = time.time()
            logger.info(f"✅ [{request_id}] Basic fact-check completed")
            logger.info(f"   Verdict: {fact_check_result.get('verdict', 'Unknown')}")
            logger.info(f"   Confidence: {fact_check_result.get('confidence', 'Unknown')}")
        
        # Stage 6: Structure Response Data
        logger.info(f"📊 [{request_id}] STAGE 5: Structuring response data...")
        stage_times["structuring_start"] = time.time()
        
        # Prepare clean structured data
        prosody_data = None
        if prosody_features:
            prosody_data = ProsodyAnalysis(
                sarcasm_probability=prosody_features.get('sarcasm_probability', 0.0),
                pitch_mean=prosody_features.get('pitch_mean', 0.0),
                pitch_std=prosody_features.get('pitch_std', 0.0),
                energy_mean=prosody_features.get('energy_mean', 0.0),
                speaking_rate=prosody_features.get('speaking_rate', 0.0),
                tempo=prosody_features.get('tempo', 0.0)
            )
        
        # Extract key values from fact_check_result for fallbacks
        if isinstance(fact_check_result, dict):
            explanation_val = fact_check_result.get('explanation', 'No explanation available')
            verdict_val = fact_check_result.get('verdict', 'Unverifiable')
        else:
            explanation_val = getattr(fact_check_result, 'explanation', 'No explanation available')
            verdict_val = getattr(fact_check_result, 'verdict', 'Unverifiable')
        
        # Prepare evidence assessment
        evidence_data = EvidenceAssessment()
        
        # Debug: Log the fact_check_result structure
        logger.debug(f"fact_check_result type: {type(fact_check_result)}")
        if isinstance(fact_check_result, dict):
            logger.debug(f"fact_check_result keys: {list(fact_check_result.keys())}")
        else:
            logger.debug(f"fact_check_result attributes: {[attr for attr in dir(fact_check_result) if not attr.startswith('_')]}")
        
        # Extract evidence from different possible sources
        details = None
        if hasattr(fact_check_result, 'rating_details') and fact_check_result.rating_details:
            details = fact_check_result.rating_details
        elif isinstance(fact_check_result, dict):
            # Check for details in the dict structure
            if 'details' in fact_check_result:
                details = fact_check_result['details']
            elif 'rating_details' in fact_check_result:
                details = fact_check_result['rating_details']
            else:
                # Look for direct evidence fields in the result
                details = fact_check_result
        
        if details:
            evidence_data.primary_claims = details.get('primary_claims_extracted', details.get('primary_claims', []))
            evidence_data.claim_evaluations = details.get('claim_evaluations', details.get('evidence_assessment', {}))
            evidence_data.web_sources_consulted = details.get('web_sources_consulted', [])
            evidence_data.reasoning = details.get('reasoning', '')
        
        # Always ensure evidence data is populated with meaningful content
        if not evidence_data.primary_claims:
            evidence_data.primary_claims = [claim]  # Use the original claim
            
        if not evidence_data.reasoning:
            evidence_data.reasoning = explanation_val
            
        if not evidence_data.claim_evaluations:
            evidence_data.claim_evaluations = {claim: verdict_val}
            
        # Always try to extract web sources from the sources list  
        if not evidence_data.web_sources_consulted:
            web_sources = []
            if isinstance(fact_check_result, dict) and 'sources' in fact_check_result:
                web_sources = [s.get('name', '') for s in fact_check_result['sources'] if s.get('type') == 'Web Source']
            elif hasattr(fact_check_result, 'sources') and fact_check_result.sources:
                web_sources = [s.get('name', '') for s in fact_check_result.sources if s.get('type') == 'Web Source']
            evidence_data.web_sources_consulted = web_sources
        
        # Prepare sources
        sources_data = []
        if hasattr(fact_check_result, 'sources') and fact_check_result.sources:
            for source in fact_check_result.sources:
                sources_data.append(SourceInfo(
                    name=source.get('name', ''),
                    type=source.get('type', ''),
                    url=source.get('url'),
                    rating=source.get('rating')
                ))
        elif isinstance(fact_check_result, dict) and 'sources' in fact_check_result:
            for source in fact_check_result['sources']:
                sources_data.append(SourceInfo(
                    name=source.get('name', ''),
                    type=source.get('type', ''),
                    url=source.get('url'),
                    rating=source.get('rating')
                ))
        
        # Prepare credibility metrics
        credibility_analysis = fact_check_result.get('credibility_analysis', {})
        credibility_data = CredibilityMetrics(
            language_quality=credibility_analysis.get('credibility_score', 0.5),
            audio_authenticity=1.0 - prosody_features.get('sarcasm_probability', 0.0) if prosody_features else 0.5,
            source_reliability=0.8 if sources_data else 0.3,
            factual_accuracy=fact_check_result.get('confidence', 0.5) if isinstance(fact_check_result, dict) else getattr(fact_check_result, 'confidence', 0.5),
            flags=credibility_analysis.get('flags', [])
        )
        
        # Generate dynamic debate content using LLM
        debate_data = None
        if evidence_data.primary_claims and is_api_available("openai"):
            try:
                debate_data = await generate_dynamic_debate_content(claim, verdict_val, explanation_val, evidence_data)
                logger.info(f"✅ Generated dynamic debate content")
            except Exception as e:
                logger.warning(f"Failed to generate debate content: {e}")
                debate_data = None
        
        stage_times["structuring_done"] = time.time()
        processing_time = time.time() - start_time
        
        # Log timing breakdown
        logger.info(f"⏱️ [{request_id}] TIMING BREAKDOWN:")
        if "audio_processed" in stage_times:
            logger.info(f"   Audio Processing: {stage_times['audio_processed'] - stage_times['start']:.2f}s")
        if "transcription_done" in stage_times:
            logger.info(f"   Transcription: {stage_times['transcription_done'] - stage_times.get('audio_processed', stage_times['start']):.2f}s")
        if "prosody_done" in stage_times:
            logger.info(f"   Prosody Analysis: {stage_times['prosody_done'] - stage_times['transcription_done']:.2f}s")
        if "fact_check_done" in stage_times:
            logger.info(f"   Fact-Checking: {stage_times['fact_check_done'] - stage_times['fact_check_start']:.2f}s")
        if "structuring_done" in stage_times:
            logger.info(f"   Data Structuring: {stage_times['structuring_done'] - stage_times['structuring_start']:.2f}s")
        logger.info(f"   TOTAL TIME: {processing_time:.2f}s")
        
        # Extract confidence and verdict
        if isinstance(fact_check_result, dict):
            confidence_val = fact_check_result.get('confidence', 0.0)
            verdict_val = fact_check_result.get('verdict', 'Unknown')
            explanation_val = fact_check_result.get('explanation', 'No explanation available')
            provider_val = fact_check_result.get('provider', 'Unknown')
        else:
            confidence_val = getattr(fact_check_result, 'confidence', 0.0)
            verdict_val = getattr(fact_check_result, 'verdict', 'Unknown')
            explanation_val = getattr(fact_check_result, 'explanation', 'No explanation available')
            provider_val = getattr(fact_check_result, 'provider', 'Unknown')
        
        result = AnalysisResponse(
            transcription=transcription,
            claim=claim,
            verdict=verdict_val,
            confidence=float(confidence_val) if confidence_val != 'Unknown' else 0.0,
            explanation=explanation_val,
            evidence=evidence_data,
            sources=sources_data,
            prosody=prosody_data,
            credibility_metrics=credibility_data,
            debate_content=debate_data,
            provider=provider_val,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"🎉 [{request_id}] ANALYSIS COMPLETE!")
        logger.info(f"   Final Verdict: {result.verdict}")
        logger.info(f"   Final Confidence: {result.confidence}")
        logger.info(f"   Total Processing Time: {processing_time:.2f}s")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"💥 [{request_id}] ANALYSIS FAILED after {error_time:.2f}s")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"     {line}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-file", summary="Analyze Claim with File Upload")
async def analyze_claim_with_file(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    text_claim: str = Form("", description="Optional text claim to fact-check"),
    enable_prosody: bool = Form(True, description="Enable prosody analysis")
):
    """Analyze a claim with file upload support"""
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info(f"📝 Received file analysis request")
    
    try:
        # Save uploaded file
        logger.info(f"🎤 Processing uploaded audio file: {audio_file.filename}")
        audio_file_path = await save_uploaded_file(audio_file)
        
        # Convert to base64 for processing
        import base64
        with open(audio_file_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        request = AnalysisRequest(
            text_claim=text_claim or "",
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

# Dynamic content generation using LLM
async def generate_dynamic_debate_content(claim: str, verdict: str, explanation: str, evidence: EvidenceAssessment) -> DebateContent:
    """Generate dynamic debate content using OpenAI"""
    
    openai_key = get_api_key("openai")
    if not openai_key:
        return None
    
    prompt = f"""Generate balanced debate content for this fact-checked claim. Provide realistic arguments from both sides, expert perspectives, and context.

CLAIM: "{claim}"
VERDICT: {verdict}
EXPLANATION: {explanation}
PRIMARY CLAIMS: {evidence.primary_claims}

Generate JSON with this structure:
{{
    "supporting_arguments": ["arg1", "arg2", "arg3"],
    "opposing_arguments": ["arg1", "arg2", "arg3"], 
    "expert_opinions": [
        {{"expert": "Organization/Person", "opinion": "Their specific position"}},
        {{"expert": "Organization/Person", "opinion": "Their specific position"}}
    ],
    "historical_context": "Brief historical background of this claim/topic",
    "scientific_consensus": "Current scientific/expert consensus on this topic"
}}

Guidelines:
- Supporting arguments: Present the strongest case FOR the claim (even if verdict is False)
- Opposing arguments: Present evidence/reasoning AGAINST the claim  
- Expert opinions: Include real organizations/experts when possible (NASA, CDC, universities, etc.)
- Make content factual and balanced, not biased toward the verdict
- Focus on the actual claim, not just general topic discussion
- Keep each argument 1-2 sentences, clear and specific"""

    try:
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert at generating balanced debate content. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Parse JSON response
                    try:
                        debate_data = json.loads(content.strip())
                        return DebateContent(
                            supporting_arguments=debate_data.get("supporting_arguments", []),
                            opposing_arguments=debate_data.get("opposing_arguments", []),
                            expert_opinions=debate_data.get("expert_opinions", []),
                            historical_context=debate_data.get("historical_context", ""),
                            scientific_consensus=debate_data.get("scientific_consensus", "")
                        )
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse debate content JSON")
                        return None
                else:
                    logger.error(f"OpenAI API error for debate content: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error generating debate content: {e}")
        return None

@app.get("/api/status")
async def get_api_status():
    """Get detailed API status and configuration"""
    from config import is_api_available
    
    status = {
        "version": "3.0.0-professional",
        "basic_features": {
            "audio_transcription": True,
            "prosody_analysis": True,
            "sentiment_analysis": True,
            "simple_fact_checking": True
        },
        "professional_apis": {
            "google_fact_check": is_api_available("google_fact_check"),
            "openai": is_api_available("openai"),
            "anthropic": is_api_available("anthropic"),
            "news_api": is_api_available("news_api")
        },
        "recommendations": []
    }
    
    # Add recommendations based on missing APIs
    if not status["professional_apis"]["google_fact_check"]:
        status["recommendations"].append("Add GOOGLE_FACT_CHECK_API_KEY for professional fact verification")
    
    if not status["professional_apis"]["openai"]:
        status["recommendations"].append("Add OPENAI_API_KEY for AI-powered claim analysis")
    
    if not any(status["professional_apis"].values()):
        status["recommendations"].append("Currently using basic fact-checking mode. Add API keys for enhanced verification.")
    
    return status

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "total_requests": len(request_cache),
        "cache_size": len(request_cache),
        "uptime": time.time(),
        "version": "3.0.0-professional",
        "apis_available": {
            "google_fact_check": multi_source_checker is not None,
            "openai": multi_source_checker is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
