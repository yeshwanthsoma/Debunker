"""
TruthLens - Professional Fact-Checking Backend API
Enhanced with Google Fact Check Tools API and OpenAI integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
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
import aiohttp
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import requests
from contextlib import asynccontextmanager
from config import get_settings, is_api_available, get_api_key
from fact_check_apis import MultiSourceFactChecker
from auth import verify_credentials, get_current_user
from models import (
    AnalysisResponse, ProsodyAnalysis, SourceInfo,
    EvidenceAssessment, DebateContent, CredibilityMetrics, ExpertOpinion,
    ErrorResponse, HealthResponse, StatsResponse,
    TrendingClaimResponse, TrendingClaimsListResponse, TrendingClaimDetailResponse,
    AggregationTriggerResponse, CategoryStatsResponse
)
from database import get_db, TrendingClaim, ClaimSource, ClaimAnalytics, init_db
from news_aggregator import NewsAggregator, save_claims_to_database
from grok_integration import GrokSocialAnalyzer, enhance_trending_claim_with_grok
from scheduler import start_background_scheduler, stop_background_scheduler, get_scheduler
from startup_aggregation import run_startup_aggregation
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Configure logging
import uuid
from datetime import datetime, timedelta

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Request tracking
request_counter = 0

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(
    title="TruthLens - Professional Fact-Checking API",
    description="Advanced AI-powered fact-checking with Google Fact Check Tools API, OpenAI integration, and audio analysis",
    version="3.0.1",
    lifespan=lifespan
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS - Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Enhanced AnalysisRequest with input validation
class AnalysisRequest(BaseModel):
    text_claim: str = Field(..., description="The claim to analyze")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    enable_prosody: bool = Field(True, description="Enable prosody analysis for audio")
    
    @field_validator('text_claim')
    @classmethod
    def validate_text_claim(cls, v):
        # Allow empty text_claim - will be validated in model_validator
        if v and len(v.strip()) > 10000:
            raise ValueError('Text claim is too long (maximum 10,000 characters)')
        return v.strip() if v else ""
    
    @model_validator(mode='after')
    def validate_claim_or_audio(self):
        # Require either text_claim or audio_data to be provided
        if not self.text_claim and not self.audio_data:
            raise ValueError('Either text_claim or audio_data must be provided')
        
        # If text_claim is provided, validate its length
        if self.text_claim and len(self.text_claim) < 3:
            raise ValueError('Text claim must be at least 3 characters long')
        
        return self
    
    @field_validator('audio_data')
    @classmethod
    def validate_audio_data(cls, v):
        if v and len(v) > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError('Audio data is too large (maximum 50MB)')
        return v

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
            # Check if OpenAI API is available for Whisper
            from config import is_api_available, get_api_key
            self.openai_key = get_api_key("openai")
            self.use_openai_whisper = is_api_available("openai")
            
            if self.use_openai_whisper:
                logger.info("✅ Using OpenAI Whisper API for transcription")
            else:
                logger.warning("⚠️ OpenAI API not available, falling back to local Whisper")
                # Fallback to local model only if OpenAI is not available
                self.transcriber = pipeline(
                    "automatic-speech-recognition", 
                    model="openai/whisper-tiny",  # Use tiny for faster fallback
                    device=-1
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
    
    async def transcribe_audio_openai(self, audio_file_path: str) -> str:
        """Transcribe audio using OpenAI Whisper API with timeout"""
        try:
            logger.info("🗣️ Transcribing audio with OpenAI Whisper API...")
            
            # Add timeout for API call using thread executor
            def _transcribe():
                # Read the audio file
                with open(audio_file_path, 'rb') as audio_file:
                    # Use the openai client directly
                    import openai
                    client = openai.OpenAI(api_key=self.openai_key, timeout=30.0)
                    
                    # Call Whisper API
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                    
                    return transcript.strip()
            
            # Run in thread executor with timeout
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                transcript = await asyncio.wait_for(
                    loop.run_in_executor(executor, _transcribe), 
                    timeout=60.0
                )
            logger.info(f"✅ OpenAI Whisper API transcription completed")
            return transcript
                
        except asyncio.TimeoutError:
            logger.error("❌ OpenAI Whisper API timeout (60 seconds)")
            # Fallback to local transcription if API times out
            if hasattr(self, 'transcriber'):
                logger.info("🔄 Falling back to local Whisper model due to timeout...")
                result = self.transcriber(audio_file_path, return_timestamps=True)
                return result['text']
            else:
                raise Exception("OpenAI Whisper API timeout and no local fallback available")
        except Exception as e:
            logger.error(f"❌ OpenAI Whisper API error: {e}")
            # Fallback to local transcription if API fails
            if hasattr(self, 'transcriber'):
                logger.info("🔄 Falling back to local Whisper model...")
                result = self.transcriber(audio_file_path, return_timestamps=True)
                return result['text']
            else:
                raise Exception(f"Both OpenAI Whisper API and local fallback failed: {e}")
    
    def transcribe_audio_local(self, audio_file_path: str) -> str:
        """Transcribe audio using local Whisper model"""
        logger.info("🗣️ Transcribing audio with local Whisper model...")
        result = self.transcriber(audio_file_path, return_timestamps=True)
        return result['text']
    
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
            
        # Initialize database
        logger.info("🗄️ Initializing database...")
        try:
            init_db()
            logger.info("✅ Database initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            # Don't raise - continue without database features
        
        # Run startup aggregation to get real data immediately  
        logger.info("🔍 Running startup trending claims discovery...")
        try:
            from news_aggregator import NewsAggregator, save_claims_to_database
            
            # Create aggregator and discover claims immediately
            aggregator = NewsAggregator()
            aggregator.use_grok_at_startup = False  # Skip Grok for faster startup
            
            logger.info("📰 Discovering fresh trending claims...")
            claims = await aggregator.discover_trending_claims(limit=15)
            
            if claims:
                saved_count = await save_claims_to_database(claims)
                logger.info(f"💾 Saved {saved_count} new claims to database")
                
                # Simple fact-check for immediate availability
                if saved_count > 0:
                    from scheduler import TrendingClaimsScheduler
                    scheduler_instance = TrendingClaimsScheduler()
                    fact_checked = await scheduler_instance._professional_fact_check_new_claims()
                    logger.info(f"✅ Startup complete: {saved_count} claims discovered, {fact_checked} fact-checked!")
                else:
                    logger.info("ℹ️ No new claims (duplicates of existing)")
            else:
                logger.warning("⚠️ No claims discovered at startup - will retry via background scheduler")
                
        except Exception as e:
            logger.error(f"❌ Startup aggregation failed: {e}")
            logger.info("🔄 Background scheduler will handle discovery automatically")
            # Continue without startup data
        
        # Start background scheduler for trending claims
        logger.info("⏰ Starting background scheduler...")
        try:
            await start_background_scheduler()
            logger.info("✅ Background scheduler started successfully!")
        except Exception as e:
            logger.error(f"❌ Background scheduler failed to start: {e}")
            # Continue without scheduler
        
        logger.info("=" * 60)
        logger.info("🎉 TruthLens startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize fact checker: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

async def shutdown_event():
    """Clean up on shutdown"""
    global multi_source_checker
    
    # Stop background scheduler
    try:
        await stop_background_scheduler()
        logger.info("⏰ Background scheduler stopped")
    except Exception as e:
        logger.error(f"❌ Error stopping scheduler: {e}")
    
    # Close API sessions
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

# Explicit OPTIONS handler for CORS preflight
@app.options("/api/analyze-file")
async def options_analyze_file():
    return {"message": "OK"}

@app.options("/api/analyze")
async def options_analyze():
    return {"message": "OK"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "fact_checker_ready": fact_checker is not None,
        "timestamp": time.time()
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
@limiter.limit("30/minute")
async def analyze_claim(
    request: Request, 
    analysis_request: AnalysisRequest,
    authenticated: bool = Depends(verify_credentials)
):
    """Analyze a claim for fact-checking (requires authentication)"""
    global request_counter
    request_counter += 1
    request_id = f"REQ-{request_counter:04d}"
    
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info("=" * 80)
    logger.info(f"🎯 NEW FACT-CHECK REQUEST [{request_id}]")
    logger.info(f"📝 Claim: '{analysis_request.text_claim[:100]}{'...' if len(analysis_request.text_claim) > 100 else ''}'")
    logger.info(f"🎤 Audio Data: {'Yes' if analysis_request.audio_data else 'No'}")
    logger.info(f"🎵 Prosody Analysis: {'Enabled' if analysis_request.enable_prosody else 'Disabled'}")
    logger.info(f"🔌 Professional APIs: {'Available' if multi_source_checker else 'Not Available'}")
    
    start_time = time.time()
    stage_times = {"start": start_time}
    
    try:
        # Stage 1: Audio Processing
        audio_file = None
        transcription = ""
        prosody_features = {}
        
        if analysis_request.audio_data:
            logger.info(f"🎤 [{request_id}] STAGE 1: Processing audio data...")
            audio_file = await process_audio_data(analysis_request.audio_data)
            stage_times["audio_processed"] = time.time()
            logger.debug(f"   Audio file saved: {audio_file}")
            
            # Transcribe
            logger.info(f"🗣️ [{request_id}] STAGE 2: Transcribing audio...")
            if fact_checker.use_openai_whisper:
                transcription = await fact_checker.transcribe_audio_openai(audio_file)
            else:
                transcription = fact_checker.transcribe_audio_local(audio_file)
            stage_times["transcription_done"] = time.time()
            logger.info(f"   Transcription: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            
            # Prosody analysis
            if analysis_request.enable_prosody:
                logger.info(f"🎵 [{request_id}] STAGE 3: Analyzing prosody features...")
                prosody_features = fact_checker.extract_advanced_prosody(audio_file)
                stage_times["prosody_done"] = time.time()
                logger.debug(f"   Prosody features: {prosody_features}")
                logger.info(f"   Sarcasm probability: {prosody_features.get('sarcasm_probability', 0):.2%}")
        else:
            logger.info(f"📝 [{request_id}] No audio data provided, using text claim only")
        
        # Stage 4: Claim Extraction
        claim = analysis_request.text_claim.strip()
        if not claim and transcription:
            claim = transcription.strip()
            logger.info(f"📋 [{request_id}] Using transcribed claim")
        
        if not claim:
            logger.error(f"❌ [{request_id}] No clear claim found")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Invalid Input",
                    "message": "No clear claim found to analyze. Please provide a text claim or upload audio containing a factual statement.",
                    "suggestion": "Try phrases like 'The Earth is flat' or 'Vaccines cause autism' for testing."
                }
            )
        
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
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"💥 [{request_id}] ANALYSIS FAILED after {error_time:.2f}s")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"     {line}")
        logger.info("=" * 80)
        
        # Provide user-friendly error messages
        if "timeout" in str(e).lower():
            raise HTTPException(
                status_code=408, 
                detail={
                    "error": "Request Timeout",
                    "message": "The analysis took too long to complete. This may be due to high server load or complex audio processing.",
                    "suggestion": "Please try again with a shorter audio clip or simpler text claim."
                }
            )
        elif "api" in str(e).lower() or "key" in str(e).lower():
            raise HTTPException(
                status_code=503, 
                detail={
                    "error": "Service Unavailable",
                    "message": "External fact-checking services are temporarily unavailable. The system will use basic fact-checking instead.",
                    "suggestion": "Please try again later for enhanced verification."
                }
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Analysis Error",
                    "message": "An unexpected error occurred during analysis. Our team has been notified.",
                    "suggestion": "Please try again or contact support if the issue persists."
                }
            )

@app.post("/api/analyze-file", summary="Analyze Claim with File Upload")
async def analyze_claim_with_file(
    request: Request,
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    text_claim: str = Form("", description="Optional text claim to fact-check"),
    authenticated: bool = Depends(verify_credentials),
    enable_prosody: bool = Form(True, description="Enable prosody analysis")
):
    """Analyze a claim with file upload support (requires authentication)"""
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
        
        analysis_request = AnalysisRequest(
            text_claim=text_claim or "",
            audio_data=audio_data,
            enable_prosody=enable_prosody
        )
        
        result = await analyze_claim(request, analysis_request)
        
        # Clean up temporary file
        if audio_file_path and os.path.exists(audio_file_path):
            os.unlink(audio_file_path)
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"❌ Error during file analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def process_audio_data(audio_data: str) -> str:
    """Process base64 audio data and return file path with timeout"""
    import base64
    
    try:
        # Add timeout for audio processing
        async def _process_audio():
            audio_bytes = base64.b64decode(audio_data)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                return temp_file.name
        
        return await asyncio.wait_for(_process_audio(), timeout=30.0)
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, 
            detail={
                "error": "Audio Processing Timeout",
                "message": "Audio processing took too long to complete.",
                "suggestion": "Please try with a smaller audio file (under 10MB)."
            }
        )
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Invalid Audio Data",
                "message": "The provided audio data could not be processed. Please ensure it's valid base64-encoded audio.",
                "suggestion": "Supported formats: WAV, MP3, M4A, OGG"
            }
        )

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
        "version": "3.0.1-enhanced",
        "apis_available": {
            "google_fact_check": multi_source_checker is not None,
            "openai": multi_source_checker is not None
        }
    }

@app.get("/docs-api")
async def get_api_documentation():
    """Get comprehensive API documentation"""
    return {
        "title": "TruthLens Professional Fact-Checking API",
        "version": "3.0.1-enhanced",
        "description": "Advanced AI-powered fact-checking with Google Fact Check Tools API, OpenAI integration, and audio analysis",
        "base_url": "http://localhost:8080",
        "rate_limits": {
            "analyze_endpoint": "30 requests per minute per IP",
            "other_endpoints": "100 requests per minute per IP"
        },
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint showing API status and features",
                "parameters": "None",
                "response": "API metadata and feature availability"
            },
            "/health": {
                "method": "GET", 
                "description": "Health check endpoint",
                "parameters": "None",
                "response": "Service health status"
            },
            "/api/analyze": {
                "method": "POST",
                "description": "Analyze a claim for fact-checking",
                "rate_limit": "30/minute",
                "content_type": "application/json",
                "parameters": {
                    "text_claim": {
                        "type": "string",
                        "required": True,
                        "description": "The factual claim to analyze",
                        "min_length": 3,
                        "max_length": 10000,
                        "examples": ["The Earth is flat", "Vaccines cause autism", "Climate change is real"]
                    },
                    "audio_data": {
                        "type": "string",
                        "required": False,
                        "description": "Base64 encoded audio data (WAV, MP3, M4A, OGG)",
                        "max_size": "50MB"
                    },
                    "enable_prosody": {
                        "type": "boolean",
                        "required": False,
                        "default": True,
                        "description": "Enable prosody analysis for audio (sarcasm detection, etc.)"
                    }
                },
                "response": {
                    "transcription": "Audio transcription if provided",
                    "claim": "Extracted or provided claim",
                    "verdict": "True/False/Partially True/Misleading/Unverifiable",
                    "confidence": "Confidence score (0-1)",
                    "explanation": "Detailed explanation of the verdict",
                    "evidence": "Evidence assessment with sources and reasoning",
                    "sources": "List of sources used in fact-checking",
                    "prosody": "Audio analysis results if enabled",
                    "credibility_metrics": "Credibility assessment scores",
                    "processing_time": "Analysis time in seconds"
                }
            },
            "/api/analyze-file": {
                "method": "POST",
                "description": "Analyze claim with file upload",
                "content_type": "multipart/form-data",
                "parameters": {
                    "audio_file": {
                        "type": "file",
                        "required": True,
                        "description": "Audio file to analyze",
                        "supported_formats": ["WAV", "MP3", "M4A", "OGG"]
                    },
                    "text_claim": {
                        "type": "string",
                        "required": False,
                        "description": "Optional text claim to fact-check"
                    },
                    "enable_prosody": {
                        "type": "boolean",
                        "required": False,
                        "default": True
                    }
                }
            },
            "/api/status": {
                "method": "GET",
                "description": "Get detailed API status and configuration",
                "response": "API capabilities and recommendations"
            },
            "/api/stats": {
                "method": "GET",
                "description": "Get API usage statistics",
                "response": "Request counts, cache size, uptime"
            }
        },
        "error_codes": {
            "400": {
                "description": "Bad Request",
                "common_causes": [
                    "Empty or invalid text claim",
                    "Invalid audio data format",
                    "Request validation failed"
                ]
            },
            "408": {
                "description": "Request Timeout",
                "common_causes": [
                    "Audio processing took too long",
                    "Analysis exceeded time limits"
                ]
            },
            "429": {
                "description": "Too Many Requests",
                "common_causes": [
                    "Rate limit exceeded (30/minute for /api/analyze)"
                ]
            },
            "503": {
                "description": "Service Unavailable", 
                "common_causes": [
                    "External API services down",
                    "Fact checker not initialized"
                ]
            },
            "500": {
                "description": "Internal Server Error",
                "common_causes": [
                    "Unexpected analysis failure",
                    "Model loading issues"
                ]
            }
        },
        "examples": {
            "simple_text_claim": {
                "url": "/api/analyze",
                "method": "POST",
                "body": {
                    "text_claim": "The Earth is flat",
                    "enable_prosody": False
                }
            },
            "audio_with_text": {
                "url": "/api/analyze", 
                "method": "POST",
                "body": {
                    "text_claim": "Climate change is a hoax",
                    "audio_data": "base64_encoded_audio_data_here",
                    "enable_prosody": True
                }
            },
            "file_upload": {
                "url": "/api/analyze-file",
                "method": "POST",
                "content_type": "multipart/form-data",
                "fields": {
                    "audio_file": "audio.wav",
                    "text_claim": "Optional claim text",
                    "enable_prosody": "true"
                }
            }
        },
        "features": {
            "basic_fact_checking": "Built-in knowledge base for common conspiracy theories",
            "professional_apis": "Google Fact Check API and OpenAI integration when configured",
            "audio_analysis": "Whisper-based transcription and prosody analysis",
            "sarcasm_detection": "Audio-based sarcasm and emotion detection",
            "multi_source_verification": "Cross-reference multiple fact-checking sources",
            "rate_limiting": "Protection against abuse with per-IP limits",
            "timeout_handling": "Graceful handling of long-running requests",
            "input_validation": "Comprehensive request validation with helpful error messages"
        },
        "setup_instructions": {
            "basic_mode": "No additional setup required - uses built-in knowledge base",
            "professional_mode": "Add API keys to .env file: GOOGLE_FACT_CHECK_API_KEY, OPENAI_API_KEY",
            "audio_support": "Requires librosa and transformers for audio processing"
        }
    }

# ===================== TRENDING CLAIMS API ENDPOINTS =====================

@app.get("/api/trending-claims", response_model=TrendingClaimsListResponse)
async def get_trending_claims(
    page: int = 1,
    limit: int = 20,
    category: Optional[str] = None,
    status: Optional[str] = None,
    sort: Optional[str] = "hybrid",  # hybrid, recent, trending, processed
    db: Session = Depends(get_db)
):
    """
    Get list of trending claims with pagination, filtering, and sorting options
    
    Sort options:
    - hybrid (default): Trending score + recency boost for variety
    - recent: Most recently discovered claims first
    - trending: Pure trending score (highest scoring first)
    - processed: Most recently fact-checked first
    - popular: Most viewed and shared claims first
    """
    try:
        # Calculate offset
        offset = (page - 1) * limit
        
        # Build query
        query = db.query(TrendingClaim)
        
        # Apply filters
        if category:
            query = query.filter(TrendingClaim.category == category)
        if status:
            query = query.filter(TrendingClaim.status == status)
        
        # Get total count
        total = query.count()
        
        # Apply sorting based on sort parameter
        from sqlalchemy import case, extract
        current_time = datetime.utcnow()
        
        if sort == "recent":
            # Sort by most recently discovered
            claims = query.order_by(
                desc(TrendingClaim.discovered_at),
                desc(TrendingClaim.trending_score)
            ).offset(offset).limit(limit).all()
            
        elif sort == "trending":
            # Sort by pure trending score (original behavior)
            claims = query.order_by(
                desc(TrendingClaim.trending_score),
                desc(TrendingClaim.view_count)
            ).offset(offset).limit(limit).all()
            
        elif sort == "processed":
            # Sort by most recently fact-checked
            claims = query.order_by(
                desc(TrendingClaim.processed_at),
                desc(TrendingClaim.confidence)
            ).offset(offset).limit(limit).all()
            
        elif sort == "popular":
            # Sort by engagement (views + shares)
            claims = query.order_by(
                desc(TrendingClaim.view_count + TrendingClaim.share_count * 5),
                desc(TrendingClaim.trending_score)
            ).offset(offset).limit(limit).all()
            
        else:  # sort == "hybrid" (default)
            # Hybrid score: trending + recency boost
            claims = query.order_by(
                desc(
                    TrendingClaim.trending_score + 
                    case(
                        # Boost for claims discovered in last 24 hours
                        (extract('epoch', current_time - TrendingClaim.discovered_at) < 86400, 0.3),
                        # Smaller boost for claims discovered in last 7 days  
                        (extract('epoch', current_time - TrendingClaim.discovered_at) < 604800, 0.1),
                        # No boost for older claims
                        else_=0.0
                    )
                ),
                # Secondary sort by most recently processed
                desc(TrendingClaim.processed_at)
            ).offset(offset).limit(limit).all()
        
        # Get available categories
        categories = db.query(TrendingClaim.category).distinct().all()
        categories = [cat[0] for cat in categories]
        
        # Convert to response format
        claim_responses = []
        for claim in claims:
            claim_responses.append(TrendingClaimResponse(
                id=claim.id,
                claim_text=claim.claim_text,
                title=claim.title,
                category=claim.category,
                verdict=claim.verdict,
                confidence=claim.confidence,
                explanation=claim.explanation[:200] + "..." if claim.explanation and len(claim.explanation) > 200 else claim.explanation,
                source_type=claim.source_type,
                trending_score=claim.trending_score,
                view_count=claim.view_count,
                share_count=claim.share_count,
                status=claim.status,
                discovered_at=claim.discovered_at.isoformat(),
                processed_at=claim.processed_at.isoformat() if claim.processed_at else None,
                tags=claim.tags
            ))
        
        return TrendingClaimsListResponse(
            claims=claim_responses,
            total=total,
            page=page,
            limit=limit,
            categories=categories
        )
        
    except Exception as e:
        logger.error(f"Error fetching trending claims: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trending claims")

@app.get("/api/trending-claims/{claim_id}", response_model=TrendingClaimDetailResponse)
async def get_trending_claim_detail(
    claim_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific trending claim"""
    try:
        # Get claim
        claim = db.query(TrendingClaim).filter(TrendingClaim.id == claim_id).first()
        
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        # Increment view count
        claim.view_count += 1
        db.commit()
        
        # Get sources - separate fact-checking sources from original news sources
        all_sources = db.query(ClaimSource).filter(ClaimSource.claim_id == claim_id).all()
        
        # Separate sources by type for better organization
        fact_check_sources = []
        original_sources = []
        
        for source in all_sources:
            source_info = {
                "name": source.source_name,
                "url": source.source_url,
                "type": source.source_type,
                "author": source.author,
                "published_at": source.published_at.isoformat() if source.published_at else None,
                "reliability": source.source_reliability
            }
            
            if source.source_type == 'fact_check_source':
                fact_check_sources.append(source_info)
            else:
                original_sources.append(source_info)
        
        # Prioritize fact-checking sources for display (what user expects to see)
        source_data = fact_check_sources + original_sources
        
        return TrendingClaimDetailResponse(
            id=claim.id,
            claim_text=claim.claim_text,
            title=claim.title,
            category=claim.category,
            verdict=claim.verdict,
            confidence=claim.confidence,
            explanation=claim.explanation,
            evidence_summary=claim.evidence_summary,
            source_type=claim.source_type,
            source_url=claim.source_url,
            trending_score=claim.trending_score,
            controversy_level=claim.controversy_level,
            view_count=claim.view_count,
            share_count=claim.share_count,
            status=claim.status,
            processing_time=claim.processing_time,
            discovered_at=claim.discovered_at.isoformat(),
            processed_at=claim.processed_at.isoformat() if claim.processed_at else None,
            tags=claim.tags,
            keywords=claim.keywords,
            related_entities=claim.related_entities,
            sources=source_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching claim detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch claim details")

@app.post("/api/trigger-aggregation", response_model=AggregationTriggerResponse)
@limiter.limit("5/minute")  # Limit to prevent abuse
async def trigger_news_aggregation(
    request: Request,
    authenticated: bool = Depends(verify_credentials)
):
    """Manually trigger news aggregation and claim discovery"""
    try:
        start_time = time.time()
        
        logger.info("🔍 Manual news aggregation triggered")
        
        # Initialize aggregator
        aggregator = NewsAggregator()
        
        # Discover trending claims
        claims = await aggregator.discover_trending_claims(limit=50)
        
        # Save to database
        saved_count = await save_claims_to_database(claims)
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Manual aggregation complete: {len(claims)} discovered, {saved_count} saved")
        
        return AggregationTriggerResponse(
            message="News aggregation completed successfully",
            claims_discovered=len(claims),
            claims_saved=saved_count,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in manual aggregation: {e}")
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")

@app.get("/api/claims/categories")
async def get_claim_categories(db: Session = Depends(get_db)):
    """Get available claim categories with statistics"""
    try:
        # Get category statistics
        stats = db.query(
            TrendingClaim.category,
            func.count(TrendingClaim.id).label('total'),
            func.count(func.nullif(TrendingClaim.verdict, None)).label('processed'),
            func.avg(TrendingClaim.controversy_level).label('avg_controversy'),
            func.max(TrendingClaim.discovered_at).label('most_recent')
        ).group_by(TrendingClaim.category).all()
        
        categories = []
        for stat in stats:
            categories.append(CategoryStatsResponse(
                category=stat.category,
                total_claims=stat.total,
                processed_claims=stat.processed,
                avg_controversy=float(stat.avg_controversy) if stat.avg_controversy else 0.0,
                most_recent=stat.most_recent.isoformat() if stat.most_recent else None
            ))
        
        return {
            "categories": categories,
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch categories")

@app.get("/api/claims/analytics")
async def get_claims_analytics(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get analytics for trending claims"""
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Basic statistics
        total_claims = db.query(TrendingClaim).count()
        new_claims = db.query(TrendingClaim).filter(
            TrendingClaim.discovered_at >= start_date
        ).count()
        processed_claims = db.query(TrendingClaim).filter(
            TrendingClaim.status == 'completed'
        ).count()
        
        # Top categories
        top_categories = db.query(
            TrendingClaim.category,
            func.count(TrendingClaim.id).label('count')
        ).group_by(TrendingClaim.category)\
         .order_by(desc('count')).limit(5).all()
        
        # Trending score distribution
        avg_trending = db.query(func.avg(TrendingClaim.trending_score)).scalar()
        
        return {
            "period_days": days,
            "total_claims": total_claims,
            "new_claims": new_claims,
            "processed_claims": processed_claims,
            "processing_rate": round(processed_claims / total_claims * 100, 1) if total_claims > 0 else 0,
            "avg_trending_score": float(avg_trending) if avg_trending else 0.0,
            "top_categories": [{"category": cat[0], "count": cat[1]} for cat in top_categories]
        }
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")

@app.post("/api/claims/{claim_id}/share")
async def increment_share_count(
    claim_id: int,
    db: Session = Depends(get_db)
):
    """Increment share count for a claim"""
    try:
        claim = db.query(TrendingClaim).filter(TrendingClaim.id == claim_id).first()
        
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        claim.share_count += 1
        db.commit()
        
        return {"message": "Share count updated", "new_count": claim.share_count}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating share count: {e}")
        raise HTTPException(status_code=500, detail="Failed to update share count")

@app.get("/api/claims/{claim_id}/social-context")
async def get_claim_social_context(
    claim_id: int,
    db: Session = Depends(get_db)
):
    """Get Grok-powered social context for a specific claim"""
    try:
        # Get claim
        claim = db.query(TrendingClaim).filter(TrendingClaim.id == claim_id).first()
        
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        # Check if Grok is available
        if not is_api_available("grok"):
            return {
                "available": False,
                "message": "Grok API not configured",
                "recommendation": "Add GROK_API_KEY to enable social context analysis"
            }
        
        # Get enhanced social context
        enhancement = await enhance_trending_claim_with_grok(claim.claim_text, claim.title)
        
        if enhancement:
            return {
                "available": True,
                "claim_id": claim_id,
                "social_context": enhancement.get('social_context', {}),
                "viral_metrics": enhancement.get('viral_metrics', {}),
                "analysis_timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "available": True,
                "claim_id": claim_id,
                "social_context": {},
                "viral_metrics": {},
                "error": "Failed to analyze social context"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting social context: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze social context")

@app.post("/api/claims/{claim_id}/enhance-with-grok")
@limiter.limit("10/minute")
async def enhance_claim_with_grok(
    claim_id: int,
    request: Request,
    db: Session = Depends(get_db),
    authenticated: bool = Depends(verify_credentials)
):
    """Enhance a claim's fact-check with Grok social context"""
    try:
        # Get claim
        claim = db.query(TrendingClaim).filter(TrendingClaim.id == claim_id).first()
        
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        if not is_api_available("grok"):
            raise HTTPException(status_code=503, detail="Grok API not available")
        
        # Only enhance if claim has been fact-checked
        if not claim.verdict:
            raise HTTPException(status_code=400, detail="Claim must be fact-checked first")
        
        logger.info(f"🌐 Enhancing claim {claim_id} with Grok social context")
        
        # Get social enhancement
        async with GrokSocialAnalyzer() as grok:
            social_enhancement = await grok.enhance_fact_check_with_social_context(
                claim.claim_text,
                claim.verdict,
                claim.explanation or ""
            )
        
        # Update claim with social context
        if social_enhancement:
            # Store social context in evidence_summary or create new field
            existing_evidence = claim.evidence_summary or ""
            social_summary = f"\n\n--- SOCIAL CONTEXT ---\n"
            social_summary += f"Social Reception: {social_enhancement.get('social_reception', 'N/A')}\n"
            social_summary += f"Community Response: {social_enhancement.get('community_response', 'N/A')}\n"
            
            if social_enhancement.get('expert_reactions'):
                social_summary += f"Expert Reactions: {'; '.join(social_enhancement['expert_reactions'])}\n"
            
            if social_enhancement.get('platform_actions'):
                social_summary += f"Platform Actions: {social_enhancement['platform_actions']}\n"
            
            claim.evidence_summary = existing_evidence + social_summary
            db.commit()
            
            logger.info(f"✅ Enhanced claim {claim_id} with Grok social context")
            
            return {
                "message": "Claim enhanced with social context",
                "social_enhancement": social_enhancement,
                "updated_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to get social enhancement")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing claim with Grok: {e}")
        raise HTTPException(status_code=500, detail="Enhancement failed")

@app.get("/api/grok/trending-claims")
async def get_grok_trending_claims(
    categories: Optional[str] = None
):
    """Get trending claims discovered by Grok from social media"""
    try:
        if not is_api_available("grok"):
            return {
                "available": False,
                "claims": [],
                "message": "Grok API not configured"
            }
        
        # Parse categories
        category_list = categories.split(',') if categories else ['health', 'politics', 'science']
        
        logger.info(f"🔥 Fetching Grok trending claims for categories: {category_list}")
        
        # Get trending claims from Grok
        async with GrokSocialAnalyzer() as grok:
            trending_claims = await grok.get_trending_misinformation(category_list)
        
        return {
            "available": True,
            "claims": trending_claims,
            "categories_analyzed": category_list,
            "count": len(trending_claims),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Grok trending claims: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trending claims")

@app.get("/api/grok/status")
async def get_grok_status():
    """Get Grok API status and capabilities"""
    return {
        "available": is_api_available("grok"),
        "capabilities": {
            "social_context_analysis": True,
            "viral_metrics": True,
            "trending_misinformation_detection": True,
            "real_time_x_data": True,
            "fact_check_enhancement": True
        },
        "endpoints": {
            "social_context": "/api/claims/{id}/social-context",
            "enhance_fact_check": "/api/claims/{id}/enhance-with-grok",
            "trending_claims": "/api/grok/trending-claims"
        },
        "setup_instructions": "Add GROK_API_KEY to environment variables" if not is_api_available("grok") else "Ready to use"
    }

# ===================== SCHEDULER MANAGEMENT ENDPOINTS =====================

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get background scheduler status and job information"""
    try:
        scheduler = await get_scheduler()
        status = scheduler.get_status()
        
        return {
            "status": status["status"],
            "jobs": status.get("jobs", []),
            "uptime": status.get("uptime"),
            "enabled": True
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {
            "status": "error",
            "jobs": [],
            "uptime": None,
            "enabled": False,
            "error": str(e)
        }

@app.post("/api/scheduler/trigger/{job_id}")
@limiter.limit("5/minute")
async def trigger_scheduler_job(
    request: Request,
    job_id: str,
    authenticated: bool = Depends(verify_credentials)
):
    """Manually trigger a specific scheduled job"""
    try:
        scheduler = await get_scheduler()
        success = await scheduler.trigger_job(job_id)
        
        if success:
            return {
                "message": f"Job '{job_id}' triggered successfully",
                "job_id": job_id,
                "triggered_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger job: {str(e)}")

@app.get("/api/scheduler/jobs")
async def list_scheduler_jobs():
    """List all scheduled jobs with their next run times"""
    try:
        scheduler = await get_scheduler()
        status = scheduler.get_status()
        
        return {
            "jobs": status.get("jobs", []),
            "total_jobs": len(status.get("jobs", [])),
            "scheduler_running": status["status"] == "running"
        }
        
    except Exception as e:
        logger.error(f"Error listing scheduler jobs: {e}")
        return {
            "jobs": [],
            "total_jobs": 0,
            "scheduler_running": False,
            "error": str(e)
        }

# ===================== END TRENDING CLAIMS API ENDPOINTS =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
