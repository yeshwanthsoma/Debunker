"""
Enhanced Conspiracy Theory Debunker - Backend API
A FastAPI backend for fact-checking claims with advanced AI analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import asyncio
import logging
import traceback
import time
import hashlib

from fact_checker import EnhancedFactChecker
from models import AnalysisRequest, AnalysisResponse

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

# Initialize fact checker
fact_checker = None
request_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the fact checker on startup"""
    global fact_checker
    logger.info("🚀 Starting Enhanced Fact Checker...")
    try:
        fact_checker = EnhancedFactChecker()
        logger.info("✅ Fact checker initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize fact checker: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Enhanced Conspiracy Theory Debunker API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "fact_checker_ready": fact_checker is not None,
        "timestamp": time.time()
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_claim(request: AnalysisRequest):
    """
    Analyze a claim for fact-checking
    
    Args:
        request: AnalysisRequest containing text_claim, audio_data, and options
    
    Returns:
        AnalysisResponse with comprehensive analysis results
    """
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info(f"📝 Received analysis request for claim: '{request.text_claim[:50]}...'")
    
    # Check cache
    claim_hash = hashlib.md5(request.text_claim.encode()).hexdigest()
    if claim_hash in request_cache and not request.audio_data:
        logger.info("🔍 Using cached result")
        return request_cache[claim_hash]
    
    start_time = time.time()
    
    try:
        # Process audio if provided
        audio_file = None
        if request.audio_data:
            logger.info("🎤 Processing audio data")
            audio_file = await process_audio_data(request.audio_data)
        
        # Perform analysis
        result = await perform_analysis(
            audio_file=audio_file,
            text_claim=request.text_claim,
            enable_prosody=request.enable_prosody
        )
        
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        # Cache result (only for text-only requests)
        if not request.audio_data:
            request_cache[claim_hash] = result
        
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
    """
    Analyze a claim with file upload support
    
    Args:
        audio_file: Optional audio file upload
        text_claim: Text claim to analyze
        enable_prosody: Whether to enable prosody analysis
    
    Returns:
        Analysis results in JSON format
    """
    if not fact_checker:
        raise HTTPException(status_code=503, detail="Fact checker not initialized")
    
    logger.info(f"📝 Received file analysis request")
    
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily if provided
        audio_file_path = None
        if audio_file:
            logger.info(f"🎤 Processing uploaded audio file: {audio_file.filename}")
            audio_file_path = await save_uploaded_file(audio_file)
        
        # Perform analysis
        result = await perform_analysis(
            audio_file=audio_file_path,
            text_claim=text_claim,
            enable_prosody=enable_prosody
        )
        
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        logger.info(f"✅ File analysis completed in {processing_time:.2f}s")
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Error during file analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def process_audio_data(audio_data: str) -> str:
    """Process base64 audio data and return file path"""
    import base64
    import tempfile
    import os
    
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio data")

async def save_uploaded_file(audio_file: UploadFile) -> str:
    """Save uploaded file and return path"""
    import tempfile
    import os
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to save uploaded file")

async def perform_analysis(audio_file: Optional[str], text_claim: str, enable_prosody: bool) -> AnalysisResponse:
    """Perform the actual fact-checking analysis"""
    
    # Initialize variables
    transcription = ""
    claim = text_claim.strip()
    prosody_features = {}
    
    # Process audio if provided
    if audio_file:
        logger.info("🎤 Transcribing audio...")
        
        # Transcribe audio
        transcription_result = fact_checker.transcriber(audio_file, return_timestamps=True)
        transcription = transcription_result['text']
        
        # Prosody analysis
        if enable_prosody:
            logger.info("🎭 Analyzing prosody...")
            prosody_features = fact_checker.extract_advanced_prosody(audio_file)
        
        # Extract claim if not provided
        if not claim:
            logger.info("📝 Extracting claim from transcription...")
            claim_prompt = f"""
            Extract the main factual claim from this transcription. 
            Return only the claim, no explanation:
            
            "{transcription}"
            """
            claim_result = fact_checker.llm.invoke(claim_prompt)
            claim = claim_result.content.strip()
    
    if not claim:
        raise HTTPException(status_code=400, detail="No clear claim found")
    
    # Perform fact-checking
    logger.info("🔍 Performing fact-check...")
    fact_check_result = fact_checker.enhanced_fact_check(claim, prosody_features)
    
    # Generate visualizations
    timeline_html = fact_checker.create_enhanced_timeline(claim, fact_check_result, 0)
    credibility_html = fact_checker.create_credibility_radar(
        fact_check_result.get('credibility_analysis', {}),
        prosody_features
    )
    
    # Format prosody summary
    prosody_summary = format_prosody_summary(prosody_features) if prosody_features else "No audio analysis performed"
    
    # Create response
    return AnalysisResponse(
        transcription=transcription,
        claim=claim,
        analysis=format_analysis_report(fact_check_result, claim),
        timeline=timeline_html,
        credibility=credibility_html,
        prosody=prosody_summary,
        confidence=fact_check_result.get('confidence', 'Unknown'),
        verdict=fact_check_result.get('verdict', 'Unknown'),
        processing_time=0  # Will be set by caller
    )

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
**Emotional Indicators:** {prosody_features.get('emotion_indicators', 'N/A')}
**Pitch Analysis:** Mean: {prosody_features.get('pitch_mean', 0):.1f}Hz, Std: {prosody_features.get('pitch_std', 0):.1f}
**Energy Level:** {prosody_features.get('energy_mean', 0):.3f}
**Speaking Rate:** {prosody_features.get('speaking_rate', 0):.1f} beats/second
"""

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "total_requests": len(request_cache),
        "cache_size": len(request_cache),
        "uptime": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")