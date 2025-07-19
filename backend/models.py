"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import time

class AnalysisRequest(BaseModel):
    """Request model for claim analysis"""
    text_claim: str = Field(..., description="The claim to analyze")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    enable_prosody: bool = Field(True, description="Enable prosody analysis for audio")
    
    class Config:
        schema_extra = {
            "example": {
                "text_claim": "The moon landing was faked",
                "audio_data": None,
                "enable_prosody": True
            }
        }

class AnalysisResponse(BaseModel):
    """Response model for claim analysis results"""
    transcription: str = Field("", description="Audio transcription")
    claim: str = Field("", description="Extracted or provided claim")
    analysis: str = Field("", description="Detailed analysis report")
    timeline: str = Field("", description="HTML timeline visualization")
    credibility: str = Field("", description="HTML credibility radar chart")
    prosody: str = Field("", description="Prosody analysis summary")
    confidence: str = Field("", description="Confidence level")
    verdict: str = Field("", description="Fact-check verdict")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "transcription": "The moon landing was faked by NASA",
                "claim": "The moon landing was faked",
                "analysis": "# Analysis Report...",
                "timeline": "<div>Timeline HTML</div>",
                "credibility": "<div>Credibility HTML</div>",
                "prosody": "**Sarcasm Probability:** 15%...",
                "confidence": "High",
                "verdict": "False",
                "processing_time": 2.5
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    fact_checker_ready: bool = Field(..., description="Whether fact checker is initialized")
    timestamp: float = Field(..., description="Response timestamp")

class StatsResponse(BaseModel):
    """Statistics response model"""
    total_requests: int = Field(..., description="Total number of requests processed")
    cache_size: int = Field(..., description="Number of cached results")
    uptime: float = Field(..., description="Service uptime in seconds")