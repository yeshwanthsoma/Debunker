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
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text_claim": "The moon landing was faked",
                "audio_data": None,
                "enable_prosody": True
            }
        }
    }

class ProsodyAnalysis(BaseModel):
    """Prosody analysis results from audio"""
    sarcasm_probability: float = Field(0.0, description="Probability of sarcasm (0-1)")
    pitch_mean: float = Field(0.0, description="Mean pitch in Hz")
    pitch_std: float = Field(0.0, description="Pitch standard deviation")
    energy_mean: float = Field(0.0, description="Mean energy level")
    speaking_rate: float = Field(0.0, description="Speaking rate in beats/sec")
    tempo: float = Field(0.0, description="Tempo in BPM")

class SourceInfo(BaseModel):
    """Information about a source used in fact-checking"""
    name: str = Field("", description="Source name")
    type: str = Field("", description="Source type (Web, Academic, etc.)")
    url: Optional[str] = Field(None, description="Source URL")
    rating: Optional[str] = Field(None, description="Source reliability rating")

class EvidenceAssessment(BaseModel):
    """Detailed evidence assessment"""
    primary_claims: List[str] = Field(default_factory=list, description="Primary factual claims extracted")
    claim_evaluations: Dict[str, str] = Field(default_factory=dict, description="Individual claim evaluations")
    web_sources_consulted: List[str] = Field(default_factory=list, description="Web sources consulted")
    reasoning: str = Field("", description="Logical reasoning for the assessment")

class ExpertOpinion(BaseModel):
    """Expert opinion on a topic"""
    expert: str = Field("", description="Expert name and credentials")
    opinion: str = Field("", description="Expert's opinion or statement")

class DebateContent(BaseModel):
    """Debate and discussion content around a claim"""
    supporting_arguments: List[str] = Field(default_factory=list, description="Arguments supporting the claim")
    opposing_arguments: List[str] = Field(default_factory=list, description="Arguments opposing the claim")
    expert_opinions: List[ExpertOpinion] = Field(default_factory=list, description="Expert opinions")
    historical_context: str = Field("", description="Historical context and background")
    scientific_consensus: str = Field("", description="Current scientific consensus")

class CredibilityMetrics(BaseModel):
    """Credibility assessment metrics"""
    language_quality: float = Field(0.0, description="Quality of language used (0-1)")
    audio_authenticity: float = Field(0.0, description="Audio authenticity score (0-1)")
    source_reliability: float = Field(0.0, description="Reliability of sources (0-1)")
    factual_accuracy: float = Field(0.0, description="Factual accuracy score (0-1)")
    flags: List[str] = Field(default_factory=list, description="Warning flags detected")

class AnalysisResponse(BaseModel):
    """Response model for claim analysis results"""
    transcription: str = Field("", description="Audio transcription")
    claim: str = Field("", description="Extracted or provided claim")
    verdict: str = Field("", description="Fact-check verdict (True/False/Misleading/Unverifiable)")
    confidence: float = Field(0.0, description="Confidence level (0-1)")
    explanation: str = Field("", description="Detailed explanation of the verdict")
    evidence: EvidenceAssessment = Field(default_factory=EvidenceAssessment, description="Evidence assessment details")
    sources: List[SourceInfo] = Field(default_factory=list, description="Sources used in fact-checking")
    prosody: Optional[ProsodyAnalysis] = Field(None, description="Audio prosody analysis")
    credibility_metrics: CredibilityMetrics = Field(default_factory=CredibilityMetrics, description="Credibility assessment")
    debate_content: Optional[DebateContent] = Field(None, description="Debate and discussion content")
    provider: str = Field("", description="Fact-checking provider used")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    timestamp: str = Field("", description="Analysis timestamp")

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