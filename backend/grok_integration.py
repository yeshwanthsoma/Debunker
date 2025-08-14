"""
Grok Integration Service for Social Context Analysis
Provides real-time social media context and community reactions
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from config import get_api_key, is_api_available

logger = logging.getLogger(__name__)

@dataclass
class SocialContext:
    """Social context analysis from Grok"""
    community_consensus: str
    trending_status: str
    social_sentiment: float  # -1.0 to 1.0
    viral_indicators: List[str]
    expert_voices: List[Dict[str, str]]
    counter_narratives: List[str]
    related_discussions: List[str]
    confidence: float
    timestamp: str

@dataclass
class ViralMetrics:
    """Viral spread metrics from social media"""
    spread_rate: str
    engagement_type: str
    geographic_spread: List[str]
    demographic_breakdown: Dict[str, float]
    trending_hashtags: List[str]
    influencer_mentions: List[str]

class GrokSocialAnalyzer:
    """Grok-powered social context analyzer"""
    
    def __init__(self):
        self.api_key = get_api_key("grok")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4"
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def is_available(self) -> bool:
        """Check if Grok API is available"""
        return is_api_available("grok")
    
    async def analyze_claim_social_context(self, claim: str, context: str = "") -> Optional[SocialContext]:
        """Analyze social context and community reactions for a claim"""
        if not self.is_available():
            logger.warning("🔑 Grok API key not available")
            return None
        
        try:
            logger.info(f"🌐 Analyzing social context with Grok: '{claim[:50]}...'")
            
            prompt = self._build_social_context_prompt(claim, context)
            
            response_data = await self._call_grok_api(prompt)
            
            if response_data:
                social_context = self._parse_social_context_response(response_data)
                logger.info(f"✅ Grok social analysis complete: {social_context.community_consensus}")
                return social_context
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Grok social context analysis failed: {e}")
            return None
    
    async def get_viral_metrics(self, claim: str) -> Optional[ViralMetrics]:
        """Get viral spread metrics and trending analysis"""
        if not self.is_available():
            return None
        
        try:
            logger.info(f"📊 Getting viral metrics from Grok: '{claim[:50]}...'")
            
            prompt = self._build_viral_metrics_prompt(claim)
            
            response_data = await self._call_grok_api(prompt)
            
            if response_data:
                viral_metrics = self._parse_viral_metrics_response(response_data)
                logger.info(f"✅ Viral metrics analysis complete")
                return viral_metrics
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Grok viral metrics failed: {e}")
            return None
    
    async def get_trending_misinformation(self, categories: List[str] = None) -> List[Dict]:
        """Get currently trending misinformation on X/Twitter"""
        if not self.is_available():
            return []
        
        try:
            logger.info("🔥 Fetching trending misinformation from Grok...")
            
            categories_str = ", ".join(categories) if categories else "all topics"
            
            prompt = f"""Based on real-time X/Twitter data, identify currently trending misinformation or controversial claims in these categories: {categories_str}.

Focus on:
1. Claims spreading rapidly (high engagement)
2. Disputed or controversial statements
3. Potential misinformation or conspiracy theories
4. Claims that would benefit from fact-checking

Return JSON array of trending claims:
[
  {{
    "claim": "specific claim text",
    "category": "category name",
    "viral_score": 0.8,
    "engagement_level": "high/medium/low",
    "controversy_indicators": ["reason1", "reason2"],
    "source_platform": "X/Twitter",
    "first_seen": "2025-08-12T10:00:00Z"
  }}
]

Limit to top 10 most relevant trending claims."""

            response_data = await self._call_grok_api(prompt)
            
            if response_data:
                # Parse JSON response
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                try:
                    trending_claims = json.loads(content)
                    if isinstance(trending_claims, list):
                        logger.info(f"✅ Found {len(trending_claims)} trending claims")
                        return trending_claims
                except json.JSONDecodeError:
                    logger.warning("Failed to parse trending claims JSON")
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Trending misinformation fetch failed: {e}")
            return []
    
    async def enhance_fact_check_with_social_context(self, claim: str, verdict: str, explanation: str) -> Dict:
        """Enhance fact-check results with social context"""
        if not self.is_available():
            return {}
        
        try:
            logger.info(f"🔍 Enhancing fact-check with social context...")
            
            prompt = f"""You have access to real-time X/Twitter data. Enhance this fact-check with social context:

CLAIM: "{claim}"
VERDICT: {verdict}
EXPLANATION: {explanation}

Provide social context in JSON format:
{{
  "social_reception": "How is this claim being received on social media?",
  "community_response": "What are users saying about this?",
  "expert_reactions": ["Expert responses or official statements"],
  "viral_patterns": "How is this spreading? Any concerning patterns?",
  "related_hashtags": ["trending hashtags related to this topic"],
  "counter_evidence_shared": ["User-shared counter-evidence or debunks"],
  "misinformation_variants": ["Related false claims being spread"],
  "platform_actions": "Any platform labels, fact-checks, or moderation actions",
  "credible_voices": ["Credible accounts discussing this topic"],
  "recommendation": "How should this fact-check be communicated given social context?"
}}

Focus on real current social media activity around this claim."""

            response_data = await self._call_grok_api(prompt)
            
            if response_data:
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                try:
                    social_enhancement = json.loads(content)
                    logger.info("✅ Social context enhancement complete")
                    return social_enhancement
                except json.JSONDecodeError:
                    logger.warning("Failed to parse social enhancement JSON")
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Social context enhancement failed: {e}")
            return {}
    
    def _build_social_context_prompt(self, claim: str, context: str) -> str:
        """Build prompt for social context analysis"""
        return f"""You have access to real-time X/Twitter data and social media discussions. Analyze the social context for this claim:

CLAIM: "{claim}"
CONTEXT: {context}

Provide analysis in JSON format:
{{
  "community_consensus": "What's the general consensus among users discussing this?",
  "trending_status": "Is this trending? How viral is it?",
  "social_sentiment": 0.5,
  "viral_indicators": ["indicators of viral spread"],
  "expert_voices": [{{"expert": "verified account/expert", "position": "their stance"}}],
  "counter_narratives": ["opposing viewpoints being shared"],
  "related_discussions": ["related topics being discussed"],
  "confidence": 0.8,
  "timestamp": "{datetime.now().isoformat()}"
}}

Focus on:
1. Current social media activity around this claim
2. Expert and verified account responses  
3. Community fact-checking efforts
4. Viral spread patterns and engagement
5. Counter-evidence being shared

Sentiment scale: -1.0 (very negative) to 1.0 (very positive)
Confidence: 0.0 to 1.0 based on available social data"""
    
    def _build_viral_metrics_prompt(self, claim: str) -> str:
        """Build prompt for viral metrics analysis"""
        return f"""Analyze viral spread patterns for this claim using real-time X/Twitter data:

CLAIM: "{claim}"

Return viral metrics in JSON format:
{{
  "spread_rate": "1000 shares/hour",
  "engagement_type": "mostly skeptical responses",
  "geographic_spread": ["US", "UK", "Canada"],
  "demographic_breakdown": {{"age_18_34": 0.4, "age_35_54": 0.35, "age_55+": 0.25}},
  "trending_hashtags": ["#hashtag1", "#hashtag2"],
  "influencer_mentions": ["@influencer1", "@influencer2"]
}}

Focus on quantifiable viral metrics and engagement patterns."""
    
    async def _call_grok_api(self, prompt: str) -> Optional[Dict]:
        """Make API call to Grok"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Grok API response: {response.status}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Grok API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Grok API timeout")
            return None
        except Exception as e:
            logger.error(f"Grok API call failed: {e}")
            return None
    
    def _parse_social_context_response(self, data: Dict) -> SocialContext:
        """Parse Grok response into SocialContext object"""
        try:
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            parsed = json.loads(content)
            
            return SocialContext(
                community_consensus=parsed.get('community_consensus', 'Unknown'),
                trending_status=parsed.get('trending_status', 'Not trending'),
                social_sentiment=float(parsed.get('social_sentiment', 0.0)),
                viral_indicators=parsed.get('viral_indicators', []),
                expert_voices=parsed.get('expert_voices', []),
                counter_narratives=parsed.get('counter_narratives', []),
                related_discussions=parsed.get('related_discussions', []),
                confidence=float(parsed.get('confidence', 0.5)),
                timestamp=parsed.get('timestamp', datetime.now().isoformat())
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse social context: {e}")
            # Return default values
            return SocialContext(
                community_consensus="Analysis unavailable",
                trending_status="Unknown",
                social_sentiment=0.0,
                viral_indicators=[],
                expert_voices=[],
                counter_narratives=[],
                related_discussions=[],
                confidence=0.1,
                timestamp=datetime.now().isoformat()
            )
    
    def _parse_viral_metrics_response(self, data: Dict) -> ViralMetrics:
        """Parse Grok response into ViralMetrics object"""
        try:
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            parsed = json.loads(content)
            
            return ViralMetrics(
                spread_rate=parsed.get('spread_rate', 'Unknown'),
                engagement_type=parsed.get('engagement_type', 'Mixed'),
                geographic_spread=parsed.get('geographic_spread', []),
                demographic_breakdown=parsed.get('demographic_breakdown', {}),
                trending_hashtags=parsed.get('trending_hashtags', []),
                influencer_mentions=parsed.get('influencer_mentions', [])
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse viral metrics: {e}")
            return ViralMetrics(
                spread_rate="Unknown",
                engagement_type="Unknown",
                geographic_spread=[],
                demographic_breakdown={},
                trending_hashtags=[],
                influencer_mentions=[]
            )

# Convenience functions for integration
async def enhance_trending_claim_with_grok(claim_text: str, title: str) -> Dict:
    """Enhance a trending claim with Grok social context"""
    async with GrokSocialAnalyzer() as grok:
        if not grok.is_available():
            return {}
        
        # Get social context
        social_context = await grok.analyze_claim_social_context(claim_text, title)
        
        # Get viral metrics
        viral_metrics = await grok.get_viral_metrics(claim_text)
        
        enhancement = {}
        
        if social_context:
            enhancement['social_context'] = {
                'community_consensus': social_context.community_consensus,
                'trending_status': social_context.trending_status,
                'social_sentiment': social_context.social_sentiment,
                'expert_voices': social_context.expert_voices,
                'viral_indicators': social_context.viral_indicators,
                'confidence': social_context.confidence
            }
        
        if viral_metrics:
            enhancement['viral_metrics'] = {
                'spread_rate': viral_metrics.spread_rate,
                'engagement_type': viral_metrics.engagement_type,
                'trending_hashtags': viral_metrics.trending_hashtags,
                'geographic_spread': viral_metrics.geographic_spread
            }
        
        return enhancement

async def get_grok_trending_claims(categories: List[str] = None) -> List[Dict]:
    """Get trending claims discovered by Grok from social media"""
    async with GrokSocialAnalyzer() as grok:
        if not grok.is_available():
            return []
        
        return await grok.get_trending_misinformation(categories)

# Example usage and testing
async def main():
    """Test Grok integration"""
    async with GrokSocialAnalyzer() as grok:
        if not grok.is_available():
            print("❌ Grok API not available - check GROK_API_KEY")
            return
        
        # Test social context analysis
        test_claim = "Vaccines cause autism in children"
        
        print(f"🧪 Testing Grok integration with claim: '{test_claim}'")
        
        social_context = await grok.analyze_claim_social_context(test_claim)
        if social_context:
            print(f"✅ Social Context: {social_context.community_consensus}")
            print(f"   Sentiment: {social_context.social_sentiment}")
            print(f"   Trending: {social_context.trending_status}")
        
        # Test trending claims
        trending = await grok.get_trending_misinformation(['health', 'politics'])
        print(f"✅ Found {len(trending)} trending claims from Grok")

if __name__ == "__main__":
    asyncio.run(main())