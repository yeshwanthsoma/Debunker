"""
Professional Fact-Checking API Integrations for TruthLens
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import get_api_key, is_api_available

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FactCheckResult:
    """Standardized fact-check result"""
    claim: str
    verdict: str  # "True", "False", "Misleading", "Unverifiable"
    confidence: float  # 0.0 to 1.0
    explanation: str
    sources: List[Dict[str, Any]]
    provider: str
    timestamp: datetime
    rating_details: Optional[Dict] = None

class GoogleFactCheckAPI:
    """Google Fact Check Tools API integration"""
    
    def __init__(self):
        self.api_key = get_api_key("google_fact_check")
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.session = None
        # Initialize OpenAI for claim shortening
        self.openai_key = get_api_key("openai")
    
    async def _extract_clean_claims(self, original_text: str) -> List[str]:
        """Use LLM to extract clean, complete factual claims from conversational text"""
        if not is_api_available("openai"):
            # Fallback: return original text
            return [original_text]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Extract the main factual claims from this text. Keep claims complete and specific, but remove conversational elements.

Text: "{original_text}"

Rules:
- Extract COMPLETE factual assertions (not just keywords)
- Remove conversational words like "apparently", "actually", "people say"
- Remove corrections and commentary ("that's been debunked", "proven wrong")
- Keep the core factual statement intact
- Maximum 2-3 claims

Examples:
Input: "Great Wall of China is visible from space. Actually, NASA debunked that."
Output: ["Great Wall of China is visible from space"]

Input: "Vaccines cause autism in children. Multiple studies have disproven this."
Output: ["Vaccines cause autism in children"]

Input: "The moon landing was faked by Hollywood. That's a conspiracy theory though."
Output: ["The moon landing was faked by Hollywood"]

Respond with JSON array: ["claim1", "claim2"]"""

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.1
            }
            
            session = await self._get_session()
            async with session.post("https://api.openai.com/v1/chat/completions", 
                                  json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    try:
                        import json
                        clean_claims = json.loads(content)
                        if isinstance(clean_claims, list) and clean_claims:
                            logger.info(f"🔍 Extracted {len(clean_claims)} clean claims from conversational text")
                            for i, claim in enumerate(clean_claims):
                                logger.debug(f"   Claim {i+1}: '{claim}'")
                            return clean_claims
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM response for claim extraction")
                
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
        
        # Fallback: return original text cleaned of obvious conversational elements
        import re
        cleaned = re.sub(r'\b(apparently|actually|people say|that\'s been|proven wrong|debunked|though)\b', '', original_text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return [cleaned] if cleaned else [original_text]
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
    
    async def search_claims(self, query: str, language_code: str = "en", max_age_days: int = 365) -> List[FactCheckResult]:
        """Search for fact-checked claims using Google Fact Check Tools API with claim extraction"""
        if not is_api_available("google_fact_check"):
            logger.warning("🔑 Google Fact Check API key not available")
            return []
        
        logger.info(f"🌐 CALLING Google Fact Check Tools API")
        logger.info(f"   Original Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        # Extract clean claims from conversational text
        clean_claims = await self._extract_clean_claims(query)
        
        all_results = []
        
        # Search for each extracted claim
        try:
            session = await self._get_session()
            
            for claim in clean_claims:
                logger.info(f"🔍 Searching Google for: '{claim}'")
                
                params = {
                    "key": self.api_key,
                    "query": claim,
                    "languageCode": language_code,
                    "maxAgeDays": max_age_days,
                    "pageSize": 10
                }
                
                import time
                start_time = time.time()
                
                async with session.get(self.base_url, params=params) as response:
                    api_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_google_results(data, claim)
                        if results:
                            logger.info(f"✅ Found {len(results)} Google results for '{claim}' (took {api_time:.2f}s)")
                            all_results.extend(results)
                        else:
                            logger.debug(f"📡 No Google results for '{claim}' (took {api_time:.2f}s)")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Google API error {response.status} for '{claim}': {error_text[:100]}...")
            
            logger.info(f"📊 Total Google fact-check results: {len(all_results)}")
            return all_results
                    
        except Exception as e:
            logger.error(f"💥 Error calling Google Fact Check API: {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
            return []
    
    def _parse_google_results(self, data: Dict, original_query: str) -> List[FactCheckResult]:
        """Parse Google Fact Check API response"""
        results = []
        claims = data.get("claims", [])
        
        for claim_data in claims:
            try:
                claim_text = claim_data.get("text", "")
                claim_reviews = claim_data.get("claimReview", [])
                
                for review in claim_reviews:
                    # Extract verdict and normalize it
                    rating = review.get("textualRating", "").lower()
                    verdict = self._normalize_verdict(rating)
                    
                    # Calculate confidence based on publisher credibility
                    publisher = review.get("publisher", {})
                    confidence = self._calculate_confidence(publisher, rating)
                    
                    # Extract sources
                    sources = [{
                        "name": publisher.get("name", "Unknown"),
                        "url": review.get("url", ""),
                        "date": review.get("reviewDate", ""),
                        "title": review.get("title", "")
                    }]
                    
                    result = FactCheckResult(
                        claim=claim_text or original_query,
                        verdict=verdict,
                        confidence=confidence,
                        explanation=self._extract_explanation(review),
                        sources=sources,
                        provider="Google Fact Check Tools",
                        timestamp=datetime.now(),
                        rating_details={
                            "original_rating": rating,
                            "publisher": publisher.get("name", "Unknown"),
                            "review_date": review.get("reviewDate", "")
                        }
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error parsing Google fact check result: {e}")
                continue
        
        return results
    
    def _normalize_verdict(self, rating: str) -> str:
        """Normalize various rating formats to standard verdicts"""
        rating = rating.lower().strip()
        
        # False ratings
        if any(term in rating for term in ["false", "incorrect", "fake", "debunked", "pants on fire"]):
            return "False"
        
        # True ratings
        if any(term in rating for term in ["true", "correct", "accurate", "confirmed"]):
            return "True"
        
        # Misleading/Mixed ratings
        if any(term in rating for term in ["misleading", "mixed", "partly", "mostly false", "mostly true"]):
            return "Misleading"
        
        # Unverifiable
        if any(term in rating for term in ["unproven", "unverifiable", "research in progress", "unclear"]):
            return "Unverifiable"
        
        return "Unverifiable"  # Default fallback
    
    def _calculate_confidence(self, publisher: Dict, rating: str) -> float:
        """Calculate confidence score based on publisher and rating clarity"""
        base_confidence = 0.7
        
        # Boost confidence for well-known fact-checkers
        reputable_sources = [
            "snopes", "politifact", "factcheck.org", "reuters", "ap news", 
            "bbc", "washington post", "new york times", "associated press"
        ]
        
        publisher_name = publisher.get("name", "").lower()
        if any(source in publisher_name for source in reputable_sources):
            base_confidence += 0.2
        
        # Adjust based on rating clarity
        rating_lower = rating.lower()
        if any(term in rating_lower for term in ["false", "true", "correct", "incorrect"]):
            base_confidence += 0.1
        elif any(term in rating_lower for term in ["mixed", "partly", "unclear"]):
            base_confidence -= 0.1
        
        return min(0.95, max(0.1, base_confidence))
    
    def _extract_explanation(self, review: Dict) -> str:
        """Extract explanation from review data"""
        title = review.get("title", "")
        if title:
            return f"According to {review.get('publisher', {}).get('name', 'the fact-checker')}: {title}"
        return "Fact-check result from verified source."

class OpenAIFactChecker:
    """OpenAI-powered fact checking"""
    
    def __init__(self):
        self.api_key = get_api_key("openai")
        self.model = "gpt-4o"  # Using 4o for internet access
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
    
    async def analyze_claim(self, claim: str, context: Optional[str] = None, stream: bool = False) -> FactCheckResult:
        """Analyze a claim using OpenAI /v1/responses API with web search"""
        if not is_api_available("openai"):
            logger.warning("🔑 OpenAI API key not available")
            return self._create_fallback_result(claim)
        
        logger.info(f"🤖 CALLING OpenAI /v1/responses API with Web Search")
        logger.info(f"   Claim: '{claim[:100]}{'...' if len(claim) > 100 else ''}'")
        logger.info(f"   Context: {'Yes' if context else 'No'}")
        logger.info(f"   Model: {self.model}")
        
        try:
            session = await self._get_session()
            
            prompt = self._create_web_search_prompt(claim, context)
            
            payload = {
                "model": self.model,
                "input": prompt,
                "instructions": "You are a professional fact-checker with real-time internet access. Use web search to find current, authoritative sources and provide evidence-based fact-checking with specific citations.",
                "tools": [
                    {"type": "web_search"}
                ],
                "max_output_tokens": 1500,
                "temperature": 0.1,
                "store": True,
                "stream": stream
            }
            
            logger.debug(f"   Prompt length: {len(prompt)} characters")
            
            import time
            start_time = time.time()
            
            async with session.post("https://api.openai.com/v1/responses", json=payload) as response:
                api_time = time.time() - start_time
                logger.info(f"📡 OpenAI API Response: {response.status} (took {api_time:.2f}s)")
                
                if response.status == 200:
                    if stream:
                        # Handle streaming response
                        return await self._handle_streaming_response(response, claim, start_time)
                    else:
                        # Handle non-streaming response
                        data = await response.json()
                        
                        # Log token usage for /v1/responses format
                        usage = data.get('usage', {})
                        if usage:
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                            logger.info(f"   Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                        
                        # Log web search activity
                        web_searches = self._count_web_searches(data)
                        if web_searches > 0:
                            logger.info(f"🌐 Performed {web_searches} web searches for real-time information")
                        
                        # Parse /v1/responses format
                        result = self._parse_responses_api_data(data, claim)
                        logger.info(f"✅ OpenAI web search analysis complete: {result.verdict} (confidence: {result.confidence:.2f})")
                        return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ OpenAI API error: {response.status}")
                    logger.error(f"   Error details: {error_text[:200]}...")
                    return self._create_fallback_result(claim)
                    
        except Exception as e:
            logger.error(f"💥 Error calling OpenAI API: {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
            return self._create_fallback_result(claim)
    
    def _create_fact_check_prompt(self, claim: str, context: Optional[str] = None) -> str:
        """Create an advanced fact-checking prompt leveraging GPT-4o's internet access"""
        prompt = f"""You are an expert fact-checker with real-time internet access. Use current, authoritative sources to verify claims with scientific rigor.

ANALYSIS TARGET: "{claim}"
{f"CONTEXT: {context}" if context else ""}

METHODOLOGY:
1. EXTRACT primary factual assertions (ignore corrections, opinions, commentary)
2. SEARCH current authoritative sources for each claim
3. CROSS-REFERENCE multiple reliable sources
4. PROVIDE evidence-based verdict

FACTUAL CLAIM EXTRACTION:
✓ Include: Specific assertions, scientific claims, historical facts, statistics
✗ Exclude: Personal opinions, hearsay, self-corrections, meta-commentary

INTERNET RESEARCH INSTRUCTIONS:
- Search recent scientific studies, government sources, fact-checking sites
- Prioritize: NASA, CDC, WHO, peer-reviewed journals, established news outlets
- Check publication dates - prefer recent sources (2020-2024)
- Look for consensus across multiple authoritative sources

VERDICT LOGIC:
- FALSE: Core factual claims contradict authoritative sources
- MISLEADING: Contains some truth but significant inaccuracies or missing context
- TRUE: Primary assertions confirmed by reliable sources
- UNVERIFIABLE: Insufficient authoritative sources available

CRITICAL: Respond ONLY with valid JSON. No markdown, no explanations, just JSON.

REQUIRED JSON RESPONSE FORMAT:
{{
    "verdict": "True|False|Misleading|Unverifiable",
    "confidence": 0.85,
    "explanation": "Evidence-based analysis with specific source references",
    "primary_claims_extracted": ["Exact claim 1", "Exact claim 2"],
    "evidence_assessment": {{
        "claim_1": {{"status": "True/False", "evidence": "Source-specific evidence", "sources": ["URL or publication"]}},
        "claim_2": {{"status": "True/False", "evidence": "Source-specific evidence", "sources": ["URL or publication"]}}
    }},
    "web_sources_consulted": ["Source 1", "Source 2", "Source 3"],
    "reasoning": "Logic connecting evidence to verdict",
    "fact_check_date": "{datetime.now().strftime('%Y-%m-%d')}"
}}

EXAMPLES:

CLAIM: "The Great Wall of China is visible from space. NASA debunked this."
→ EXTRACT: "The Great Wall of China is visible from space"
→ RESEARCH: Check NASA.gov, astronaut testimonies, space agency statements
→ VERDICT: FALSE (NASA confirms not visible to naked eye from space)

CLAIM: "COVID-19 vaccines cause myocarditis in all recipients."
→ EXTRACT: "COVID-19 vaccines cause myocarditis in all recipients"
→ RESEARCH: CDC data, medical journals, clinical trial results
→ VERDICT: FALSE (Myocarditis is rare, not universal; benefits outweigh risks)

CLAIM: "Water boils at 100 degrees Celsius at sea level."
→ EXTRACT: "Water boils at 100 degrees Celsius at sea level"
→ RESEARCH: Check physics textbooks, scientific sources, NIST standards
→ VERDICT: TRUE (Confirmed by scientific consensus and physical laws)

CLAIM: "Climate change is primarily caused by human activities according to scientific consensus."
→ EXTRACT: "Climate change is primarily caused by human activities according to scientific consensus"
→ RESEARCH: Check IPCC reports, NASA climate data, peer-reviewed studies
→ VERDICT: TRUE (Overwhelming scientific evidence supports human causation)

CLAIM: "Vaccines are effective at preventing diseases they target."
→ EXTRACT: "Vaccines are effective at preventing diseases they target"
→ RESEARCH: Check CDC vaccine effectiveness data, WHO reports, medical studies
→ VERDICT: TRUE (Extensive clinical evidence demonstrates vaccine effectiveness)

Use your internet access to find the most current and authoritative information available."""
        return prompt
    
    def _create_web_search_prompt(self, claim: str, context: Optional[str] = None) -> str:
        """Create an optimized prompt for the /v1/responses API with web search"""
        from datetime import datetime
        
        prompt = f"""Fact-check this claim using real-time web search: "{claim}"
{f"Context: {context}" if context else ""}

SEARCH STRATEGY:
1. Search for recent news and information about this claim
2. Check authoritative sources: Reuters, AP News, BBC, government agencies
3. Look for fact-checking websites: Snopes, PolitiFact, FactCheck.org
4. Find scientific studies and peer-reviewed research if applicable
5. Cross-reference multiple reliable sources

SEARCH QUERIES TO USE:
- "{claim}" fact check
- "{claim}" Reuters OR "AP News" OR BBC
- "{claim}" scientific study research
- "{claim}" government official statement

REQUIRED OUTPUT FORMAT (JSON):
{{
    "verdict": "True|False|Misleading|Unverifiable",
    "confidence": 0.85,
    "explanation": "Detailed analysis based on web search results",
    "primary_claims": ["main factual assertion 1", "main factual assertion 2"],
    "evidence_found": [
        {{
            "source_name": "Source Publication Name",
            "source_url": "https://complete-url-to-specific-article.com/article-title",
            "article_title": "Exact title of the article or page",
            "date": "YYYY-MM-DD",
            "finding": "What this source says about the claim",
            "credibility": "High|Medium|Low",
            "quote": "Relevant quote from the article if available"
        }}
    ],
    "web_searches_performed": ["search query 1", "search query 2"],
    "reasoning": "Step-by-step logic connecting evidence to verdict",
    "fact_check_date": "{datetime.now().strftime('%Y-%m-%d')}"
}}

CRITICAL REQUIREMENTS:
- Use web search to find CURRENT information (2023-2025)
- MUST provide complete URLs (https://...) for every source
- Include exact article titles and publication dates
- Provide specific quotes or excerpts from articles when available
- Look for consensus across multiple authoritative sources
- Be transparent about what you found or couldn't find
- Focus on factual claims, ignore opinions or speculation

URL EXAMPLES:
- Good: "https://www.reuters.com/technology/musk-completes-twitter-takeover-2022-10-28/"
- Good: "https://apnews.com/article/elon-musk-twitter-acquisition-123abc"
- Bad: "Reuters" or "AP News" (no specific article URL)"""
        return prompt
    
    def _count_web_searches(self, data: Dict) -> int:
        """Count web search tool calls in the response"""
        try:
            web_search_count = 0
            output = data.get('output', [])
            for item in output:
                if item.get('type') == 'tool_call':
                    function = item.get('function', {})
                    if function.get('name') == 'web_search':
                        web_search_count += 1
            return web_search_count
        except Exception:
            return 0
    
    def _parse_responses_api_data(self, data: Dict, claim: str) -> FactCheckResult:
        """Parse /v1/responses API format into FactCheckResult"""
        try:
            # Extract the main text content from output array
            output = data.get('output', [])
            content = None
            
            for item in output:
                if item.get('type') == 'message' and item.get('role') == 'assistant':
                    content_parts = item.get('content', [])
                    for part in content_parts:
                        if part.get('type') == 'output_text':
                            content = part.get('text', '')
                            break
                    if content:
                        break
            
            if not content:
                logger.error(f"No assistant message content found in /v1/responses output")
                return self._create_fallback_result(claim)
            
            logger.debug(f"OpenAI /v1/responses content: {content[:500]}...")
            
            # Try to parse JSON response
            try:
                # Clean content - sometimes GPT adds markdown formatting
                clean_content = content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()
                
                import json
                parsed_data = json.loads(clean_content)
                
                # Extract data with fallbacks
                verdict = parsed_data.get("verdict", "Unverifiable")
                confidence = float(parsed_data.get("confidence", 0.5))
                explanation = parsed_data.get("explanation", "Analysis completed with web search")
                
                # Extract evidence and sources with URLs
                evidence_found = parsed_data.get("evidence_found", [])
                web_sources = []
                for evidence in evidence_found:
                    source_name = evidence.get("source_name", evidence.get("source", "Unknown Source"))
                    source_url = evidence.get("source_url", "")
                    article_title = evidence.get("article_title", "")
                    
                    # Create source object with proper URL
                    if source_url and source_url.startswith("http"):
                        source_obj = {
                            "name": source_name,
                            "url": source_url,
                            "type": "web_search",
                            "title": article_title
                        }
                    else:
                        # If no URL provided, use source name as fallback
                        source_obj = {
                            "name": source_name,
                            "url": f"Web search: {source_name}",
                            "type": "web_search",
                            "title": article_title
                        }
                    web_sources.append(source_obj)
                
                # Fallback to other source fields if no evidence_found
                if not web_sources:
                    fallback_sources = parsed_data.get("web_sources_consulted", [])
                    for source in fallback_sources:
                        web_sources.append({
                            "name": source,
                            "url": f"Web search: {source}",
                            "type": "web_search",
                            "title": ""
                        })
                    
                    if not web_sources:
                        web_sources = [{
                            "name": "Web Search Analysis",
                            "url": "Multiple sources consulted",
                            "type": "web_search",
                            "title": ""
                        }]
                
                # Extract web searches performed
                searches_performed = parsed_data.get("web_searches_performed", [])
                if searches_performed:
                    logger.info(f"   Search queries used: {', '.join(searches_performed[:3])}...")
                
                logger.info(f"   Sources found: {len(web_sources)} web sources")
                logger.info(f"   Verdict: {verdict} (confidence: {confidence:.2f})")
                
                return FactCheckResult(
                    claim=claim,
                    verdict=verdict,
                    confidence=confidence,
                    explanation=explanation,
                    sources=web_sources[:5],  # Limit to top 5 sources, already formatted with URLs
                    provider="OpenAI Web Search",
                    timestamp=datetime.now()
                )
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from OpenAI response: {e}")
                logger.debug(f"Raw content: {content[:200]}...")
                
                # Fallback: try to extract basic info from text
                return self._extract_from_text_response(content, claim)
                
        except Exception as e:
            logger.error(f"Error parsing /v1/responses data: {e}")
            return self._create_fallback_result(claim)
    
    def _extract_from_text_response(self, content: str, claim: str) -> FactCheckResult:
        """Extract fact-check info from non-JSON text response"""
        try:
            content_lower = content.lower()
            
            # Determine verdict
            if "false" in content_lower and "misleading" not in content_lower:
                verdict = "False"
                confidence = 0.8
            elif "true" in content_lower and "misleading" not in content_lower:
                verdict = "True"
                confidence = 0.8
            elif "misleading" in content_lower:
                verdict = "Misleading" 
                confidence = 0.7
            else:
                verdict = "Unverifiable"
                confidence = 0.5
            
            # Extract URLs if present
            import re
            urls = re.findall(r'https?://[^\s]+', content)
            
            sources = []
            if urls:
                for i, url in enumerate(urls[:3]):
                    sources.append({
                        "name": f"Source {i+1}",
                        "url": url.rstrip('.,;'),  # Remove trailing punctuation
                        "type": "web_search",
                        "title": ""
                    })
            else:
                sources = [{
                    "name": "Web Search Analysis",
                    "url": "Multiple sources consulted",
                    "type": "web_search",
                    "title": ""
                }]
            
            return FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                explanation=content[:500] + "..." if len(content) > 500 else content,
                sources=sources,
                provider="OpenAI Web Search (Text)",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error extracting from text response: {e}")
            return self._create_fallback_result(claim)
    
    async def _handle_streaming_response(self, response, claim: str, start_time: float) -> FactCheckResult:
        """Handle streaming Server-Sent Events from /v1/responses API"""
        try:
            import json
            import time
            
            logger.info("🔄 Processing streaming response with real-time updates...")
            
            full_content = ""
            web_searches_started = 0
            web_searches_completed = 0
            
            async for line in response.content:
                if not line:
                    continue
                    
                line_str = line.decode('utf-8').strip()
                if not line_str.startswith('data: '):
                    continue
                
                data_str = line_str[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break
                
                try:
                    event_data = json.loads(data_str)
                    event_type = event_data.get('type', '')
                    
                    # Log web search events
                    if event_type == 'response.web_search_call.in_progress':
                        web_searches_started += 1
                        logger.info(f"🔍 Web search {web_searches_started} started...")
                    elif event_type == 'response.web_search_call.completed':
                        web_searches_completed += 1
                        logger.info(f"✅ Web search {web_searches_completed} completed")
                    
                    # Collect text content
                    elif event_type == 'response.output_text.delta':
                        delta = event_data.get('delta', '')
                        if delta:
                            full_content += delta
                    
                    # Log when response is complete
                    elif event_type == 'response.completed':
                        total_time = time.time() - start_time
                        logger.info(f"🎉 Streaming response completed in {total_time:.2f}s")
                        usage = event_data.get('response', {}).get('usage', {})
                        if usage:
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                            logger.info(f"   Tokens - Input: {input_tokens}, Output: {output_tokens}")
                        
                except json.JSONDecodeError:
                    continue  # Skip malformed JSON
            
            logger.info(f"🌐 Completed {web_searches_completed} web searches")
            logger.debug(f"Full streaming content: {full_content[:500]}...")
            
            # Parse the final content
            if not full_content:
                logger.error("No content received from streaming response")
                return self._create_fallback_result(claim)
            
            # Create a mock response data structure for parsing
            mock_data = {
                'output': [{
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{
                        'type': 'output_text',
                        'text': full_content
                    }]
                }]
            }
            
            result = self._parse_responses_api_data(mock_data, claim)
            logger.info(f"✅ Streaming analysis complete: {result.verdict} (confidence: {result.confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error handling streaming response: {e}")
            return self._create_fallback_result(claim)
    
    def _parse_openai_response(self, data: Dict, claim: str) -> FactCheckResult:
        """Parse OpenAI response into FactCheckResult with improved error handling"""
        try:
            # Check if choices exist and has content
            if not data.get("choices") or len(data["choices"]) == 0:
                logger.error(f"No choices in OpenAI response: {data}")
                return self._create_fallback_result(claim)
            
            message = data["choices"][0].get("message", {})
            content = message.get("content")
            
            # Tool calls are now handled in the main analyze_claim method
            
            if not content:
                logger.error(f"No content in OpenAI message: {message}")
                return self._create_fallback_result(claim)
            
            logger.debug(f"OpenAI response content: {content[:500]}...")
            
            # Try to parse JSON response
            try:
                # Clean content - sometimes GPT adds markdown formatting
                clean_content = content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content.replace("```json", "").replace("```", "").strip()
                elif clean_content.startswith("```"):
                    clean_content = clean_content.replace("```", "").strip()
                
                result_data = json.loads(clean_content)
                logger.info(f"✅ Successfully parsed JSON response")
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode failed: {e}")
                logger.debug(f"Raw content that failed: {content}")
                # Fallback: extract information from text
                return self._parse_text_response(content, claim)
            
            # Extract sources from web research
            sources = []
            if "web_sources_consulted" in result_data and isinstance(result_data["web_sources_consulted"], list):
                for source in result_data["web_sources_consulted"]:
                    if isinstance(source, str):
                        sources.append({
                            "name": source,
                            "type": "Web Source",
                            "url": source if source.startswith("http") else None
                        })
            
            # Add standard AI analysis source
            sources.append({
                "name": f"OpenAI {self.model} Analysis",
                "type": "AI Analysis with Internet Access",
                "reasoning": result_data.get("reasoning", ""),
                "evidence": result_data.get("evidence_assessment", {})
            })
            
            return FactCheckResult(
                claim=claim,
                verdict=result_data.get("verdict", "Unverifiable"),
                confidence=float(result_data.get("confidence", 0.5)),
                explanation=result_data.get("explanation", "AI analysis completed"),
                sources=sources,
                provider=f"OpenAI {self.model}",
                timestamp=datetime.now(),
                rating_details=result_data
            )
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            logger.error(f"Full response data: {data}")
            return self._create_fallback_result(claim)
    
    def _parse_text_response(self, content: str, claim: str) -> FactCheckResult:
        """Parse text response when JSON parsing fails"""
        # Simple text parsing fallback
        verdict = "Unverifiable"
        confidence = 0.5
        
        content_lower = content.lower()
        if "false" in content_lower or "incorrect" in content_lower:
            verdict = "False"
            confidence = 0.7
        elif "true" in content_lower or "correct" in content_lower:
            verdict = "True"
            confidence = 0.7
        elif "misleading" in content_lower or "mixed" in content_lower:
            verdict = "Misleading"
            confidence = 0.6
        
        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=content[:500],  # Truncate if too long
            sources=[{"name": "OpenAI GPT-4", "content": content}],
            provider="OpenAI GPT-4",
            timestamp=datetime.now()
        )
    
    def _create_fallback_result(self, claim: str) -> FactCheckResult:
        """Create a fallback result when API is unavailable"""
        return FactCheckResult(
            claim=claim,
            verdict="Unverifiable",
            confidence=0.1,
            explanation="AI fact-checking service temporarily unavailable",
            sources=[],
            provider="Fallback",
            timestamp=datetime.now()
        )
    
    async def _handle_tool_calls(self, tool_calls, claim, original_prompt, session):
        """Handle OpenAI tool calls and get final response"""
        logger.info(f"🔧 Processing {len(tool_calls)} tool calls...")
        
        # Simulate tool responses (since we can't actually execute web_search)
        # In a real implementation, you'd execute the actual tools
        tool_responses = []
        for i, tool_call in enumerate(tool_calls):
            tool_response = {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": f"Web search {i+1}: Found relevant information about the claim from authoritative sources."
            }
            tool_responses.append(tool_response)
        
        # Continue the conversation with tool results
        messages = [
            {"role": "system", "content": "You are a professional fact-checker. Analyze the tool results and provide your final assessment in JSON format."},
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
        ] + tool_responses + [
            {"role": "user", "content": "Based on the web search results, provide your final fact-check analysis in the required JSON format."}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
            if response.status == 200:
                data = await response.json()
                logger.info("✅ Got final response after tool calls")
                return self._parse_openai_response(data, claim)
            else:
                raise Exception(f"API error: {response.status}")
    
    async def _analyze_without_tools(self, claim, session, original_prompt):
        """Fallback analysis without web search tools"""
        logger.info("🔄 Falling back to analysis without web search...")
        
        # Simplified prompt without tool requirements
        simple_prompt = f"""You are a professional fact-checker. Analyze this claim and respond in JSON format.

Claim: "{claim}"

Respond with JSON only:
{{
    "verdict": "True|False|Misleading|Unverifiable",
    "confidence": 0.85,
    "explanation": "Analysis based on known facts",
    "reasoning": "Logical basis for verdict"
}}"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional fact-checker. Respond only in valid JSON format."},
                {"role": "user", "content": simple_prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.1
        }
        
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
            if response.status == 200:
                data = await response.json()
                logger.info("✅ Fallback analysis completed")
                return self._parse_openai_response(data, claim)
            else:
                return self._create_fallback_result(claim)

class GrokFactChecker:
    """Grok-powered fact-checking using X.AI API"""
    
    def __init__(self):
        self.api_key = get_api_key("grok")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4"
        self.session = None
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def analyze_claim(self, claim: str, context: Optional[str] = None) -> FactCheckResult:
        """Analyze a claim using Grok's real-time knowledge and social context"""
        if not self.api_key:
            raise ValueError("Grok API key not available")
        
        if not self.session:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            logger.info(f"🌐 Grok analyzing claim: '{claim[:60]}...'")
            
            prompt = f"""You are a professional fact-checker with access to real-time social media data and current information. Analyze this claim:

CLAIM: "{claim}"
{f"CONTEXT: {context}" if context else ""}

Provide a comprehensive fact-check response in JSON format. Use your access to real-time X/Twitter data and current information to assess:

1. Current social media discussions about this topic
2. Recent developments or news related to this claim  
3. Expert opinions and official statements
4. Evidence from credible sources
5. Social consensus and misinformation patterns

Respond with valid JSON only:
{{
    "verdict": "True|False|Misleading|Unverifiable",
    "confidence": 0.85,
    "explanation": "Detailed analysis including social context and real-time information",
    "social_context": "Current social media discussion and expert reactions",
    "evidence_assessment": "Key evidence and source reliability",
    "misinformation_patterns": "Any related misinformation being spread"
}}"""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a professional fact-checker with real-time social media access. Respond only in valid JSON format."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return self._parse_grok_response(data, claim)
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Grok API error {response.status}: {error_text}")
                    return self._create_fallback_result(claim)
                    
        except Exception as e:
            logger.error(f"❌ Grok fact-check failed: {e}")
            return self._create_fallback_result(claim)
    
    def _parse_grok_response(self, data: Dict, claim: str) -> FactCheckResult:
        """Parse Grok API response into FactCheckResult"""
        try:
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            
            # Clean up the response - remove any markdown formatting
            content = content.replace('```json', '').replace('```', '').strip()
            
            parsed = json.loads(content)
            
            # Extract sources from social context
            sources = [{
                "name": "Grok Real-time Analysis",
                "type": "AI Analysis with Social Media Access",
                "url": "https://x.ai",
                "social_context": parsed.get("social_context", ""),
                "evidence": parsed.get("evidence_assessment", "")
            }]
            
            return FactCheckResult(
                claim=claim,
                verdict=parsed.get("verdict", "Unverifiable"),
                confidence=float(parsed.get("confidence", 0.5)),
                explanation=parsed.get("explanation", "Analysis completed using real-time social data"),
                sources=sources,
                provider="Grok (X.AI)",
                timestamp=datetime.now(),
                rating_details={
                    "social_context": parsed.get("social_context", ""),
                    "evidence_assessment": parsed.get("evidence_assessment", ""),
                    "misinformation_patterns": parsed.get("misinformation_patterns", "")
                }
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Grok response: {e}")
            return self._create_fallback_result(claim)
    
    def _create_fallback_result(self, claim: str) -> FactCheckResult:
        """Create fallback result when Grok analysis fails"""
        return FactCheckResult(
            claim=claim,
            verdict="Unverifiable",
            confidence=0.3,
            explanation="Grok analysis unavailable - unable to access real-time social context",
            sources=[{
                "name": "Grok Analysis",
                "type": "Social Media Context",
                "url": "",
                "error": "Analysis failed"
            }],
            provider="Grok (X.AI) - Error",
            timestamp=datetime.now()
        )

class MultiSourceFactChecker:
    """Orchestrates multiple fact-checking sources"""
    
    def __init__(self):
        self.google_checker = GoogleFactCheckAPI()
        self.openai_checker = OpenAIFactChecker()
        self.grok_checker = GrokFactChecker() if is_api_available("grok") else None
    
    async def close(self):
        """Close all API sessions"""
        try:
            await self.google_checker.close()
        except Exception as e:
            logger.warning(f"Error closing Google checker: {e}")
        
        try:
            await self.openai_checker.close()
        except Exception as e:
            logger.warning(f"Error closing OpenAI checker: {e}")
        
        if self.grok_checker:
            try:
                await self.grok_checker.close()
            except Exception as e:
                logger.warning(f"Error closing Grok checker: {e}")
    
    async def comprehensive_fact_check(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive fact-checking using multiple sources"""
        logger.info(f"🔍 STARTING Multi-Source Fact-Check")
        logger.info(f"   Claim: '{claim[:100]}{'...' if len(claim) > 100 else ''}'")
        
        # Check which APIs are available
        available_apis = []
        if is_api_available("google_fact_check"):
            available_apis.append("Google Fact Check")
        if is_api_available("openai"):
            available_apis.append("OpenAI")
        if is_api_available("anthropic"):
            available_apis.append("Anthropic")
        if is_api_available("grok"):
            available_apis.append("Grok")
        
        logger.info(f"   Available APIs: {', '.join(available_apis) if available_apis else 'None'}")
        
        results = []
        
        # Run fact-checks in parallel
        tasks = []
        task_names = []
        
        # Google Fact Check Tools
        if is_api_available("google_fact_check"):
            tasks.append(self.google_checker.search_claims(claim))
            task_names.append("Google")
        
        # OpenAI Analysis
        if is_api_available("openai"):
            tasks.append(self._wrap_openai_result(self.openai_checker.analyze_claim(claim, context)))
            task_names.append("OpenAI")
        
        # Grok Analysis (Real-time Social Context)
        if is_api_available("grok") and self.grok_checker:
            tasks.append(self._wrap_grok_result(self.grok_checker.analyze_claim(claim, context)))
            task_names.append("Grok")
        
        if tasks:
            logger.info(f"🚀 Running {len(tasks)} parallel API calls: {', '.join(task_names)}")
            
            try:
                import time
                parallel_start = time.time()
                
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                parallel_time = time.time() - parallel_start
                logger.info(f"⚡ Parallel API calls completed in {parallel_time:.2f}s")
                
                # Process results
                for i, result in enumerate(task_results):
                    api_name = task_names[i]
                    
                    if isinstance(result, list):
                        logger.info(f"   {api_name}: {len(result)} results")
                        results.extend(result)
                    elif isinstance(result, FactCheckResult):
                        logger.info(f"   {api_name}: 1 result ({result.verdict})")
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"   {api_name}: Failed - {result}")
                        
            except Exception as e:
                logger.error(f"💥 Error in parallel fact-checking: {e}")
        else:
            logger.warning("⚠️ No APIs available for fact-checking")
        
        # Synthesize results
        logger.info(f"🧠 Synthesizing {len(results)} total results...")
        final_result = await self._synthesize_results(claim, results)
        
        logger.info(f"✅ Multi-source analysis complete:")
        logger.info(f"   Final Verdict: {final_result.get('verdict', 'Unknown')}")
        logger.info(f"   Final Confidence: {final_result.get('confidence', 0):.2f}")
        logger.info(f"   Sources Used: {final_result.get('details', {}).get('source_count', 0)}")
        
        return final_result
    
    async def _wrap_openai_result(self, openai_task) -> FactCheckResult:
        """Wrapper to handle OpenAI task in gather"""
        return await openai_task
    
    async def _wrap_grok_result(self, grok_task) -> FactCheckResult:
        """Wrapper to handle Grok task in gather"""
        return await grok_task
    
    async def _synthesize_results(self, claim: str, results: List[FactCheckResult]) -> Dict[str, Any]:
        """Synthesize multiple fact-check results into a final assessment with improved weighting"""
        if not results:
            return {
                "verdict": "Unverifiable",
                "confidence": 0.1,
                "explanation": "No fact-checking sources were available to verify this claim.",
                "sources": [],
                "provider": "Multi-Source (No Results)",
                "details": {
                    "source_count": 0,
                    "agreement_level": 0.0,
                    "confidence_range": [0.0, 0.0]
                }
            }
        
        logger.info(f"📊 Synthesizing {len(results)} results with improved logic...")
        
        # Separate results by type for weighted analysis
        openai_results = []
        google_results = []
        grok_results = []
        
        for result in results:
            if "OpenAI" in result.provider:
                openai_results.append(result)
            elif "Google" in result.provider:
                google_results.append(result)
            elif "Grok" in result.provider:
                grok_results.append(result)
        
        logger.info(f"   OpenAI Results: {len(openai_results)}, Google Results: {len(google_results)}, Grok Results: {len(grok_results)}")
        
        # Enhanced Google result interpretation using LLM
        processed_google = await self._process_google_results_with_llm(google_results, claim)
        logger.info(f"   Processed Google results: {processed_google}")
        
        # Calculate weighted verdict using source authority
        weighted_scores = {
            "True": 0.0,
            "False": 0.0, 
            "Misleading": 0.0,
            "Unverifiable": 0.0
        }
        
        total_weight = 0.0
        confidence_sum = 0.0
        confidence_weights = 0.0
        
        # Weight OpenAI analysis heavily (it has internet access and context understanding)
        openai_weight = 3.0
        for result in openai_results:
            weight = openai_weight * result.confidence
            weighted_scores[result.verdict] += weight
            total_weight += weight
            confidence_sum += result.confidence * weight
            confidence_weights += weight
            logger.info(f"   OpenAI: {result.verdict} (weight: {weight:.2f}, conf: {result.confidence:.2f})")
        
        # Weight Grok analysis moderately (real-time social context, but can be influenced by social media bias)
        grok_weight = 2.0
        for result in grok_results:
            weight = grok_weight * result.confidence
            weighted_scores[result.verdict] += weight
            total_weight += weight
            confidence_sum += result.confidence * weight
            confidence_weights += weight
            logger.info(f"   Grok: {result.verdict} (weight: {weight:.2f}, conf: {result.confidence:.2f})")
        
        # Weight processed Google results
        google_weight = 1.0
        google_verdict = processed_google.get("consensus_verdict", "Unverifiable")
        google_confidence = processed_google.get("confidence", 0.5)
        google_total_weight = google_weight * google_confidence * processed_google.get("result_count", 1)
        
        if google_results:  # Only if we have Google results
            weighted_scores[google_verdict] += google_total_weight
            total_weight += google_total_weight
            confidence_sum += google_confidence * google_total_weight
            confidence_weights += google_total_weight
            logger.info(f"   Google: {google_verdict} (weight: {google_total_weight:.2f}, conf: {google_confidence:.2f})")
        
        # Determine final verdict
        if total_weight > 0:
            primary_verdict = max(weighted_scores, key=weighted_scores.get)
            final_confidence = confidence_sum / confidence_weights if confidence_weights > 0 else 0.5
        else:
            primary_verdict = "Unverifiable"
            final_confidence = 0.1
        
        # Add scientific consensus boost for well-established facts
        scientific_consensus_topics = [
            ("vaccine", ["safe", "effective", "work"], "True"),
            ("climate change", ["real", "human", "caused", "anthropogenic"], "True"), 
            ("evolution", ["theory", "true", "real"], "True"),
            ("earth", ["round", "spherical", "oblate"], "True"),
            ("water", ["boils", "100", "celsius"], "True"),
            ("gravity", ["exists", "real", "pulls"], "True")
        ]
        
        claim_lower = claim.lower()
        for topic, keywords, consensus_verdict in scientific_consensus_topics:
            if topic in claim_lower and any(kw in claim_lower for kw in keywords):
                if primary_verdict != consensus_verdict and final_confidence < 0.9:
                    # Check if any AI and Google actually support the scientific consensus
                    openai_supports_consensus = any(r.verdict == consensus_verdict for r in openai_results)
                    grok_supports_consensus = any(r.verdict == consensus_verdict for r in grok_results)
                    google_supports_consensus = processed_google.get("consensus_verdict") == consensus_verdict
                    
                    if openai_supports_consensus or grok_supports_consensus or google_supports_consensus:
                        logger.info(f"🧬 Scientific consensus boost applied for {topic} → {consensus_verdict}")
                        logger.info(f"   Support - OpenAI: {openai_supports_consensus}, Grok: {grok_supports_consensus}, Google: {google_supports_consensus}")
                        primary_verdict = consensus_verdict
                        final_confidence = min(0.95, final_confidence + 0.1)
                        break
        
        # Combine explanations with priority to OpenAI, then Grok, then Google
        explanations = []
        
        # Prioritize OpenAI explanations (highest weight)
        for result in openai_results:
            if result.explanation:
                explanations.append(f"[{result.provider}] {result.explanation}")
        
        # Add Grok explanations (real-time social context)
        for result in grok_results:
            if result.explanation:
                explanations.append(f"[{result.provider}] {result.explanation}")
        
        # Add Google summary if meaningful
        if processed_google.get("summary"):
            explanations.append(f"[Google Fact Check Tools] {processed_google['summary']}")
        
        # Fallback to individual Google results if no summary
        if not processed_google.get("summary") and google_results:
            for result in google_results[:2]:  # Limit to top 2
                if result.explanation:
                    explanations.append(f"[{result.provider}] {result.explanation}")
        
        combined_explanation = " | ".join(explanations[:3])  # Limit to top 3
        
        # Collect all sources
        all_sources = []
        for result in results:
            all_sources.extend(result.sources)
        
        logger.info(f"✅ Final synthesis: {primary_verdict} (confidence: {final_confidence:.2f})")
        
        return {
            "verdict": primary_verdict,
            "confidence": min(0.95, max(0.1, final_confidence)),
            "explanation": combined_explanation,
            "sources": all_sources,
            "provider": "Multi-Source Analysis",
            "details": {
                "source_count": len(results),
                "openai_results": len(openai_results),
                "grok_results": len(grok_results),
                "google_results": len(google_results),
                "weighted_scores": weighted_scores,
                "total_weight": total_weight,
                "google_processing": processed_google,
                "primary_claims_extracted": self._extract_primary_claims(results),
                "claim_evaluations": self._extract_claim_evaluations(results),
                "web_sources_consulted": self._extract_web_sources(all_sources),
                "reasoning": f"Weighted analysis: OpenAI results ({len(openai_results)}) + Grok results ({len(grok_results)}) + Google results ({len(google_results)}) = {primary_verdict}"
            }
        }
    
    async def _process_google_results_with_llm(self, google_results: List[FactCheckResult], claim: str) -> Dict[str, Any]:
        """Use LLM to interpret Google fact-check results contextually"""
        if not google_results:
            return {"consensus_verdict": "Unverifiable", "confidence": 0.1, "result_count": 0}
        
        if not is_api_available("openai"):
            # Fallback to simple aggregation
            return self._simple_google_aggregation(google_results)
        
        # Prepare Google results for LLM analysis
        fact_check_summaries = []
        for i, result in enumerate(google_results[:10]):  # Limit to top 10 to avoid token limits
            summary = {
                "source": result.sources[0].get("name", "Unknown") if result.sources else "Unknown",
                "verdict": result.verdict,
                "explanation": result.explanation[:200],  # Truncate to save tokens
                "confidence": result.confidence
            }
            fact_check_summaries.append(f"{i+1}. {summary['source']}: '{summary['explanation']}' (Verdict: {summary['verdict']})")
        
        prompt = f"""You are analyzing fact-check results from Google Fact Check Tools API. Your task is to interpret these results in context to determine what they actually say about the original claim.

ORIGINAL CLAIM: "{claim}"

FACT-CHECK RESULTS FROM GOOGLE:
{chr(10).join(fact_check_summaries)}

CRITICAL INTERPRETATION RULES:
Google Fact Check often contains articles debunking MISINFORMATION about established scientific topics. You must distinguish between:

A) DEBUNKING THE CLAIM ITSELF (claim is false)
B) DEBUNKING MISINFORMATION ABOUT THE CLAIM (claim is actually true)

DETAILED EXAMPLES:

CLAIM: "Vaccines are safe and effective"
- Google Result: "Fact-check debunks RFK Jr. vaccine claims" → SUPPORTS vaccine safety (debunking misinformation)
- Google Result: "Study finds vaccines cause autism" → REFUTES vaccine safety (debunking the claim itself)

CLAIM: "Climate change is caused by human activities"  
- Google Result: "Fact-check: Climate denial study misleading" → SUPPORTS climate science (debunking denial)
- Google Result: "IPCC report confirms human causation" → SUPPORTS climate science (direct confirmation)
- Google Result: "Study proves climate change is natural" → REFUTES climate science (debunking the claim)

CLAIM: "The Earth is round"
- Google Result: "Flat Earth theory debunked by NASA" → SUPPORTS round Earth (debunking misinformation)
- Google Result: "NASA admits Earth is flat" → REFUTES round Earth (if true, which it's not)

INTERPRETATION LOGIC:
1. Look for keywords: "debunks", "misleading", "false claims about", "misinformation about"
2. Identify the TARGET of the debunking - is it debunking the claim or debunking misinformation about the claim?
3. For scientific consensus topics (vaccines, climate, basic science), assume debunking is about misinformation UNLESS explicitly stated otherwise

SCIENTIFIC CONSENSUS TOPICS (default to True unless clearly refuted):
- Vaccine safety and effectiveness
- Climate change and human causation
- Basic physics (water boiling, gravity, etc.)
- Earth's shape and astronomy
- Evolution and biology basics

Your job: Determine if these Google fact-checks are:
1. SUPPORTING THE CLAIM (by debunking misinformation about it)
2. REFUTING THE CLAIM (by providing evidence against it)
3. MIXED/UNCLEAR

Respond with ONLY this JSON:
{{
    "interpretation": "supporting_claim|refuting_claim|mixed|unclear",
    "consensus_verdict": "True|False|Misleading|Unverifiable", 
    "confidence": 0.75,
    "reasoning": "Detailed explanation focusing on whether Google results debunk the claim itself or misinformation about the claim",
    "evidence_summary": "What the fact-checks actually indicate about the original claim with specific focus on scientific consensus"
}}"""

        try:
            headers = {
                "Authorization": f"Bearer {get_api_key('openai')}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are an expert at interpreting fact-check context. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1
                }
                
                logger.info(f"🧠 Analyzing Google results with LLM...")
                
                async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        
                        try:
                            # Clean and parse JSON
                            clean_content = content.strip()
                            if clean_content.startswith("```json"):
                                clean_content = clean_content.replace("```json", "").replace("```", "").strip()
                            
                            result = json.loads(clean_content)
                            
                            logger.info(f"✅ LLM interpretation: {result.get('interpretation')} → {result.get('consensus_verdict')}")
                            logger.info(f"   Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                            
                            return {
                                "consensus_verdict": result.get("consensus_verdict", "Unverifiable"),
                                "confidence": float(result.get("confidence", 0.5)),
                                "result_count": len(google_results),
                                "interpretation": result.get("interpretation", "unclear"),
                                "summary": result.get("evidence_summary", "LLM analysis of fact-check context"),
                                "reasoning": result.get("reasoning", "LLM-based interpretation")
                            }
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse LLM response for Google analysis: {e}")
                            logger.debug(f"Raw response: {content}")
                            return self._simple_google_aggregation(google_results)
                    else:
                        logger.error(f"LLM analysis failed: {response.status}")
                        return self._simple_google_aggregation(google_results)
                    
        except Exception as e:
            logger.error(f"Error in LLM Google analysis: {e}")
            return self._simple_google_aggregation(google_results)
    
    def _simple_google_aggregation(self, google_results: List[FactCheckResult]) -> Dict[str, Any]:
        """Simple fallback aggregation of Google results"""
        if not google_results:
            return {"consensus_verdict": "Unverifiable", "confidence": 0.1, "result_count": 0}
        
        # Simple majority vote
        verdicts = [r.verdict for r in google_results]
        verdict_counts = {}
        for verdict in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        consensus_verdict = max(verdict_counts, key=verdict_counts.get)
        avg_confidence = sum(r.confidence for r in google_results) / len(google_results)
        
        return {
            "consensus_verdict": consensus_verdict,
            "confidence": avg_confidence,
            "result_count": len(google_results),
            "summary": f"Simple aggregation: {consensus_verdict} from {len(google_results)} results"
        }
    
    def _extract_primary_claims(self, results: List[FactCheckResult]) -> List[str]:
        """Extract primary claims from all results"""
        claims = []
        for result in results:
            if hasattr(result, 'rating_details') and result.rating_details:
                if isinstance(result.rating_details, dict):
                    extracted_claims = result.rating_details.get('primary_claims_extracted', [])
                    if extracted_claims:
                        claims.extend(extracted_claims)
            if result.claim and result.claim not in claims:
                claims.append(result.claim)
        return claims[:3]  # Limit to top 3
    
    def _extract_claim_evaluations(self, results: List[FactCheckResult]) -> Dict[str, str]:
        """Extract claim evaluations from results"""
        evaluations = {}
        for result in results:
            if hasattr(result, 'rating_details') and result.rating_details:
                if isinstance(result.rating_details, dict):
                    evidence_assessment = result.rating_details.get('evidence_assessment', {})
                    if evidence_assessment:
                        # Handle both dict and string evidence assessments
                        if isinstance(evidence_assessment, dict):
                            evaluations.update(evidence_assessment)
                        elif isinstance(evidence_assessment, str):
                            # If it's a string (like from Grok), use claim as key
                            evaluations[result.claim] = evidence_assessment
                        else:
                            logger.debug(f"Unexpected evidence_assessment type: {type(evidence_assessment)}")
            # Fallback to basic claim -> verdict mapping
            if result.claim:
                evaluations[result.claim] = result.verdict
        return evaluations
    
    def _extract_web_sources(self, all_sources: List[Dict]) -> List[str]:
        """Extract web source names/URLs from sources"""
        web_sources = []
        for source in all_sources:
            if source.get("type") == "Web Source":
                name = source.get("name", source.get("url", "Unknown"))
                if name and name not in web_sources:
                    web_sources.append(name)
        return web_sources[:10]  # Limit to top 10
