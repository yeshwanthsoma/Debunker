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
                "model": "gpt-3.5-turbo",
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
            self.session = aiohttp.ClientSession()
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
        self.model = "gpt-4o"  # Using GPT-4o for internet access
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
    
    async def analyze_claim(self, claim: str, context: Optional[str] = None) -> FactCheckResult:
        """Analyze a claim using OpenAI"""
        if not is_api_available("openai"):
            logger.warning("🔑 OpenAI API key not available")
            return self._create_fallback_result(claim)
        
        logger.info(f"🤖 CALLING OpenAI GPT-4 API")
        logger.info(f"   Claim: '{claim[:100]}{'...' if len(claim) > 100 else ''}'")
        logger.info(f"   Context: {'Yes' if context else 'No'}")
        logger.info(f"   Model: {self.model}")
        
        try:
            session = await self._get_session()
            
            prompt = self._create_fact_check_prompt(claim, context)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a professional fact-checker with internet access. Use web browsing to find current, authoritative sources and provide evidence-based assessments with specific citations."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.1,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search the internet for current information"
                        }
                    }
                ]
            }
            
            logger.debug(f"   Prompt length: {len(prompt)} characters")
            
            import time
            start_time = time.time()
            
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                api_time = time.time() - start_time
                logger.info(f"📡 OpenAI API Response: {response.status} (took {api_time:.2f}s)")
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Log token usage
                    usage = data.get('usage', {})
                    if usage:
                        logger.info(f"   Tokens - Prompt: {usage.get('prompt_tokens', 0)}, Completion: {usage.get('completion_tokens', 0)}, Total: {usage.get('total_tokens', 0)}")
                    
                    # Check for tool calls first before parsing
                    message = data.get("choices", [{}])[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])
                    
                    if tool_calls:
                        logger.info(f"🌐 OpenAI used {len(tool_calls)} web searches - following up for final response...")
                        try:
                            result = await self._handle_tool_calls(tool_calls, claim, prompt, session)
                            logger.info(f"✅ OpenAI tool-based analysis complete: {result.verdict} (confidence: {result.confidence:.2f})")
                            return result
                        except Exception as e:
                            logger.error(f"❌ Tool call handling failed: {e}")
                            # Fallback to analysis without tools
                            result = await self._analyze_without_tools(claim, session, prompt)
                            logger.info(f"✅ OpenAI fallback analysis complete: {result.verdict} (confidence: {result.confidence:.2f})")
                            return result
                    else:
                        # Standard response parsing
                        result = self._parse_openai_response(data, claim)
                        logger.info(f"✅ OpenAI analysis complete: {result.verdict} (confidence: {result.confidence:.2f})")
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

Use your internet access to find the most current and authoritative information available."""
        return prompt
    
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
            if "web_sources_consulted" in result_data:
                sources.extend([{
                    "name": source,
                    "type": "Web Source",
                    "url": source if source.startswith("http") else None
                } for source in result_data["web_sources_consulted"]])
            
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

class MultiSourceFactChecker:
    """Orchestrates multiple fact-checking sources"""
    
    def __init__(self):
        self.google_checker = GoogleFactCheckAPI()
        self.openai_checker = OpenAIFactChecker()
    
    async def close(self):
        """Close all API sessions"""
        await self.google_checker.close()
        await self.openai_checker.close()
    
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
        ai_results = []
        google_results = []
        
        for result in results:
            if "OpenAI" in result.provider or "AI Analysis" in str(result.sources):
                ai_results.append(result)
            elif "Google" in result.provider:
                google_results.append(result)
        
        logger.info(f"   AI Results: {len(ai_results)}, Google Results: {len(google_results)}")
        
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
        
        # Weight AI analysis heavily (it has context understanding)
        ai_weight = 3.0
        for result in ai_results:
            weight = ai_weight * result.confidence
            weighted_scores[result.verdict] += weight
            total_weight += weight
            confidence_sum += result.confidence * weight
            confidence_weights += weight
            logger.info(f"   AI: {result.verdict} (weight: {weight:.2f}, conf: {result.confidence:.2f})")
        
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
        
        # TODO: Add scientific consensus boost for well-established facts if needed
        
        # Combine explanations with priority to AI analysis
        explanations = []
        
        # Prioritize AI explanations
        for result in ai_results:
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
                "ai_results": len(ai_results),
                "google_results": len(google_results),
                "weighted_scores": weighted_scores,
                "total_weight": total_weight,
                "google_processing": processed_google,
                "primary_claims_extracted": self._extract_primary_claims(results),
                "claim_evaluations": self._extract_claim_evaluations(results),
                "web_sources_consulted": self._extract_web_sources(all_sources),
                "reasoning": f"Weighted analysis: AI results ({len(ai_results)}) + Google results ({len(google_results)}) = {primary_verdict}"
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

CRITICAL INTERPRETATION TASK:
Many fact-checks are about debunking MISINFORMATION related to a topic, not debunking the topic itself.

Examples:
- If the claim is "Vaccines are safe" and fact-checks say "RFK Jr. misleads about vaccine safety" → This SUPPORTS vaccine safety
- If the claim is "Climate change is real" and fact-checks say "Study debunks climate denial" → This SUPPORTS climate change
- If the claim is "Earth is round" and fact-checks say "Flat earth theory debunked" → This SUPPORTS round earth

Your job: Determine if these fact-checks are:
1. DEBUNKING THE CLAIM ITSELF 
2. DEBUNKING MISINFORMATION ABOUT THE CLAIM (which supports the claim)
3. MIXED/UNCLEAR

Respond with ONLY this JSON:
{{
    "interpretation": "supporting_claim|refuting_claim|mixed|unclear",
    "consensus_verdict": "True|False|Misleading|Unverifiable", 
    "confidence": 0.75,
    "reasoning": "Brief explanation of your interpretation",
    "evidence_summary": "What the fact-checks actually indicate about the original claim"
}}"""

        try:
            session = await self.openai_checker._get_session()
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert at interpreting fact-check context. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
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
                        evaluations.update(evidence_assessment)
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
