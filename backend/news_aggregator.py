"""
News Aggregation Service for Trending Claims Discovery
Fetches trending topics from news sources and social media
"""

import asyncio
import aiohttp
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from newsapi import NewsApiClient
import praw
import openai
from sqlalchemy.orm import Session

from config import get_api_key, is_api_available
from database import get_db, TrendingClaim, ClaimSource, NewsSource
from grok_integration import get_grok_trending_claims, enhance_trending_claim_with_grok

logger = logging.getLogger(__name__)

@dataclass
class ExtractedClaim:
    """Represents a claim extracted from news/social media"""
    text: str
    title: str
    category: str
    source_url: str
    source_type: str
    controversy_score: float
    keywords: List[str]
    entities: List[str]
    original_content: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None

class NewsAggregator:
    """Main news aggregation service"""
    
    def __init__(self):
        self.news_api_key = get_api_key("news_api")
        self.openai_key = get_api_key("openai")
        self.reddit_client = None
        self.use_grok_at_startup = True  # Flag to control Grok usage
        self.setup_clients()
    
    def setup_clients(self):
        """Initialize API clients"""
        try:
            # Reddit client setup (if credentials available)
            reddit_client_id = get_api_key("reddit_client_id")
            reddit_secret = get_api_key("reddit_secret")
            
            if reddit_client_id and reddit_secret:
                self.reddit_client = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_secret,
                    user_agent="TruthLens Fact Checker v1.0"
                )
                logger.info("✅ Reddit client initialized")
            else:
                logger.warning("⚠️ Reddit API credentials not available")
                
        except Exception as e:
            logger.error(f"❌ Error setting up clients: {e}")
    
    async def discover_trending_claims(self, limit: int = 50) -> List[ExtractedClaim]:
        """Main method to discover trending claims from all sources"""
        logger.info(f"🔍 Starting trending claims discovery (limit: {limit})")
        
        all_claims = []
        
        # Fetch from different sources concurrently
        tasks = []
        
        if self.news_api_key:
            tasks.append(self.fetch_news_claims())
        
        if self.reddit_client:
            tasks.append(self.fetch_reddit_claims())
        
        # Add Grok trending claims (skip during startup optimization)
        if is_api_available("grok") and self.use_grok_at_startup:
            tasks.append(self.fetch_grok_claims())
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_claims.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"❌ Error in news aggregation: {result}")
        
        # Sort by controversy score and limit
        all_claims.sort(key=lambda x: x.controversy_score, reverse=True)
        
        logger.info(f"✅ Discovered {len(all_claims)} claims, returning top {limit}")
        return all_claims[:limit]
    
    async def fetch_news_claims(self) -> List[ExtractedClaim]:
        """Fetch claims from news sources using NewsAPI"""
        logger.info("📰 Fetching trending news claims...")
        
        claims = []
        
        try:
            # Initialize NewsAPI client
            newsapi = NewsApiClient(api_key=self.news_api_key)
            
            # Categories to search for debatable content (valid NewsAPI categories)
            categories = ['general', 'health', 'science', 'technology', 'business']
            
            for category in categories:
                try:
                    # Get top headlines (reduced for startup)
                    headlines = newsapi.get_top_headlines(
                        category=category,
                        language='en',
                        page_size=5  # Reduced from 20 to 5 per category
                    )
                    
                    if headlines['status'] == 'ok':
                        for article in headlines['articles']:
                            # Extract claims from article
                            extracted = await self.extract_claims_from_article(
                                article, category
                            )
                            claims.extend(extracted)
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching {category} news: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"❌ Error in news fetching: {e}")
        
        logger.info(f"📰 Extracted {len(claims)} claims from news sources")
        return claims
    
    async def fetch_reddit_claims(self) -> List[ExtractedClaim]:
        """Fetch claims from Reddit trending posts"""
        logger.info("🔸 Fetching trending Reddit claims...")
        
        claims = []
        
        try:
            if not self.reddit_client:
                return claims
            
            # Subreddits known for debatable content (reduced for startup)
            subreddits = [
                'worldnews', 'news', 'science'  # Reduced from 9 to 3 subreddits
            ]
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get hot posts (reduced for startup)
                    for post in subreddit.hot(limit=3):
                        # Accept both self posts and link posts, just check title length
                        if len(post.title) < 20:
                            continue
                        
                        # Extract claims from Reddit post
                        extracted = await self.extract_claims_from_reddit_post(
                            post, subreddit_name
                        )
                        claims.extend(extracted)
                
                except Exception as e:
                    logger.error(f"❌ Error fetching {subreddit_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"❌ Error in Reddit fetching: {e}")
        
        logger.info(f"🔸 Extracted {len(claims)} claims from Reddit")
        return claims
    
    async def fetch_grok_claims(self) -> List[ExtractedClaim]:
        """Fetch trending claims from Grok social media analysis"""
        logger.info("🌐 Fetching trending claims from Grok...")
        
        claims = []
        
        try:
            # Get trending misinformation from Grok
            categories = ['health', 'politics', 'science', 'technology', 'environment']
            grok_claims = await get_grok_trending_claims(categories)
            
            for grok_claim in grok_claims:
                try:
                    claim_text = grok_claim.get('claim', '')
                    category = grok_claim.get('category', 'Social Media')
                    viral_score = grok_claim.get('viral_score', 0.5)
                    engagement_level = grok_claim.get('engagement_level', 'medium')
                    
                    if len(claim_text) < 20:  # Skip very short claims
                        continue
                    
                    # Convert viral score to controversy score
                    controversy_score = min(viral_score + 0.2, 1.0)  # Boost social media claims
                    
                    # Create ExtractedClaim
                    claim = ExtractedClaim(
                        text=claim_text,
                        title=f"Trending on Social Media: {claim_text[:60]}...",
                        category=self.categorize_claim(claim_text, category),
                        source_url="https://x.com/search",  # Generic X search URL
                        source_type='grok',
                        controversy_score=controversy_score,
                        keywords=self.extract_keywords(claim_text),
                        entities=self.extract_entities(claim_text),
                        original_content=f"Trending claim detected by Grok with {engagement_level} engagement",
                        author="Social Media Users",
                        published_at=datetime.now()
                    )
                    claims.append(claim)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing Grok claim: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"❌ Error in Grok claims fetching: {e}")
        
        logger.info(f"🌐 Extracted {len(claims)} claims from Grok")
        return claims
    
    async def extract_claims_from_article(self, article: Dict, category: str) -> List[ExtractedClaim]:
        """Extract factual claims from a news article"""
        claims = []
        
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            url = article.get('url', '')
            
            # Combine text for analysis
            full_text = f"{title}. {description}. {content}"
            
            if len(full_text) < 50:
                return claims
            
            # Use AI to extract factual claims
            extracted_claims = await self.ai_extract_claims(full_text)
            
            for claim_text in extracted_claims:
                if len(claim_text) < 20:  # Skip very short claims
                    continue
                
                # Calculate controversy score
                controversy = self.calculate_controversy_score(claim_text, full_text)
                
                # Lower threshold to include more claims for testing
                if controversy > 0.1:  # Lowered from 0.3 to 0.1
                    claim = ExtractedClaim(
                        text=claim_text,
                        title=title,
                        category=self.categorize_claim(claim_text, category),
                        source_url=url,
                        source_type='news',
                        controversy_score=controversy,
                        keywords=self.extract_keywords(claim_text),
                        entities=self.extract_entities(claim_text),
                        original_content=full_text[:1000],  # First 1000 chars
                        author=article.get('author'),
                        published_at=self.parse_datetime(article.get('publishedAt'))
                    )
                    claims.append(claim)
        
        except Exception as e:
            logger.error(f"❌ Error extracting claims from article: {e}")
        
        return claims
    
    async def extract_claims_from_reddit_post(self, post, subreddit_name: str) -> List[ExtractedClaim]:
        """Extract claims from Reddit post"""
        claims = []
        
        try:
            title = post.title
            # For link posts, use title + any available text content
            content = getattr(post, 'selftext', '') or ''
            # Also use post URL as context for link posts
            if not content and hasattr(post, 'url'):
                content = f"Link post: {post.url}"
            full_text = f"{title}. {content}"
            
            if len(full_text) < 30:
                return claims
            
            # Extract claims using AI
            extracted_claims = await self.ai_extract_claims(full_text)
            
            for claim_text in extracted_claims:
                if len(claim_text) < 20:
                    continue
                
                controversy = self.calculate_controversy_score(claim_text, full_text)
                
                if controversy > 0.2:  # Lowered threshold for Reddit
                    claim = ExtractedClaim(
                        text=claim_text,
                        title=title,
                        category=self.categorize_subreddit(subreddit_name),
                        source_url=f"https://reddit.com{post.permalink}",
                        source_type='reddit',
                        controversy_score=controversy,
                        keywords=self.extract_keywords(claim_text),
                        entities=self.extract_entities(claim_text),
                        original_content=full_text[:1000],
                        author=str(post.author) if post.author else None,
                        published_at=datetime.fromtimestamp(post.created_utc)
                    )
                    claims.append(claim)
        
        except Exception as e:
            logger.error(f"❌ Error extracting claims from Reddit post: {e}")
        
        return claims
    
    async def ai_extract_claims(self, text: str) -> List[str]:
        """Use AI to extract factual claims from text"""
        if not self.openai_key:
            # Fallback to simple pattern matching
            return self.simple_claim_extraction(text)
        
        try:
            client = openai.OpenAI(api_key=self.openai_key)
            
            prompt = f"""Extract specific factual claims from this text that could be fact-checked. 
Focus on statements that:
1. Make specific assertions about reality
2. Could be verified or debunked
3. Are potentially controversial or debatable

Text: {text[:1500]}

Return ONLY a JSON list of extracted claims, like: ["claim 1", "claim 2"]
Maximum 2 claims. Each claim should be a complete sentence."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Clean up the response - remove any markdown formatting
                content = content.replace('```json', '').replace('```', '').strip()
                claims = json.loads(content)
                if isinstance(claims, list):
                    return [claim for claim in claims if isinstance(claim, str) and len(claim) > 10]
                else:
                    logger.warning("AI response is not a list, using fallback")
                    return self.simple_claim_extraction(text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AI response as JSON: {e}")
                logger.warning(f"Response content: {content[:200]}...")
                return self.simple_claim_extraction(text)
        
        except Exception as e:
            logger.error(f"❌ AI claim extraction failed: {e}")
            return self.simple_claim_extraction(text)
        
        return []
    
    def simple_claim_extraction(self, text: str) -> List[str]:
        """Simple pattern-based claim extraction as fallback"""
        claims = []
        
        # Look for sentences with strong assertion words
        assertion_patterns = [
            r'[^.!?]*(?:is|are|was|were|will be|has been|have been)[^.!?]*[.!?]',
            r'[^.!?]*(?:proves?|shows?|demonstrates?|reveals?)[^.!?]*[.!?]',
            r'[^.!?]*(?:claims?|states?|argues?|believes?)[^.!?]*[.!?]'
        ]
        
        for pattern in assertion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Limit to 2 per pattern
                clean_claim = match.strip()
                if 20 <= len(clean_claim) <= 200:
                    claims.append(clean_claim)
        
        return claims[:3]  # Maximum 3 claims
    
    def calculate_controversy_score(self, claim: str, context: str) -> float:
        """Calculate how controversial/debatable a claim is"""
        score = 0.0
        
        # Controversial keywords
        controversial_words = [
            'conspiracy', 'hoax', 'fake', 'lie', 'fraud', 'scam',
            'hidden', 'secret', 'cover-up', 'truth', 'really',
            'actually', 'proves', 'debunked', 'myth', 'false',
            'dangerous', 'toxic', 'harmful', 'safe', 'effective',
            'claims', 'alleges', 'accused', 'lawsuit', 'ruling',
            'verdict', 'trial', 'court', 'evidence', 'study',
            'research', 'scientists', 'experts', 'officials'
        ]
        
        claim_lower = claim.lower()
        for word in controversial_words:
            if word in claim_lower:
                score += 0.1
        
        # Absolute statements (often controversial)
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'completely', 'totally']
        for word in absolute_words:
            if word in claim_lower:
                score += 0.05
        
        # Question marks or uncertainty indicators
        if '?' in claim or any(word in claim_lower for word in ['maybe', 'perhaps', 'possibly']):
            score += 0.1
        
        # Length factor (longer claims often more complex/controversial)
        if len(claim) > 100:
            score += 0.1
        
        # Base score for any factual claim that made it this far
        if len(claim) > 20:
            score += 0.05
        
        # Context factors
        context_lower = context.lower()
        if any(word in context_lower for word in ['debate', 'controversy', 'disputed', 'argument']):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def categorize_claim(self, claim: str, default_category: str) -> str:
        """Categorize a claim based on its content"""
        claim_lower = claim.lower()
        
        if any(word in claim_lower for word in ['vaccine', 'medicine', 'health', 'disease', 'treatment']):
            return 'Health'
        elif any(word in claim_lower for word in ['climate', 'global warming', 'environment', 'carbon']):
            return 'Environment'
        elif any(word in claim_lower for word in ['election', 'vote', 'government', 'president', 'policy']):
            return 'Politics'
        elif any(word in claim_lower for word in ['study', 'research', 'scientist', 'data', 'experiment']):
            return 'Science'
        elif any(word in claim_lower for word in ['economy', 'market', 'financial', 'money', 'inflation']):
            return 'Economics'
        else:
            return default_category.title()
    
    def categorize_subreddit(self, subreddit: str) -> str:
        """Map subreddit to category"""
        mapping = {
            'worldnews': 'Politics',
            'news': 'General',
            'science': 'Science',
            'politics': 'Politics',
            'conspiracy': 'Conspiracy',
            'unpopularopinion': 'Opinion',
            'changemyview': 'Debate',
            'todayilearned': 'Education',
            'askscience': 'Science'
        }
        return mapping.get(subreddit, 'General')
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common words
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'were', 'been', 'their', 'said'}
        keywords = [word for word in words if word not in stop_words]
        
        # Return top 5 most relevant
        return list(set(keywords))[:5]
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (people, organizations, places)"""
        # Simple pattern-based entity extraction
        entities = []
        
        # Look for capitalized words/phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Common organization patterns
        org_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms like NASA, FDA
            r'\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|Ltd)\b'  # Companies
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(set(entities))[:5]
    
    def parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from various formats"""
        if not date_str:
            return None
        
        try:
            # ISO format from NewsAPI
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
            except:
                return None

async def save_claims_to_database(claims: List[ExtractedClaim]) -> int:
    """Save extracted claims to database"""
    saved_count = 0
    
    # Get database session
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        for claim in claims:
            # Check if claim already exists
            existing = db.query(TrendingClaim).filter(
                TrendingClaim.claim_text == claim.text
            ).first()
            
            if existing:
                continue  # Skip duplicates
            
            # Create new trending claim
            db_claim = TrendingClaim(
                claim_text=claim.text,
                title=claim.title,
                category=claim.category,
                source_type=claim.source_type,
                source_url=claim.source_url,
                trending_score=claim.controversy_score,
                controversy_level=claim.controversy_score,
                status='discovered',
                tags=claim.keywords,
                keywords=claim.keywords,
                related_entities=claim.entities,
                discovered_at=datetime.utcnow()
            )
            
            db.add(db_claim)
            db.flush()  # Get the ID
            
            # Create source record
            db_source = ClaimSource(
                claim_id=db_claim.id,
                source_name=claim.source_type.title(),
                source_url=claim.source_url,
                source_type=claim.source_type,
                original_content=claim.original_content,
                extracted_claim=claim.text,
                author=claim.author,
                published_at=claim.published_at,
                created_at=datetime.utcnow()
            )
            
            db.add(db_source)
            saved_count += 1
        
        db.commit()
        logger.info(f"✅ Saved {saved_count} new claims to database")
    
    except Exception as e:
        logger.error(f"❌ Error saving claims to database: {e}")
        db.rollback()
    finally:
        db.close()
    
    return saved_count

# Main function for testing
async def main():
    """Test the news aggregation service"""
    aggregator = NewsAggregator()
    claims = await aggregator.discover_trending_claims(limit=10)
    
    print(f"\n🔍 Discovered {len(claims)} trending claims:")
    for i, claim in enumerate(claims, 1):
        print(f"\n{i}. [{claim.category}] {claim.title}")
        print(f"   Claim: {claim.text}")
        print(f"   Controversy: {claim.controversy_score:.2f}")
        print(f"   Source: {claim.source_type} - {claim.source_url}")
    
    # Save to database
    saved = await save_claims_to_database(claims)
    print(f"\n💾 Saved {saved} claims to database")

if __name__ == "__main__":
    asyncio.run(main())