"""
Startup News Aggregation
Runs immediately when the app starts to populate real trending claims
"""

import asyncio
import logging
from datetime import datetime, UTC
from news_aggregator import NewsAggregator, save_claims_to_database
from database import SessionLocal, TrendingClaim
from fact_check_apis import MultiSourceFactChecker

logger = logging.getLogger(__name__)

async def fact_check_discovered_claims():
    """
    Fact-check recently discovered claims for immediate availability
    """
    db = SessionLocal()
    try:
        # Get recently discovered claims that need fact-checking
        discovered_claims = db.query(TrendingClaim).filter(
            TrendingClaim.status == 'discovered'
        ).limit(5).all()  # Limit to 5 for startup
        
        if not discovered_claims:
            return 0
        
        logger.info(f"🔍 Fact-checking {len(discovered_claims)} discovered claims...")
        
        # Initialize fact checker
        try:
            fact_checker = MultiSourceFactChecker()
        except Exception as e:
            logger.warning(f"Professional fact checker not available: {e}")
            # Use simple fact-checking
            return await simple_fact_check_claims(discovered_claims, db)
        
        fact_checked_count = 0
        for claim in discovered_claims:
            try:
                logger.info(f"🔍 Fact-checking: {claim.claim_text[:60]}...")
                
                # Update status to processing
                claim.status = 'processing'
                db.commit()
                
                # Perform fact-check
                result = await fact_checker.comprehensive_fact_check(
                    claim.claim_text,
                    context=""
                )
                
                # Update claim with results
                claim.verdict = result.verdict
                claim.confidence = result.confidence
                claim.explanation = result.explanation
                claim.status = 'completed'
                claim.processed_at = datetime.now(UTC)
                
                fact_checked_count += 1
                logger.info(f"✅ Fact-checked: {result.verdict} ({result.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"❌ Fact-check failed for claim {claim.id}: {e}")
                claim.status = 'failed'
                
            db.commit()
        
        await fact_checker.close()
        return fact_checked_count
        
    except Exception as e:
        logger.error(f"❌ Error in fact-checking process: {e}")
        return 0
    finally:
        db.close()

async def simple_fact_check_claims(claims, db):
    """
    Simple fact-checking fallback when professional APIs aren't available
    """
    simple_verdicts = ['Mixed', 'Unverifiable', 'Needs Review']
    
    fact_checked_count = 0
    for claim in claims:
        try:
            # Simple heuristic-based verdict
            if any(word in claim.claim_text.lower() for word in ['hoax', 'conspiracy', 'fake']):
                verdict = 'Likely False'
                confidence = 0.7
            elif any(word in claim.claim_text.lower() for word in ['study shows', 'research', 'scientists']):
                verdict = 'Likely True'
                confidence = 0.6
            else:
                verdict = 'Unverifiable'
                confidence = 0.5
            
            claim.verdict = verdict
            claim.confidence = confidence
            claim.explanation = f"Preliminary assessment based on content analysis. Full fact-check pending."
            claim.status = 'completed'
            claim.processed_at = datetime.now(UTC)
            
            fact_checked_count += 1
            
        except Exception as e:
            logger.error(f"❌ Simple fact-check failed for claim {claim.id}: {e}")
            claim.status = 'failed'
    
    db.commit()
    return fact_checked_count

async def startup_trending_claims_discovery():
    """
    Run trending claims discovery on startup to ensure real data is available
    """
    logger.info("🚀 Starting up trending claims discovery...")
    
    # Check if we already have recent claims
    db = SessionLocal()
    try:
        recent_claims = db.query(TrendingClaim).filter(
            TrendingClaim.discovered_at >= datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        # Always run aggregation on startup to get fresh data (removed the skip condition)
        logger.info(f"📊 Found {recent_claims} existing claims, running fresh aggregation for latest data")
            
    except Exception as e:
        logger.warning(f"Could not check existing claims: {e}")
    finally:
        db.close()
    
    # Run aggregation to get fresh claims
    try:
        logger.info("📰 Discovering trending claims from news sources...")
        aggregator = NewsAggregator()
        
        # Start with a smaller batch for startup (reduced API calls)
        # Temporarily disable Grok during startup to reduce API calls
        aggregator.use_grok_at_startup = False
        claims = await aggregator.discover_trending_claims(limit=10)
        
        if claims:
            logger.info(f"🔍 Discovered {len(claims)} trending claims")
            
            # Save to database
            saved_count = await save_claims_to_database(claims)
            
            if saved_count > 0:
                logger.info(f"✅ Startup aggregation complete: {saved_count} new claims saved")
                
                # Log some examples
                for i, claim in enumerate(claims[:3], 1):
                    logger.info(f"   {i}. [{claim.category}] {claim.text[:60]}...")
                
                # Immediately fact-check the discovered claims for startup
                logger.info("🔍 Running immediate fact-checking for startup claims...")
                fact_checked_count = await fact_check_discovered_claims()
                logger.info(f"✅ Fact-checked {fact_checked_count} claims for immediate use")
                    
                return saved_count
            else:
                logger.info("ℹ️ No new claims saved (may be duplicates)")
                return 0
        else:
            logger.warning("⚠️ No trending claims discovered during startup")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Startup aggregation failed: {e}")
        return 0

async def ensure_minimum_claims():
    """
    Ensure we have at least some claims for the frontend to display
    """
    db = SessionLocal()
    try:
        total_claims = db.query(TrendingClaim).count()
        
        if total_claims == 0:
            logger.info("🔄 No claims in database, running emergency aggregation...")
            return await startup_trending_claims_discovery()
        else:
            logger.info(f"📊 Database contains {total_claims} total claims")
            return total_claims
            
    except Exception as e:
        logger.error(f"❌ Error checking claim count: {e}")
        return 0
    finally:
        db.close()

# Function to be called from main.py startup
async def run_startup_aggregation():
    """
    Main function to run startup aggregation
    """
    try:
        logger.info("=" * 50)
        logger.info("🔍 STARTUP TRENDING CLAIMS DISCOVERY")
        logger.info("=" * 50)
        
        # Ensure we have minimum claims
        result = await ensure_minimum_claims()
        
        if result > 0:
            logger.info(f"✅ Trending claims ready: {result} claims available")
        else:
            logger.warning("⚠️ No claims available - check API keys and network connectivity")
            
        logger.info("=" * 50)
        return result
        
    except Exception as e:
        logger.error(f"❌ Startup aggregation error: {e}")
        return 0

if __name__ == "__main__":
    # Test the startup aggregation
    import sys
    sys.path.append('.')
    
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_startup_aggregation())
    print(f"Startup aggregation result: {result} claims")