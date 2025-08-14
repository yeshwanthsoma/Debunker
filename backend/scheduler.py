"""
Background Task Scheduler for Trending Claims Discovery
Automatically runs news aggregation and claim processing on schedule
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager
import psutil

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from news_aggregator import NewsAggregator, save_claims_to_database
from config import get_settings, is_api_available

logger = logging.getLogger(__name__)

class TrendingClaimsScheduler:
    """Background scheduler for trending claims discovery"""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.aggregator = NewsAggregator()
        self.settings = get_settings()
        self.is_running = False
        
        # Scheduler configuration
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': AsyncIOExecutor()
        }
        job_defaults = {
            'coalesce': True,  # Combine multiple pending executions
            'max_instances': 1,  # Only one instance of each job at a time
            'misfire_grace_time': 300  # 5 minutes grace period
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
    
    async def start(self):
        """Start the background scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            # Add scheduled jobs
            self._setup_jobs()
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info("🕐 Background scheduler started successfully")
            logger.info("📋 Scheduled jobs:")
            for job in self.scheduler.get_jobs():
                logger.info(f"   - {job.id}: {job.next_run_time}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")
            raise
    
    async def stop(self):
        """Stop the background scheduler"""
        if not self.is_running:
            return
        
        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("🛑 Background scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {e}")
    
    def _setup_jobs(self):
        """Setup all scheduled jobs"""
        
        # Job 1: Quick news check every 15 minutes (faster for testing)
        self.scheduler.add_job(
            func=self._quick_news_aggregation,
            trigger=IntervalTrigger(minutes=15),
            id='quick_news_check',
            name='Quick News Aggregation',
            replace_existing=True
        )
        
        # Note: Startup aggregation now runs directly in main.py startup, not as a scheduled job
        
        # Job 2: Full news aggregation every 2 hours
        self.scheduler.add_job(
            func=self._full_news_aggregation,
            trigger=IntervalTrigger(hours=2),
            id='full_news_aggregation',
            name='Full News Aggregation',
            replace_existing=True
        )
        
        # Job 3: Social media trending check every hour (if available)
        if is_api_available("grok") or is_api_available("reddit_client_id"):
            self.scheduler.add_job(
                func=self._social_media_check,
                trigger=IntervalTrigger(hours=1),
                id='social_media_check',
                name='Social Media Trending Check',
                replace_existing=True
            )
        
        # Job 4: Professional fact-check pending claims every 30 minutes
        self.scheduler.add_job(
            func=self._professional_fact_check_new_claims,
            trigger=IntervalTrigger(minutes=30),
            id='fact_check_pending',
            name='Fact-Check Pending Claims',
            replace_existing=True
        )
        
        # Job 5: Daily cleanup and maintenance (at 3 AM UTC)
        self.scheduler.add_job(
            func=self._daily_maintenance,
            trigger=CronTrigger(hour=3, minute=0),
            id='daily_maintenance',
            name='Daily Maintenance',
            replace_existing=True
        )
        
        # Job 6: Weekly analytics processing (Sundays at 4 AM UTC)
        self.scheduler.add_job(
            func=self._weekly_analytics,
            trigger=CronTrigger(day_of_week=6, hour=4, minute=0),
            id='weekly_analytics',
            name='Weekly Analytics Processing',
            replace_existing=True
        )
    
    async def _quick_news_aggregation(self):
        """Quick news aggregation - limit to 20 claims"""
        logger.info("🔍 Starting quick news aggregation...")
        
        try:
            start_time = time.time()
            
            # Discover trending claims with lower limit for speed
            claims = await self.aggregator.discover_trending_claims(limit=20)
            
            if claims:
                # Save to database
                saved_count = await save_claims_to_database(claims)
                
                execution_time = time.time() - start_time
                logger.info(f"✅ Quick aggregation complete: {saved_count} claims saved in {execution_time:.1f}s")
            else:
                logger.info("📰 No new trending claims found in quick check")
                
        except Exception as e:
            logger.error(f"❌ Quick news aggregation failed: {e}")
    
    async def _startup_aggregation(self):
        """Immediate aggregation after startup - runs once"""
        logger.info("🚀 Running immediate startup aggregation...")
        
        try:
            start_time = time.time()
            
            # Use higher limit for startup to get good initial data
            claims = await self.aggregator.discover_trending_claims(limit=15)
            
            if claims:
                # Save to database
                saved_count = await save_claims_to_database(claims)
                
                # Professional fact-check for immediate results
                if saved_count > 0:
                    fact_checked_count = await self._professional_fact_check_new_claims()
                    
                execution_time = time.time() - start_time
                logger.info(f"✅ Startup aggregation complete: {saved_count} claims saved, {fact_checked_count} fact-checked in {execution_time:.1f}s")
            else:
                logger.info("📰 No new claims discovered in startup aggregation")
                
        except Exception as e:
            logger.error(f"❌ Startup aggregation failed: {e}")
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within safe limits for Railway"""
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                logger.warning(f"⚠️ High memory usage detected: {memory_percent:.1f}% - reducing processing load")
                return False
            elif memory_percent > 70:
                logger.info(f"📊 Memory usage: {memory_percent:.1f}% - monitoring closely")
            return True
        except Exception as e:
            logger.error(f"❌ Error checking memory usage: {e}")
            return True  # Assume safe if can't check

    async def _professional_fact_check_new_claims(self):
        """Professional fact-checking for trending claims using multi-source APIs"""
        from database import SessionLocal, TrendingClaim
        from fact_check_apis import MultiSourceFactChecker
        
        # Check memory usage before intensive processing
        if not self._check_memory_usage():
            logger.warning("⚠️ Skipping fact-checking due to high memory usage")
            return 0
        
        db = SessionLocal()
        try:
            # Get recently discovered claims that need fact-checking
            discovered_claims = db.query(TrendingClaim).filter(
                TrendingClaim.status == 'discovered'
            ).limit(10).all()
            
            if not discovered_claims:
                return 0
            
            fact_checked_count = 0
            
            # Initialize professional fact checker if APIs are available
            multi_source_checker = None
            try:
                if is_api_available("google_fact_check") or is_api_available("openai") or is_api_available("anthropic"):
                    multi_source_checker = MultiSourceFactChecker()
                    logger.info("✅ Professional fact-checker initialized for trending claims")
                else:
                    logger.info("⚠️ No professional fact-checking APIs available, using enhanced fallback")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize professional fact-checker: {e}")
            
            for claim in discovered_claims:
                try:
                    claim.status = 'processing'
                    db.commit()  # Mark as processing immediately
                    
                    # Use professional fact-checking if available
                    if multi_source_checker:
                        try:
                            logger.info(f"🔍 Professional fact-checking: '{claim.claim_text[:60]}...'")
                            fact_check_result = await multi_source_checker.comprehensive_fact_check(
                                claim.claim_text, 
                                claim.title or ""
                            )
                            
                            # Extract results from professional fact-check (always returns Dict)
                            if isinstance(fact_check_result, dict):
                                claim.verdict = fact_check_result.get('verdict', 'Unverifiable')
                                claim.confidence = fact_check_result.get('confidence', 0.5)
                                claim.explanation = fact_check_result.get('explanation', 'Professional fact-check completed')
                                
                                # Extract evidence summary from details
                                details = fact_check_result.get('details', {})
                                if details:
                                    evidence_parts = []
                                    if details.get('primary_claims_extracted'):
                                        evidence_parts.append(f"Primary claims: {', '.join(details['primary_claims_extracted'][:3])}")
                                    if details.get('web_sources_consulted'):
                                        evidence_parts.append(f"Sources consulted: {', '.join(details['web_sources_consulted'][:3])}")
                                    if details.get('source_count'):
                                        evidence_parts.append(f"Total sources: {details['source_count']}")
                                    if details.get('agreement_level'):
                                        evidence_parts.append(f"Source agreement: {details['agreement_level']:.0%}")
                                    
                                    claim.evidence_summary = '; '.join(evidence_parts) if evidence_parts else None
                                
                                # Store fact-checking source information
                                sources = fact_check_result.get('sources', [])
                                if sources:
                                    source_names = [s.get('name', 'Unknown') for s in sources[:5]]
                                    if not claim.evidence_summary:
                                        claim.evidence_summary = f"Sources: {', '.join(source_names)}"
                                    
                                    # Store fact-checking sources in database for proper display
                                    from database import ClaimSource
                                    logger.info(f"📊 Processing {len(sources)} fact-checking sources for claim {claim.id}")
                                    
                                    for i, fc_source in enumerate(sources[:5]):  # Limit to top 5 fact-checking sources
                                        logger.debug(f"   Source {i+1}: {fc_source}")
                                        
                                        # Get the actual fact-checking URL 
                                        source_url = fc_source.get('url', '').strip()
                                        source_name = fc_source.get('name', 'Unknown Fact-Checker')
                                        
                                        # Log what we found
                                        if source_url:
                                            logger.info(f"✅ Found URL for {source_name}: {source_url[:80]}...")
                                        else:
                                            logger.warning(f"⚠️ No URL provided for {source_name} - storing as name only")
                                            source_url = None  # Store as NULL in database
                                        
                                        fact_check_source = ClaimSource(
                                            claim_id=claim.id,
                                            source_name=source_name,
                                            source_url=source_url,
                                            source_type='fact_check_source',  # Distinguish from news source
                                            original_content=f"Fact-checking source: {source_name}",
                                            extracted_claim=claim.claim_text,
                                            source_reliability=fc_source.get('rating', 0.8),  # Default reliability for fact-checkers
                                            created_at=datetime.now()
                                        )
                                        db.add(fact_check_source)
                                    
                            else:
                                # Fallback for unexpected result format
                                logger.warning(f"Unexpected fact-check result format: {type(fact_check_result)}")
                                claim.verdict = 'Unverifiable'
                                claim.confidence = 0.5
                                claim.explanation = 'Professional fact-check completed with unexpected result format'
                            
                            claim.status = 'completed'
                            logger.info(f"✅ Professional fact-check complete: {claim.verdict} ({claim.confidence:.0%} confidence)")
                            
                        except Exception as e:
                            logger.error(f"❌ Professional fact-check failed for claim {claim.id}: {e}")
                            # Fall back to enhanced heuristic method
                            self._enhanced_heuristic_fact_check(claim)
                    else:
                        # Use enhanced heuristic method when no APIs available
                        self._enhanced_heuristic_fact_check(claim)
                    
                    claim.processed_at = datetime.now()
                    fact_checked_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Fact-check failed for claim {claim.id}: {e}")
                    claim.status = 'failed'
                    claim.explanation = f"Error during fact-checking: {str(e)}"
                    claim.processed_at = datetime.now()
            
            db.commit()
            return fact_checked_count
            
        except Exception as e:
            logger.error(f"❌ Error in professional fact-checking: {e}")
            db.rollback()  # Rollback transaction on error
            return 0
        finally:
            db.close()
    
    def _enhanced_heuristic_fact_check(self, claim):
        """Enhanced heuristic-based fact-checking as fallback"""
        text_lower = claim.claim_text.lower()
        
        # Enhanced heuristic analysis
        if any(word in text_lower for word in ['hoax', 'conspiracy', 'fake news', 'lie', 'debunked', 'false']):
            verdict = 'False'
            confidence = 0.8
            explanation = "Content analysis suggests this claim contains misinformation indicators."
        elif any(word in text_lower for word in ['study shows', 'research confirms', 'scientists', 'data shows', 'proven', 'evidence']):
            verdict = 'True'
            confidence = 0.7
            explanation = "Content analysis indicates this claim references credible sources or research."
        elif any(word in text_lower for word in ['claims', 'alleges', 'reportedly', 'rumor', 'speculation']):
            verdict = 'Unverifiable'
            confidence = 0.6
            explanation = "This claim appears to be unverified or speculative based on language indicators."
        elif any(word in text_lower for word in ['partially', 'some', 'mixed', 'context']):
            verdict = 'Mixed'
            confidence = 0.6
            explanation = "Content analysis suggests this claim may be partially true or requires additional context."
        else:
            verdict = 'Mixed'
            confidence = 0.5
            explanation = "Initial assessment completed. This claim requires further verification."
        
        # Update claim with enhanced heuristic results
        claim.verdict = verdict
        claim.confidence = confidence
        claim.explanation = explanation
        claim.status = 'completed'
    
    async def _full_news_aggregation(self):
        """Full news aggregation - comprehensive discovery"""
        logger.info("📰 Starting full news aggregation...")
        
        try:
            start_time = time.time()
            
            # Discover trending claims with higher limit
            claims = await self.aggregator.discover_trending_claims(limit=100)
            
            if claims:
                # Save to database
                saved_count = await save_claims_to_database(claims)
                
                execution_time = time.time() - start_time
                logger.info(f"✅ Full aggregation complete: {saved_count} claims saved in {execution_time:.1f}s")
                
                # Send notification if many new claims found
                if saved_count > 10:
                    logger.info(f"🚨 High activity detected: {saved_count} new claims discovered")
                    
            else:
                logger.info("📰 No new trending claims found in full aggregation")
                
        except Exception as e:
            logger.error(f"❌ Full news aggregation failed: {e}")
    
    async def _social_media_check(self):
        """Check social media for trending topics"""
        logger.info("🌐 Starting social media trending check...")
        
        try:
            # This will only run if social media APIs are available
            if is_api_available("grok"):
                from grok_integration import get_grok_trending_claims
                
                grok_claims = await get_grok_trending_claims(['health', 'politics', 'science'])
                logger.info(f"🌐 Found {len(grok_claims)} trending claims from Grok")
                
            if is_api_available("reddit_client_id"):
                reddit_claims = await self.aggregator.fetch_reddit_claims()
                logger.info(f"🔸 Found {len(reddit_claims)} claims from Reddit")
                
        except Exception as e:
            logger.error(f"❌ Social media check failed: {e}")
    
    async def _daily_maintenance(self):
        """Daily database cleanup and maintenance"""
        logger.info("🧹 Starting daily maintenance...")
        
        try:
            from database import SessionLocal, TrendingClaim
            
            db = SessionLocal()
            
            try:
                # Clean up old unprocessed claims (older than 7 days)
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                deleted_count = db.query(TrendingClaim).filter(
                    TrendingClaim.status == 'discovered',
                    TrendingClaim.discovered_at < cutoff_date
                ).delete()
                
                # Clean up failed claims older than 3 days
                failed_cutoff = datetime.utcnow() - timedelta(days=3)
                
                failed_deleted = db.query(TrendingClaim).filter(
                    TrendingClaim.status == 'failed',
                    TrendingClaim.discovered_at < failed_cutoff
                ).delete()
                
                db.commit()
                
                logger.info(f"🧹 Maintenance complete: {deleted_count} old claims, {failed_deleted} failed claims removed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Daily maintenance failed: {e}")
    
    async def _weekly_analytics(self):
        """Weekly analytics processing and reporting"""
        logger.info("📊 Starting weekly analytics processing...")
        
        try:
            from database import SessionLocal, TrendingClaim
            from sqlalchemy import func
            
            db = SessionLocal()
            
            try:
                # Calculate weekly statistics
                week_ago = datetime.utcnow() - timedelta(days=7)
                
                total_claims = db.query(TrendingClaim).filter(
                    TrendingClaim.discovered_at >= week_ago
                ).count()
                
                processed_claims = db.query(TrendingClaim).filter(
                    TrendingClaim.discovered_at >= week_ago,
                    TrendingClaim.status == 'completed'
                ).count()
                
                # Top categories
                top_categories = db.query(
                    TrendingClaim.category,
                    func.count(TrendingClaim.id).label('count')
                ).filter(
                    TrendingClaim.discovered_at >= week_ago
                ).group_by(TrendingClaim.category).order_by(
                    func.count(TrendingClaim.id).desc()
                ).limit(5).all()
                
                # Log weekly report
                logger.info(f"📊 Weekly Report:")
                logger.info(f"   Total claims discovered: {total_claims}")
                logger.info(f"   Claims processed: {processed_claims}")
                if total_claims > 0:
                    processing_rate = (processed_claims / total_claims) * 100
                    logger.info(f"   Processing rate: {processing_rate:.1f}%")
                
                logger.info(f"   Top categories:")
                for category, count in top_categories:
                    logger.info(f"     - {category}: {count} claims")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Weekly analytics failed: {e}")
    
    def get_status(self) -> dict:
        """Get scheduler status and job information"""
        if not self.is_running:
            return {"status": "stopped", "jobs": []}
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "status": "running",
            "jobs": jobs,
            "uptime": datetime.utcnow().isoformat()
        }
    
    async def trigger_job(self, job_id: str) -> bool:
        """Manually trigger a specific job"""
        if not self.is_running:
            return False
        
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.utcnow())
                logger.info(f"🎯 Manually triggered job: {job_id}")
                return True
            else:
                logger.warning(f"Job not found: {job_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to trigger job {job_id}: {e}")
            return False

# Global scheduler instance
_scheduler_instance: Optional[TrendingClaimsScheduler] = None

async def get_scheduler() -> TrendingClaimsScheduler:
    """Get the global scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TrendingClaimsScheduler()
    return _scheduler_instance

@asynccontextmanager
async def scheduler_lifespan():
    """Context manager for scheduler lifecycle"""
    scheduler = await get_scheduler()
    try:
        await scheduler.start()
        yield scheduler
    finally:
        await scheduler.stop()

# Convenience functions for FastAPI integration
async def start_background_scheduler():
    """Start the background scheduler (for FastAPI startup)"""
    scheduler = await get_scheduler()
    await scheduler.start()

async def stop_background_scheduler():
    """Stop the background scheduler (for FastAPI shutdown)"""
    global _scheduler_instance
    if _scheduler_instance:
        await _scheduler_instance.stop()
        _scheduler_instance = None

# Testing function
async def main():
    """Test the scheduler"""
    print("🧪 Testing Trending Claims Scheduler...")
    
    async with scheduler_lifespan() as scheduler:
        print(f"✅ Scheduler started")
        
        # Show status
        status = scheduler.get_status()
        print(f"📊 Status: {status['status']}")
        print(f"📋 Jobs scheduled: {len(status['jobs'])}")
        
        for job in status['jobs']:
            print(f"   - {job['name']}: {job['next_run']}")
        
        # Wait a bit to see if jobs work
        print("⏳ Waiting 10 seconds...")
        await asyncio.sleep(10)
        
        print("✅ Scheduler test complete")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())