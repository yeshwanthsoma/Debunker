#!/usr/bin/env python3
"""
Database Backlog Cleanup Script
Handles accumulated 'discovered' claims efficiently
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict

from database import SessionLocal, TrendingClaim
from fact_check_apis import MultiSourceFactChecker
from config import is_api_available
from sqlalchemy import func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class BacklogManager:
    """Manages and processes accumulated discovered claims"""
    
    def __init__(self):
        self.db = SessionLocal()
        self.fact_checker = None
        
    async def analyze_backlog(self) -> Dict:
        """Analyze the current backlog situation"""
        logger.info("🔍 Analyzing database backlog...")
        
        try:
            # Get status counts
            status_counts = self.db.query(
                TrendingClaim.status, 
                func.count(TrendingClaim.id)
            ).group_by(TrendingClaim.status).all()
            
            status_dict = {status: count for status, count in status_counts}
            discovered_count = status_dict.get('discovered', 0)
            
            # Get age distribution of discovered claims
            now = datetime.utcnow()
            age_ranges = {
                'last_24h': 0,
                'last_week': 0, 
                'last_month': 0,
                'older': 0
            }
            
            if discovered_count > 0:
                discovered_claims = self.db.query(TrendingClaim).filter(
                    TrendingClaim.status == 'discovered'
                ).all()
                
                for claim in discovered_claims:
                    age = now - claim.discovered_at
                    if age <= timedelta(days=1):
                        age_ranges['last_24h'] += 1
                    elif age <= timedelta(days=7):
                        age_ranges['last_week'] += 1
                    elif age <= timedelta(days=30):
                        age_ranges['last_month'] += 1
                    else:
                        age_ranges['older'] += 1
            
            analysis = {
                'total_claims': sum(status_dict.values()),
                'status_breakdown': status_dict,
                'discovered_count': discovered_count,
                'age_distribution': age_ranges
            }
            
            logger.info(f"📊 Backlog Analysis:")
            logger.info(f"   Total claims: {analysis['total_claims']}")
            logger.info(f"   Discovered (pending): {discovered_count}")
            logger.info(f"   Age distribution: {age_ranges}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing backlog: {e}")
            return {}
    
    async def cleanup_old_claims(self, days_threshold: int = 14) -> int:
        """Remove very old discovered claims that are likely stale"""
        logger.info(f"🧹 Cleaning up discovered claims older than {days_threshold} days...")
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # Count claims to be deleted
            old_claims_count = self.db.query(TrendingClaim).filter(
                TrendingClaim.status == 'discovered',
                TrendingClaim.discovered_at < cutoff_date
            ).count()
            
            if old_claims_count > 0:
                # Delete old claims
                deleted_count = self.db.query(TrendingClaim).filter(
                    TrendingClaim.status == 'discovered',
                    TrendingClaim.discovered_at < cutoff_date
                ).delete()
                
                self.db.commit()
                logger.info(f"✅ Deleted {deleted_count} old discovered claims")
                return deleted_count
            else:
                logger.info("ℹ️ No old claims to delete")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up old claims: {e}")
            self.db.rollback()
            return 0
    
    async def mark_low_priority_claims(self, limit: int = 100) -> int:
        """Mark low-priority claims as 'skipped' to reduce backlog"""
        logger.info(f"🏷️ Marking low-priority claims as skipped (limit: {limit})...")
        
        try:
            # Get low-priority claims (low trending score, older claims)
            low_priority_claims = self.db.query(TrendingClaim).filter(
                TrendingClaim.status == 'discovered',
                TrendingClaim.trending_score < 0.3  # Low viral potential
            ).order_by(TrendingClaim.discovered_at).limit(limit).all()
            
            marked_count = 0
            for claim in low_priority_claims:
                claim.status = 'skipped'
                claim.explanation = 'Marked as low priority during backlog cleanup'
                claim.processed_at = datetime.utcnow()
                marked_count += 1
            
            if marked_count > 0:
                self.db.commit()
                logger.info(f"✅ Marked {marked_count} low-priority claims as skipped")
            
            return marked_count
            
        except Exception as e:
            logger.error(f"❌ Error marking low-priority claims: {e}")
            self.db.rollback()
            return 0
    
    async def batch_process_high_priority(self, batch_size: int = 10) -> int:
        """Process a batch of high-priority discovered claims"""
        logger.info(f"🚀 Processing batch of high-priority claims (size: {batch_size})...")
        
        try:
            # Initialize fact checker if APIs available
            if not self.fact_checker and (
                is_api_available("openai") or 
                is_api_available("grok") or 
                is_api_available("google_fact_check")
            ):
                self.fact_checker = MultiSourceFactChecker()
                logger.info("✅ Professional fact-checker initialized")
            
            # Get high-priority claims
            high_priority_claims = self.db.query(TrendingClaim).filter(
                TrendingClaim.status == 'discovered',
                TrendingClaim.trending_score >= 0.5  # High viral potential
            ).order_by(
                TrendingClaim.trending_score.desc(),  # Highest priority first
                TrendingClaim.discovered_at.desc()     # Newest first
            ).limit(batch_size).all()
            
            if not high_priority_claims:
                logger.info("ℹ️ No high-priority claims found")
                return 0
            
            processed_count = 0
            
            for claim in high_priority_claims:
                try:
                    claim.status = 'processing'
                    self.db.commit()
                    
                    if self.fact_checker:
                        # Professional fact-checking
                        logger.info(f"🔍 Fact-checking: '{claim.claim_text[:50]}...'")
                        
                        result = await self.fact_checker.comprehensive_fact_check(
                            claim.claim_text,
                            claim.title or ""
                        )
                        
                        if isinstance(result, dict):
                            claim.verdict = result.get('verdict', 'Unverifiable')
                            claim.confidence = result.get('confidence', 0.5)
                            claim.explanation = result.get('explanation', 'Fact-check completed')
                            claim.status = 'completed'
                        else:
                            claim.status = 'failed'
                            claim.explanation = 'Unexpected result format'
                    else:
                        # Basic heuristic fact-checking
                        self._basic_fact_check(claim)
                    
                    claim.processed_at = datetime.utcnow()
                    processed_count += 1
                    
                    logger.info(f"✅ Processed claim {processed_count}/{len(high_priority_claims)}: {claim.verdict}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing claim {claim.id}: {e}")
                    claim.status = 'failed'
                    claim.explanation = f"Processing error: {str(e)}"
                    claim.processed_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"🎉 Batch processing complete: {processed_count} claims processed")
            return processed_count
            
        except Exception as e:
            logger.error(f"❌ Error in batch processing: {e}")
            self.db.rollback()
            return 0
    
    def _basic_fact_check(self, claim):
        """Basic heuristic fact-checking as fallback"""
        text_lower = claim.claim_text.lower()
        
        if any(word in text_lower for word in ['hoax', 'conspiracy', 'fake news', 'debunked']):
            claim.verdict = 'False'
            claim.confidence = 0.7
            claim.explanation = 'Contains misinformation indicators'
        elif any(word in text_lower for word in ['study', 'research', 'scientists', 'evidence']):
            claim.verdict = 'True'
            claim.confidence = 0.6
            claim.explanation = 'References research or evidence'
        else:
            claim.verdict = 'Unverifiable'
            claim.confidence = 0.5
            claim.explanation = 'Requires further verification'
        
        claim.status = 'completed'
    
    async def comprehensive_cleanup(self, 
                                  clean_old_days: int = 14,
                                  mark_low_priority_limit: int = 50,
                                  process_high_priority_batch: int = 20) -> Dict:
        """Run comprehensive backlog cleanup"""
        logger.info("🚀 Starting comprehensive backlog cleanup...")
        
        start_time = time.time()
        
        # Step 1: Analyze current situation
        analysis = await self.analyze_backlog()
        initial_discovered = analysis.get('discovered_count', 0)
        
        # Step 2: Clean up very old claims
        deleted_count = await self.cleanup_old_claims(clean_old_days)
        
        # Step 3: Mark low-priority claims as skipped
        skipped_count = await self.mark_low_priority_claims(mark_low_priority_limit)
        
        # Step 4: Process high-priority claims
        processed_count = await self.batch_process_high_priority(process_high_priority_batch)
        
        # Step 5: Final analysis
        final_analysis = await self.analyze_backlog()
        final_discovered = final_analysis.get('discovered_count', 0)
        
        execution_time = time.time() - start_time
        
        summary = {
            'initial_discovered': initial_discovered,
            'final_discovered': final_discovered,
            'deleted_old': deleted_count,
            'marked_skipped': skipped_count,
            'processed_claims': processed_count,
            'total_reduced': initial_discovered - final_discovered,
            'execution_time': execution_time
        }
        
        logger.info(f"🎯 Cleanup Summary:")
        logger.info(f"   Initial discovered claims: {initial_discovered}")
        logger.info(f"   Final discovered claims: {final_discovered}")
        logger.info(f"   Total reduction: {summary['total_reduced']} claims")
        logger.info(f"   Deleted old claims: {deleted_count}")
        logger.info(f"   Marked as skipped: {skipped_count}")
        logger.info(f"   Processed claims: {processed_count}")
        logger.info(f"   Execution time: {execution_time:.1f}s")
        
        return summary
    
    async def close(self):
        """Clean up resources"""
        if self.fact_checker:
            try:
                await self.fact_checker.close()
            except Exception as e:
                logger.warning(f"Error closing fact checker: {e}")
        
        if self.db:
            self.db.close()

async def main():
    """Main cleanup script"""
    print("🚀 Database Backlog Cleanup Tool")
    print("=" * 50)
    
    manager = BacklogManager()
    
    try:
        # Run comprehensive cleanup
        summary = await manager.comprehensive_cleanup(
            clean_old_days=14,      # Delete claims older than 2 weeks
            mark_low_priority_limit=100,  # Mark up to 100 low-priority as skipped
            process_high_priority_batch=20  # Process 20 high-priority claims
        )
        
        print("\n🎉 Cleanup completed successfully!")
        print(f"   Reduced backlog by {summary['total_reduced']} claims")
        print(f"   Remaining discovered claims: {summary['final_discovered']}")
        
    except Exception as e:
        logger.error(f"💥 Cleanup failed: {e}")
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(main())