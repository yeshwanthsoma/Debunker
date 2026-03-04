# Reduce API Scheduler Frequency to Once Daily

**Status:** ✅ COMPLETED

## Objective
Change all scheduler jobs that use API tokens to run once per day instead of multiple times per day to reduce API costs.

## Analysis of Current Jobs

### Jobs Using API Tokens (Need Daily Frequency)

1. **Quick News Check** (Currently: Every 4 hours)
   - Uses: NewsAggregator API calls
   - API tokens: News API, OpenAI, potentially others
   - Action: Change to once daily

2. **Full News Aggregation** (Currently: Every 5 hours)
   - Uses: NewsAggregator API calls with higher limits
   - API tokens: News API, OpenAI, potentially others
   - Action: Change to once daily

3. **Social Media Trending Check** (Currently: Every 4 hours)
   - Uses: Grok API or Reddit API
   - API tokens: Grok, Reddit client ID/secret
   - Action: Change to once daily

4. **Fact-Check Pending Claims** (Currently: Every 4 hours)
   - Uses: Google Fact Check, OpenAI, Anthropic, Grok APIs
   - API tokens: Multiple (Google, OpenAI, Anthropic, Grok)
   - Action: Change to once daily

### Jobs NOT Using API Tokens (Keep As-Is)

5. **Daily Maintenance** (Currently: Daily at 3 AM UTC)
   - Only database cleanup operations
   - No changes needed

6. **Weekly Analytics** (Currently: Weekly on Sundays at 4 AM UTC)
   - Only database queries
   - No changes needed

## Implementation Plan

### Changes to `backend/scheduler.py`

1. **Quick News Check** (Line 93-99)
   - Change from: `IntervalTrigger(hours=4)`
   - Change to: `CronTrigger(hour=6, minute=0)` (6 AM UTC daily)
   - Reasoning: Early morning to catch overnight news

2. **Full News Aggregation** (Line 104-110)
   - Change from: `IntervalTrigger(hours=5)`
   - Change to: `CronTrigger(hour=14, minute=0)` (2 PM UTC daily)
   - Reasoning: Afternoon to catch daytime news, different time than quick check

3. **Social Media Trending Check** (Line 113-120)
   - Change from: `IntervalTrigger(hours=4)`
   - Change to: `CronTrigger(hour=10, minute=0)` (10 AM UTC daily)
   - Reasoning: Mid-morning when social media activity is high

4. **Fact-Check Pending Claims** (Line 123-129)
   - Change from: `IntervalTrigger(hours=4)`
   - Change to: `CronTrigger(hour=18, minute=0)` (6 PM UTC daily)
   - Reasoning: Evening to process claims discovered during the day

### Timing Strategy

Spread the daily jobs throughout the day to:
- Avoid resource spikes
- Ensure continuous coverage
- Optimize API rate limits

**Daily Schedule:**
- 03:00 UTC - Daily Maintenance (database cleanup)
- 06:00 UTC - Quick News Check
- 10:00 UTC - Social Media Trending Check
- 14:00 UTC - Full News Aggregation
- 18:00 UTC - Fact-Check Pending Claims
- Sunday 04:00 UTC - Weekly Analytics

### Code Changes

```python
# Job 1: Quick news check - DAILY at 6 AM UTC
self.scheduler.add_job(
    func=self._quick_news_aggregation,
    trigger=CronTrigger(hour=6, minute=0),  # Changed from IntervalTrigger(hours=4)
    id='quick_news_check',
    name='Quick News Aggregation (Daily)',
    replace_existing=True
)

# Job 2: Full news aggregation - DAILY at 2 PM UTC
self.scheduler.add_job(
    func=self._full_news_aggregation,
    trigger=CronTrigger(hour=14, minute=0),  # Changed from IntervalTrigger(hours=5)
    id='full_news_aggregation',
    name='Full News Aggregation (Daily)',
    replace_existing=True
)

# Job 3: Social media trending check - DAILY at 10 AM UTC
if is_api_available("grok") or is_api_available("reddit_client_id"):
    self.scheduler.add_job(
        func=self._social_media_check,
        trigger=CronTrigger(hour=10, minute=0),  # Changed from IntervalTrigger(hours=4)
        id='social_media_check',
        name='Social Media Trending Check (Daily)',
        replace_existing=True
    )

# Job 4: Professional fact-check - DAILY at 6 PM UTC
self.scheduler.add_job(
    func=self._professional_fact_check_new_claims,
    trigger=CronTrigger(hour=18, minute=0),  # Changed from IntervalTrigger(hours=4)
    id='fact_check_pending',
    name='Fact-Check Pending Claims (Daily)',
    replace_existing=True
)
```

## Expected Impact

### Cost Reduction
- **Quick News Check**: 6 runs/day → 1 run/day (83% reduction)
- **Full News Aggregation**: ~5 runs/day → 1 run/day (80% reduction)
- **Social Media Check**: 6 runs/day → 1 run/day (83% reduction)
- **Fact-Check**: 6 runs/day → 1 run/day (83% reduction)

**Total API calls reduction: ~80-83% across all jobs**

### Trade-offs
- Slightly less real-time claim discovery
- Claims processed once daily instead of every 4-5 hours
- Still maintains continuous coverage throughout the day

### Benefits
- Significant API cost savings
- More predictable resource usage
- Easier monitoring and debugging
- Better alignment with daily news cycles

## Testing Plan

1. Update `scheduler.py` with new triggers
2. Test scheduler initialization
3. Verify job schedules are correct
4. Check logs to ensure jobs run at expected times
5. Monitor API usage to confirm reduction

## Rollback Plan

If issues arise, revert to interval-based triggers:
- Quick check: `IntervalTrigger(hours=4)`
- Full aggregation: `IntervalTrigger(hours=5)`
- Social media: `IntervalTrigger(hours=4)`
- Fact-check: `IntervalTrigger(hours=4)`

## Implementation Summary

✅ **Changes completed successfully**

All four API-dependent jobs have been updated in `backend/scheduler.py`:
- ✅ Quick News Check: Changed to daily at 6 AM UTC (line 95)
- ✅ Full News Aggregation: Changed to daily at 2 PM UTC (line 106)
- ✅ Social Media Check: Changed to daily at 10 AM UTC (line 116)
- ✅ Fact-Check Pending: Changed to daily at 6 PM UTC (line 125)

**New Daily Schedule:**
- 03:00 UTC - Daily Maintenance (no API calls)
- 06:00 UTC - Quick News Check
- 10:00 UTC - Social Media Trending Check
- 14:00 UTC - Full News Aggregation
- 18:00 UTC - Fact-Check Pending Claims
- Sunday 04:00 UTC - Weekly Analytics (no API calls)

**Syntax validation:** Passed ✅
