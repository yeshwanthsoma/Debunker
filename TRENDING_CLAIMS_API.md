# 🚀 Trending Claims API Documentation

**Updated: August 13, 2025**

This document provides complete API documentation for the trending claims dashboard feature. The system automatically discovers and fact-checks trending claims from multiple sources in real-time.

## 📋 Overview

The trending claims system provides:
- **Automatic Discovery**: Real-time claim extraction from News API, Reddit, and Grok (X/Twitter)
- **AI-Powered Fact-Checking**: GPT-4o analysis with confidence scoring
- **Social Context**: Grok integration for viral trends and social sentiment
- **Comprehensive Categories**: Health, Politics, Science, Technology, Environment, and more
- **Real-Time Updates**: Background aggregation every 15 minutes

## 🔗 Base URLs
```bash
http://localhost:8080        # Local Development
https://your-app.railway.app # Production (Railway)
```

## 🔐 Authentication
- **Read Operations**: No authentication required
- **Write Operations**: Rate limited (5 requests/minute for triggers)
- **CORS**: Enabled for all origins

---

## 🔄 Data Sources & Updates

### Automatic Discovery Sources
- **📰 News API**: Top headlines from major news outlets (5 categories)
- **🔸 Reddit**: Trending discussions from news and science subreddits  
- **🌐 Grok (X.AI)**: Real-time X/Twitter trending misinformation detection

### Update Schedule
- **Startup**: Immediate discovery when server starts
- **Quick Check**: Every 15 minutes (10-20 new claims)
- **Full Aggregation**: Every 2 hours (comprehensive sweep)
- **Social Media**: Every hour (trending viral content)
- **Maintenance**: Daily cleanup at 3 AM UTC

### Fact-Checking Process
1. **Discovery**: Claims extracted using GPT-4o
2. **Scoring**: Controversy/trending score calculated
3. **Fact-Check**: Professional APIs (Google, OpenAI, Anthropic) or simple heuristics
4. **Classification**: Verdicts: `True`, `False`, `Mixed`, `Unverifiable`
5. **Completion**: Available via API with confidence scores

---

## 📋 Core API Endpoints

### 1. Get Trending Claims List

**Endpoint:** `GET /api/trending-claims`

**Description:** Get paginated list of trending claims with filtering options

**Query Parameters:**
- `page` (int, optional): Page number (default: 1)
- `limit` (int, optional): Items per page (default: 20, max: 100)
- `category` (string, optional): Filter by category (Health, Politics, Science, Technology, Environment, etc.)
- `status` (string, optional): Filter by status
  - `completed` - Fact-checked claims (recommended for production)
  - `discovered` - Newly found claims awaiting fact-check
  - `processing` - Currently being fact-checked
  - `failed` - Processing failed

**Example Requests:**
```bash
# Get latest completed fact-checks
GET /api/trending-claims?status=completed&limit=12

# Get health-related claims
GET /api/trending-claims?category=Health&status=completed

# Get page 2 with custom limit
GET /api/trending-claims?page=2&limit=8
```

**Response:**
```json
{
  "claims": [
    {
      "id": 1,
      "claim_text": "Vaccines cause autism in children",
      "title": "Study Links Vaccines to Autism - Health Officials Respond",
      "category": "Health",
      "verdict": "False",
      "confidence": 0.95,
      "explanation": "Multiple large-scale studies have found no link...",
      "source_type": "news",
      "trending_score": 0.87,
      "view_count": 1250,
      "share_count": 45,
      "status": "completed",
      "discovered_at": "2025-08-12T10:30:00Z",
      "processed_at": "2025-08-12T10:35:00Z",
      "tags": ["vaccine", "autism", "health", "study"]
    }
  ],
  "total": 156,
  "page": 1,
  "limit": 10,
  "categories": ["Health", "Politics", "Science", "Environment"]
}
```

---

### 2. Get Claim Details

**Endpoint:** `GET /api/trending-claims/{claim_id}`

**Description:** Get detailed information about a specific claim

**Path Parameters:**
- `claim_id` (int): Unique claim identifier

**Example Request:**
```bash
GET /api/trending-claims/1
```

**Response:**
```json
{
  "id": 1,
  "claim_text": "Vaccines cause autism in children",
  "title": "Study Links Vaccines to Autism - Health Officials Respond",
  "category": "Health",
  "verdict": "False",
  "confidence": 0.95,
  "explanation": "Multiple large-scale scientific studies involving millions of children...",
  "evidence_summary": "The claim has been thoroughly debunked by medical professionals...",
  "source_type": "news",
  "source_url": "https://example-news.com/article",
  "trending_score": 0.87,
  "controversy_level": 0.92,
  "view_count": 1251,
  "share_count": 45,
  "status": "completed",
  "processing_time": 12.34,
  "discovered_at": "2025-08-12T10:30:00Z",
  "processed_at": "2025-08-12T10:35:00Z",
  "tags": ["vaccine", "autism", "health"],
  "keywords": ["vaccine", "autism", "children", "study"],
  "related_entities": ["CDC", "WHO", "autism research"],
  "sources": [
    {
      "name": "CNN Health",
      "url": "https://cnn.com/health/article",
      "type": "news",
      "author": "Dr. Jane Smith",
      "published_at": "2025-08-12T09:00:00Z",
      "reliability": 0.85
    }
  ]
}
```

---

### 3. Trigger News Aggregation

**Endpoint:** `POST /api/trigger-aggregation`

**Description:** Manually trigger news aggregation (rate limited: 5/minute)

**Example Request:**
```bash
POST /api/trigger-aggregation
```

**Response:**
```json
{
  "message": "News aggregation completed successfully",
  "claims_discovered": 23,
  "claims_saved": 15,
  "execution_time": 45.67
}
```

---

### 4. Get Categories with Statistics

**Endpoint:** `GET /api/claims/categories`

**Description:** Get available categories with statistics

**Response:**
```json
{
  "categories": [
    {
      "category": "Health",
      "total_claims": 45,
      "processed_claims": 38,
      "avg_controversy": 0.73,
      "most_recent": "2025-08-12T10:30:00Z"
    },
    {
      "category": "Politics",
      "total_claims": 62,
      "processed_claims": 55,
      "avg_controversy": 0.89,
      "most_recent": "2025-08-12T11:15:00Z"
    }
  ],
  "total_categories": 6
}
```

---

### 5. Get Analytics

**Endpoint:** `GET /api/claims/analytics`

**Description:** Get platform analytics

**Query Parameters:**
- `days` (int, optional): Number of days to analyze (default: 7)

**Example Request:**
```bash
GET /api/claims/analytics?days=30
```

**Response:**
```json
{
  "period_days": 30,
  "total_claims": 234,
  "new_claims": 45,
  "processed_claims": 189,
  "processing_rate": 80.8,
  "avg_trending_score": 0.65,
  "top_categories": [
    {"category": "Politics", "count": 78},
    {"category": "Health", "count": 45},
    {"category": "Science", "count": 32}
  ]
}
```

---

### 6. Increment Share Count

**Endpoint:** `POST /api/claims/{claim_id}/share`

**Description:** Increment share count when user shares a claim

**Path Parameters:**
- `claim_id` (int): Claim identifier

**Response:**
```json
{
  "message": "Share count updated",
  "new_count": 46
}
```

---

## 🎨 Frontend Implementation Guide

### Landing Page Layout

```typescript
interface TrendingClaim {
  id: number;
  claim_text: string;
  title: string;
  category: string;
  verdict: string | null;
  confidence: number | null;
  explanation: string | null;
  source_type: string;
  trending_score: number;
  view_count: number;
  share_count: number;
  status: string;
  discovered_at: string;
  processed_at: string | null;
  tags: string[] | null;
}

interface TrendingClaimsResponse {
  claims: TrendingClaim[];
  total: number;
  page: number;
  limit: number;
  categories: string[];
}
```

### Example React Component

```jsx
import React, { useState, useEffect } from 'react';

const TrendingClaimsDashboard = () => {
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetchTrendingClaims();
  }, [page, selectedCategory]);

  const fetchTrendingClaims = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: '20',
        ...(selectedCategory && { category: selectedCategory })
      });
      
      const response = await fetch(`/api/trending-claims?${params}`);
      const data = await response.json();
      setClaims(data.claims);
    } catch (error) {
      console.error('Error fetching claims:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClaimClick = (claimId) => {
    // Navigate to detail page or show modal
    window.location.href = `/claim/${claimId}`;
  };

  const handleShare = async (claimId) => {
    try {
      await fetch(`/api/claims/${claimId}/share`, { method: 'POST' });
      // Update local share count or refetch
    } catch (error) {
      console.error('Error updating share count:', error);
    }
  };

  return (
    <div className="trending-claims-dashboard">
      <h1>Trending Fact-Checks</h1>
      
      {/* Category Filter */}
      <select 
        value={selectedCategory} 
        onChange={(e) => setSelectedCategory(e.target.value)}
      >
        <option value="">All Categories</option>
        <option value="Health">Health</option>
        <option value="Politics">Politics</option>
        <option value="Science">Science</option>
      </select>

      {/* Claims Grid */}
      <div className="claims-grid">
        {claims.map(claim => (
          <div 
            key={claim.id} 
            className="claim-card"
            onClick={() => handleClaimClick(claim.id)}
          >
            <div className={`verdict-badge ${claim.verdict?.toLowerCase()}`}>
              {claim.verdict || 'Processing...'}
            </div>
            
            <h3>{claim.title}</h3>
            <p className="claim-text">{claim.claim_text}</p>
            
            <div className="claim-meta">
              <span className="category">{claim.category}</span>
              <span className="trending-score">
                🔥 {(claim.trending_score * 100).toFixed(0)}%
              </span>
            </div>
            
            <div className="claim-stats">
              <span>👁️ {claim.view_count}</span>
              <button onClick={(e) => {
                e.stopPropagation();
                handleShare(claim.id);
              }}>
                📤 {claim.share_count}
              </button>
            </div>
          </div>
        ))}
      </div>
      
      {/* Pagination */}
      <div className="pagination">
        <button 
          disabled={page === 1}
          onClick={() => setPage(page - 1)}
        >
          Previous
        </button>
        <span>Page {page}</span>
        <button onClick={() => setPage(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
};

export default TrendingClaimsDashboard;
```

### CSS Styling Example

```css
.trending-claims-dashboard {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.claims-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.claim-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  cursor: pointer;
  transition: box-shadow 0.2s;
  position: relative;
}

.claim-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.verdict-badge {
  position: absolute;
  top: 10px;
  right: 10px;
  padding: 4px 8px;
  border-radius: 4px;
  color: white;
  font-size: 12px;
  font-weight: bold;
}

.verdict-badge.true { background: #22c55e; }
.verdict-badge.false { background: #ef4444; }
.verdict-badge.mixed { background: #f59e0b; }
.verdict-badge.unverifiable { background: #6b7280; }

.claim-text {
  color: #666;
  font-size: 14px;
  line-height: 1.4;
  margin: 10px 0;
}

.claim-meta {
  display: flex;
  justify-content: space-between;
  margin: 10px 0;
  font-size: 12px;
}

.claim-stats {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  font-size: 12px;
}
```

---

## ⏰ Scheduler Management Endpoints

### 1. Get Scheduler Status

**Endpoint:** `GET /api/scheduler/status`

**Description:** Get background scheduler status and job information

**Response:**
```json
{
  "status": "running",
  "jobs": [
    {
      "id": "quick_news_check",
      "name": "Quick News Aggregation", 
      "next_run": "2025-08-13T02:31:46-04:00",
      "trigger": "interval[0:15:00]"
    }
  ],
  "uptime": "2025-08-13T06:14:16.818568",
  "enabled": true
}
```

### 2. Trigger Manual Discovery

**Endpoint:** `POST /api/scheduler/trigger/{job_id}`

**Description:** Manually trigger news discovery (rate limited: 5/minute)

**Available Job IDs:**
- `quick_news_check` - Discover 10-20 new claims
- `full_news_aggregation` - Comprehensive discovery (50+ claims)
- `social_media_check` - Reddit + Grok trending check

**Example:**
```bash
POST /api/scheduler/trigger/quick_news_check
```

**Response:**
```json
{
  "message": "Job 'quick_news_check' triggered successfully",
  "job_id": "quick_news_check", 
  "triggered_at": "2025-08-13T06:18:43.215791"
}
```

### 3. List All Jobs

**Endpoint:** `GET /api/scheduler/jobs`

**Response:**
```json
{
  "jobs": [
    {
      "id": "quick_news_check",
      "name": "Quick News Aggregation",
      "next_run": "2025-08-13T02:31:46-04:00",
      "trigger": "interval[0:15:00]"
    }
  ],
  "total_jobs": 5,
  "scheduler_running": true
}
```

---

## 🔄 Real-time Updates

For real-time updates, consider implementing WebSocket connections or polling:

```javascript
// Polling for new claims every 5 minutes
setInterval(() => {
  fetchTrendingClaims();
}, 5 * 60 * 1000);

// Or use WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/trending-claims');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // Handle real-time updates
};
```

---

## 🚀 Deployment & Production Setup

### Required Environment Variables
```bash
# Core APIs
OPENAI_API_KEY=sk-proj-...           # GPT-4o for claim extraction
NEWS_API_KEY=6809b08bd6f...          # NewsAPI for headlines
GOOGLE_FACT_CHECK_API_KEY=AIzaSy...  # Google Fact Check API
ANTHROPIC_API_KEY=sk-ant-api03-...   # Claude for fact-checking

# Social Media Sources  
REDDIT_CLIENT_ID=lAJnxdE6...         # Reddit API
REDDIT_SECRET=afcDwRDy...            # Reddit API Secret
GROK_API_KEY=xai-UWUtKS...           # X.AI Grok for social context

# Optional
HUGGINGFACEHUB_API_TOKEN=hf_...      # HuggingFace models
```

### Production Checklist
- ✅ **All API Keys**: Configured in Railway/hosting environment
- ✅ **Database**: SQLite (dev) / PostgreSQL (production) 
- ✅ **Port**: Server runs on port 8080
- ✅ **CORS**: Configured for your frontend domain
- ✅ **Rate Limiting**: 5 requests/minute for manual triggers
- ✅ **Auto-Discovery**: Starts immediately on server boot
- ✅ **Background Jobs**: Run every 15 minutes automatically

### Performance Notes
- **Startup Time**: ~30 seconds (model loading + first discovery)
- **Discovery Speed**: 15-30 seconds for 10-20 claims
- **API Calls**: Optimized to ~35 OpenAI calls per discovery cycle
- **Storage**: ~1MB per 100 claims with full metadata

---

## 📈 Analytics Integration

Track user interactions for insights:

```javascript
// Track claim views
analytics.track('Claim Viewed', {
  claim_id: claimId,
  category: claim.category,
  verdict: claim.verdict
});

// Track shares
analytics.track('Claim Shared', {
  claim_id: claimId,
  platform: 'twitter'
});
```

---

## 🔧 Troubleshooting

### Common Issues

**Q: API returns `{"claims": [], "total": 0}`**
- ✅ Check if using `status=completed` (recommended)
- ✅ Wait 2-3 minutes after server startup for first discovery
- ✅ Trigger manual discovery: `POST /api/scheduler/trigger/quick_news_check`
- ✅ Check scheduler status: `GET /api/scheduler/status`

**Q: Claims not updating in real-time**
- ✅ Background jobs run every 15 minutes automatically
- ✅ Use manual triggers for immediate updates
- ✅ Implement polling every 5 minutes on frontend

**Q: Server startup takes long time**
- ✅ Normal behavior (30 seconds for model loading + discovery)
- ✅ Check logs for "✅ Startup complete: X claims discovered"

**Q: Rate limiting errors**
- ✅ Manual triggers limited to 5/minute
- ✅ Use automatic discovery instead of frequent manual triggers

### API Status Endpoints
```bash
# Check if backend is healthy
GET /health

# Check scheduler status  
GET /api/scheduler/status

# Check available categories
GET /api/claims/categories

# Get system analytics
GET /api/claims/analytics
```

---

## 📞 Support

For technical issues:
1. Check server logs for error messages
2. Verify all environment variables are set
3. Test individual endpoints using curl/Postman
4. Monitor scheduler status for background job health

This comprehensive API provides everything needed to build an engaging, real-time trending claims dashboard that automatically discovers and fact-checks viral misinformation! 🚀