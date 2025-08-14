# 🔄 Trending Claims Sorting Options

## Problem Solved
The trending claims API was showing the same 15 claims every time because it only sorted by `trending_score`. Now it includes time-based sorting for fresh, rotating content.

## 📡 Updated API Endpoint

### GET /api/trending-claims

**New Query Parameter:**
```
?sort={option}
```

## 🎛️ Available Sorting Options

### 1. `hybrid` (Default - Recommended)
- **Formula**: `trending_score + recency_boost`
- **Recency Boost**: +0.3 for last 24h, +0.1 for last 7 days
- **Best for**: Landing page - balances quality with freshness
- **Result**: Fresh claims appear at top, high-quality older claims still visible

### 2. `recent` 
- **Sorts by**: Most recently discovered claims first
- **Best for**: "What's New" sections
- **Result**: Latest breaking claims at top

### 3. `trending`
- **Sorts by**: Pure trending score (original behavior)  
- **Best for**: "Most Controversial" or "Top Trending" sections
- **Result**: Highest scoring claims regardless of age

### 4. `processed`
- **Sorts by**: Most recently fact-checked claims first
- **Best for**: "Latest Fact-Checks" sections
- **Result**: Claims with fresh verification results

### 5. `popular`
- **Formula**: `(view_count + share_count * 5) + trending_score`
- **Best for**: "Most Shared" or viral content sections
- **Result**: Claims with high user engagement

## 📱 Frontend Implementation Examples

### Basic Usage
```javascript
// Default (hybrid) - best for landing page
const claims = await fetch('/api/trending-claims?limit=15');

// Fresh claims for "Latest" tab  
const latestClaims = await fetch('/api/trending-claims?sort=recent&limit=10');

// Popular claims for "Trending" tab
const trendingClaims = await fetch('/api/trending-claims?sort=trending&limit=10');
```

### React Component with Tabs
```jsx
const ClaimsDashboard = () => {
  const [sortMode, setSortMode] = useState('hybrid');
  const [claims, setClaims] = useState([]);

  const sortOptions = [
    { value: 'hybrid', label: '🔥 For You', desc: 'Fresh + trending' },
    { value: 'recent', label: '🆕 Latest', desc: 'Just discovered' },
    { value: 'trending', label: '📈 Top Trending', desc: 'Most controversial' },
    { value: 'processed', label: '✅ Recently Checked', desc: 'Fresh fact-checks' },
    { value: 'popular', label: '👥 Most Shared', desc: 'High engagement' }
  ];

  useEffect(() => {
    fetchClaims();
  }, [sortMode]);

  const fetchClaims = async () => {
    const response = await fetch(`/api/trending-claims?sort=${sortMode}&limit=15`);
    const data = await response.json();
    setClaims(data.claims);
  };

  return (
    <div>
      {/* Sort Tabs */}
      <div className="sort-tabs">
        {sortOptions.map(option => (
          <button
            key={option.value}
            onClick={() => setSortMode(option.value)}
            className={sortMode === option.value ? 'active' : ''}
          >
            {option.label}
            <small>{option.desc}</small>
          </button>
        ))}
      </div>

      {/* Claims Grid */}
      <div className="claims-grid">
        {claims.map(claim => (
          <ClaimCard key={claim.id} claim={claim} />
        ))}
      </div>
    </div>
  );
};
```

### Advanced Filtering + Sorting
```javascript
const fetchFilteredClaims = async (filters) => {
  const params = new URLSearchParams({
    sort: filters.sort || 'hybrid',
    category: filters.category || '',
    status: filters.status || 'completed',
    page: filters.page || 1,
    limit: filters.limit || 15
  });

  const response = await fetch(`/api/trending-claims?${params}`);
  return response.json();
};

// Usage examples
const forYouClaims = await fetchFilteredClaims({ sort: 'hybrid' });
const healthClaims = await fetchFilteredClaims({ category: 'Health', sort: 'recent' });
const viralClaims = await fetchFilteredClaims({ sort: 'popular', limit: 5 });
```

## 🎯 Recommended Usage

### Landing Page Strategy
```javascript
// Use different sorts for different sections
const sections = {
  hero: { sort: 'hybrid', limit: 6 },      // Best overall mix
  latest: { sort: 'recent', limit: 4 },     // Fresh content
  trending: { sort: 'trending', limit: 4 }, // High controversy  
  popular: { sort: 'popular', limit: 4 }    // Most shared
};
```

### User Preferences
```javascript
// Let users choose their default sort
const userPreferences = {
  defaultSort: localStorage.getItem('preferredSort') || 'hybrid',
  refreshInterval: 5 * 60 * 1000 // 5 minutes
};

// Auto-refresh with user's preferred sort
setInterval(() => {
  fetchClaims(userPreferences.defaultSort);
}, userPreferences.refreshInterval);
```

## 🔄 Content Rotation Benefits

### Before (trending only):
- Same 15 claims every day
- Stale content for returning users
- No discovery of fresh claims

### After (with sort options):
- ✅ Daily content rotation with `hybrid`
- ✅ Fresh discoveries with `recent` 
- ✅ High-quality trending with `trending`
- ✅ Community favorites with `popular`
- ✅ Latest fact-checks with `processed`

## 📊 Expected Behavior

**`?sort=hybrid` (Default):**
```
Today:    New Claim A (0.7 + 0.3 boost) = 1.0  ← TOP
          Old Viral Claim (0.9 + 0.0) = 0.9     ← LOWER  
Tomorrow: Newer Claim B (0.6 + 0.3) = 0.9      ← NEW TOP
```

**`?sort=recent`:**
```
2025-08-14 10:00 AM claims → TOP
2025-08-14 09:30 AM claims → MIDDLE  
2025-08-13 claims → BOTTOM
```

This gives users control over their content experience while ensuring fresh, engaging trending claims! 🚀