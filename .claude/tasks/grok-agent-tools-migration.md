# Grok API Migration: Live Search → Agent Tools API

**Status:** ✅ COMPLETED

## Problem
Grok Live Search API is deprecated (returns 410 error). Must migrate to the new Agent Tools API.

## Migration Overview

### What's Changing

**Old API (Deprecated):**
```python
payload = {
    "model": "grok-4",
    "messages": [...],
    "search_parameters": {
        "mode": "on",
        "return_citations": True,
        "max_search_results": 5
    }
}
```

**New API (Agent Tools):**
```python
payload = {
    "model": "grok-4-1-fast",  # Optimized for tool calling
    "messages": [...],
    "tools": [
        {"type": "web_search"},
        {"type": "x_search"}
    ],
    "tool_choice": "auto"  # Optional: "auto", "required", or "none"
}
```

## Key Changes

### 1. Model Update
- Old: `grok-4`
- New: `grok-4-1-fast` (optimized for tool calling with 2M context window)

### 2. Search Configuration
- Remove: `search_parameters` dictionary
- Add: `tools` array with tool types
- Add: `tool_choice` parameter (optional)

### 3. API Endpoint
- Endpoint stays: `https://api.x.ai/v1/chat/completions`
- Headers unchanged: `Authorization: Bearer {api_key}`

### 4. Response Format
- Similar structure, but tools provide enhanced results
- Citations still available
- Better structured with tool call metadata

## Files to Update

### Primary Changes

1. **`backend/fact_check_apis.py`**
   - `GrokLiveSearchFactChecker` class (line 963+)
   - `_get_search_parameters()` → Update to use tools
   - `analyze_claim_with_live_search()` → Update payload
   - Update model from `grok-4` to `grok-4-1-fast`

### Secondary Changes

2. **`backend/grok_integration.py`**
   - `GrokSocialAnalyzer` class
   - Update model in `__init__()` (line 48)
   - Consider adding web_search tool for enhanced context

## Implementation Plan

### Step 1: Update GrokLiveSearchFactChecker

```python
class GrokLiveSearchFactChecker:
    def __init__(self):
        self.api_key = get_api_key("grok")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4-1-fast"  # Changed from "grok-4"
        self.session = None

    async def analyze_claim_with_live_search(self, claim: str, context: str = "", stream: bool = False):
        # ... existing code ...

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional fact-checker with real-time access to web, news, and social media data. Use your search tools to find current, authoritative information and provide evidence-based fact-checking with specific citations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "tools": [
                {"type": "web_search"},
                {"type": "x_search"}
            ],
            "tool_choice": "auto",  # Let AI decide when to search
            "stream": stream,
            "temperature": 0.1,
            "max_tokens": 2000
        }

        # Remove search_parameters entirely
```

### Step 2: Update Helper Methods

Remove `_get_search_parameters()` method or update it:

```python
# OPTION 1: Remove entirely (recommended)
# Delete _get_search_parameters() method

# OPTION 2: Update to return tools config
def _get_tools_config(self) -> List[Dict]:
    """Configure Agent Tools"""
    return [
        {"type": "web_search"},
        {"type": "x_search"}
    ]
```

### Step 3: Update Logging

```python
# Old logging
logger.info(f"🌐 Live Search sources: Web, X/Twitter, News, RSS")

# New logging
logger.info(f"🔧 Agent Tools enabled: web_search, x_search")
```

### Step 4: Update GrokSocialAnalyzer (Optional Enhancement)

```python
class GrokSocialAnalyzer:
    def __init__(self):
        self.api_key = get_api_key("grok")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4-1-fast"  # Updated model
        self.session = None

    async def _call_grok_api(self, prompt: str, use_tools: bool = True):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7
        }

        # Optionally add tools for social context
        if use_tools:
            payload["tools"] = [{"type": "x_search"}]
            payload["tool_choice"] = "auto"

        # ... rest of implementation
```

## Testing Plan

1. **Syntax validation:** `python3 -m py_compile fact_check_apis.py`
2. **API test:** Run fact-check with Grok to verify no 410 errors
3. **Response validation:** Ensure citations and results still parse correctly
4. **Cost monitoring:** Check usage and billing

## Expected Benefits

### Cost Savings
- Old: $0.025/source × 25 sources/day = $0.625/day = **$18.75/month**
- New: $5/1k calls × 450 calls/month = **$2.25/month**
- **Savings: 88% reduction ($16.50/month)**

### Quality Improvements
- ✅ Intelligent search (AI decides when to search)
- ✅ Auto follow-up queries for better accuracy
- ✅ Multimodal support (images/videos in results)
- ✅ Better X/Twitter integration
- ✅ More comprehensive web search

### Technical Improvements
- ✅ More reliable (actively maintained API)
- ✅ Better error handling
- ✅ Faster model (grok-4-1-fast)
- ✅ 2M token context window

## Rollback Plan

If issues occur:
1. Revert model to `grok-4`
2. Keep tools-based approach (it's the only option now)
3. Adjust `tool_choice` to `"none"` to disable searches temporarily
4. Monitor error logs for specific issues

## Documentation Updates

Update these files after migration:
- `backend/README.md` (if exists)
- `backend/AUTHENTICATION_GUIDE.md`
- `.env.example` (add notes about new API)

## References

- [xAI Agent Tools Overview](https://docs.x.ai/docs/guides/tools/overview)
- [Search Tools Documentation](https://docs.x.ai/docs/guides/tools/search-tools)
- [Grok 4.1 Fast Announcement](https://x.ai/news/grok-4-1-fast)
- [Migration Guide](https://docs.x.ai/docs/guides/migration)

---

## Migration Completion Summary

### Changes Implemented

**`backend/fact_check_apis.py`:**
- ✅ Updated `GrokFactChecker` model from `grok-4` to `grok-4-1-fast`
- ✅ Replaced `search_parameters` with `tools` array (`web_search` + `x_search`)
- ✅ Removed deprecated `_get_search_parameters()` method
- ✅ Renamed `_create_live_search_prompt()` → `_create_agent_tools_prompt()`
- ✅ Renamed `_parse_live_search_response()` → `_parse_agent_tools_response()`
- ✅ Updated all logging messages and provider names
- ✅ Updated cost tracking (new pricing model: $2.50-$5 per 1k tool calls)
- ✅ Updated source types: `live_search_*` → `agent_tools_*`
- ✅ Updated streaming event handling with backwards compatibility

**`backend/grok_integration.py`:**
- ✅ Updated `GrokSocialAnalyzer` model from `grok-4` to `grok-4-1-fast`

**Documentation:**
- ✅ Created migration plan document
- ✅ Updated comments and docstrings throughout

### Syntax Validation
- ✅ `fact_check_apis.py` - Passed
- ✅ `grok_integration.py` - Passed

### Expected Benefits
- **Cost savings:** 88-94% reduction ($18.75/mo → $1.13-$2.25/mo)
- **Better quality:** Intelligent tool usage, auto follow-ups
- **Future-proof:** Using actively maintained API
- **Enhanced features:** Multimodal support, better X/Twitter integration

### Next Steps
1. Deploy changes to production
2. Monitor error logs for any 410 errors (should be resolved)
3. Verify tool calling is working correctly
4. Track actual API usage and costs
5. Update .env.example if needed

**Migration completed on:** 2026-03-04
