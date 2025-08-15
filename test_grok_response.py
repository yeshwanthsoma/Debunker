#!/usr/bin/env python3
"""
Test script to understand exact Grok API response structure
"""

import asyncio
import json
import sys
import os

# Add backend path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from fact_check_apis import GrokFactChecker

async def test_grok_response():
    """Test what Grok actually returns"""
    
    # Set the API key from environment
    if not os.environ.get('GROK_API_KEY'):
        print("❌ GROK_API_KEY environment variable not set")
        return
    
    print("🧪 Testing Grok API Response Structure...")
    print("=" * 60)
    
    try:
        # Initialize Grok checker
        grok_checker = GrokFactChecker()
        print("✅ Grok checker initialized")
        
        # Test with a simple claim
        test_claim = "The moon landing was faked by Hollywood"
        print(f"🔍 Testing claim: '{test_claim}'")
        print()
        
        # Call Grok API
        result = await grok_checker.analyze_claim(test_claim)
        
        print("📊 GROK RESPONSE ANALYSIS:")
        print("=" * 40)
        print(f"Type: {type(result)}")
        print(f"Provider: {result.provider}")
        print(f"Verdict: {result.verdict}")
        print(f"Confidence: {result.confidence}")
        print(f"Explanation: {result.explanation[:100]}...")
        print()
        
        print("🔍 SOURCES ANALYSIS:")
        print("=" * 30)
        print(f"Sources type: {type(result.sources)}")
        print(f"Sources count: {len(result.sources) if result.sources else 0}")
        
        if result.sources:
            for i, source in enumerate(result.sources):
                print(f"Source {i+1}:")
                print(f"  Type: {type(source)}")
                print(f"  Content: {source}")
                if isinstance(source, dict):
                    for key, value in source.items():
                        print(f"    {key}: {value}")
                print()
        
        # Test with MultiSourceFactChecker to see integration
        print("🔗 TESTING MULTI-SOURCE INTEGRATION:")
        print("=" * 40)
        
        from fact_check_apis import MultiSourceFactChecker
        
        multi_checker = MultiSourceFactChecker()
        multi_result = await multi_checker.comprehensive_fact_check(test_claim)
        
        print(f"Multi-source result type: {type(multi_result)}")
        print(f"Verdict: {multi_result.get('verdict')}")
        print(f"Confidence: {multi_result.get('confidence')}")
        print(f"Provider: {multi_result.get('provider')}")
        
        sources = multi_result.get('sources', [])
        print(f"\nMulti-source sources type: {type(sources)}")
        print(f"Multi-source sources count: {len(sources)}")
        
        if sources:
            for i, source in enumerate(sources):
                print(f"Multi-source {i+1}:")
                print(f"  Type: {type(source)}")
                print(f"  Content: {source}")
                print()
        
        # Close sessions
        await grok_checker.close()
        await multi_checker.close()
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_grok_response())