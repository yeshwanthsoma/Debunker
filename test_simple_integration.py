#!/usr/bin/env python3
"""
Simple test for multi-source fact-checking with real Grok
"""

import asyncio
import json
import sys
import os

# Add backend path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_simple_integration():
    """Test simple multi-source integration"""
    
    # Check for API key in environment
    if not os.environ.get('GROK_API_KEY'):
        print("❌ GROK_API_KEY environment variable not set")
        return
    
    print("🧪 Testing Simple Multi-Source Integration...")
    print("=" * 50)
    
    try:
        from fact_check_apis import MultiSourceFactChecker
        
        # Test different claims
        test_claims = [
            "Vaccines are safe and effective",
            "Water boils at 100 degrees Celsius at sea level",
            "The Earth is flat"
        ]
        
        checker = MultiSourceFactChecker()
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n🔍 Test {i}: '{claim}'")
            print("-" * 40)
            
            try:
                result = await checker.comprehensive_fact_check(claim)
                
                print(f"✅ SUCCESS")
                print(f"   Verdict: {result.get('verdict')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Sources: {len(result.get('sources', []))}")
                print(f"   Provider: {result.get('provider')}")
                
                # Check details
                details = result.get('details', {})
                print(f"   Grok results: {details.get('grok_results', 0)}")
                print(f"   OpenAI results: {details.get('openai_results', 0)}")
                print(f"   Google results: {details.get('google_results', 0)}")
                
            except Exception as claim_error:
                print(f"❌ FAILED: {claim_error}")
                continue
        
        await checker.close()
        print(f"\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_integration())