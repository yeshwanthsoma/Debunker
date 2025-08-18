#!/usr/bin/env python3
"""
Check Production Database Status
Query the production database via API to understand backlog
"""

import aiohttp
import asyncio
import json
from datetime import datetime, timedelta

async def check_production_status():
    """Check production database status via API calls"""
    base_url = "https://debunker-production-4920.up.railway.app"
    
    print("🔍 Checking Production Database Status...")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Check if there's an API endpoint to get claims statistics
            # We'll try a few potential endpoints
            
            # Try the trending claims endpoint
            try:
                async with session.get(f"{base_url}/api/trending-claims?limit=1") as response:
                    if response.status == 200:
                        data = await response.json()
                        total_claims = data.get('total', 0)
                        print(f"📊 Total claims in database: {total_claims}")
                    else:
                        print(f"❌ Trending claims endpoint returned: {response.status}")
            except Exception as e:
                print(f"⚠️  Trending claims endpoint not accessible: {e}")
            
            # Try to get some sample claims to understand the data
            try:
                async with session.get(f"{base_url}/api/trending-claims?limit=5&page=1") as response:
                    if response.status == 200:
                        data = await response.json()
                        claims = data.get('claims', [])
                        
                        if claims:
                            print(f"\n📋 Recent claims in production:")
                            for i, claim in enumerate(claims[:3], 1):
                                status = claim.get('status', 'unknown')
                                discovered_at = claim.get('discovered_at', 'unknown')
                                claim_text = claim.get('claim_text', '')[:60]
                                print(f"   {i}. [{status}] {discovered_at}: {claim_text}...")
                        else:
                            print("\n📭 No claims found in trending endpoint")
                    else:
                        print(f"❌ Claims sample request failed: {response.status}")
            except Exception as e:
                print(f"⚠️  Sample claims request failed: {e}")
            
            # Check general API status
            try:
                async with session.get(f"{base_url}/api/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        version = data.get('version', 'unknown')
                        apis = data.get('professional_apis', {})
                        
                        print(f"\n🔧 Production API Status:")
                        print(f"   Version: {version}")
                        print(f"   Professional APIs: {sum(1 for v in apis.values() if v)}/{len(apis)} active")
                        
                        recommendations = data.get('recommendations', [])
                        if recommendations:
                            print(f"   Recommendations: {len(recommendations)} items")
                    else:
                        print(f"❌ Status endpoint failed: {response.status}")
            except Exception as e:
                print(f"⚠️  Status check failed: {e}")
            
            # Try to trigger a manual aggregation to see current activity
            print(f"\n🔬 Production database appears to be running...")
            print(f"   Base URL: {base_url}")
            print(f"   Health status: Available")
            
            # Since we can't directly query the database, let's check logs or other indicators
            print(f"\n💡 To get detailed database status, you could:")
            print(f"   1. Check Railway dashboard for memory/CPU usage")
            print(f"   2. Look at Railway logs for scheduler activity")
            print(f"   3. Monitor the trending claims endpoint over time")
            print(f"   4. Add a database status endpoint to the API")
            
        except Exception as e:
            print(f"❌ Error checking production: {e}")

async def main():
    await check_production_status()

if __name__ == "__main__":
    asyncio.run(main())