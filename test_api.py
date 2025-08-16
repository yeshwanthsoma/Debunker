#!/usr/bin/env python3
"""
TruthLens API Test Script
Tests the basic functionality of the fact-checking API
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status():
    """Test API status endpoint"""
    print("\nTesting API status...")
    try:
        response = requests.get(f"{API_BASE}/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Version: {data['version']}")
            
            apis = data['professional_apis']
            available_apis = [name for name, available in apis.items() if available]
            
            if available_apis:
                print(f"✅ Professional APIs available: {', '.join(available_apis)}")
            else:
                print("ℹ️  No professional APIs configured (using basic mode)")
                
            if data['recommendations']:
                print("💡 Recommendations:")
                for rec in data['recommendations']:
                    print(f"   - {rec}")
            
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_fact_check(claim, expected_verdict=None):
    """Test fact-checking endpoint"""
    print(f"\nTesting fact-check: '{claim[:50]}...'")
    try:
        payload = {
            'text_claim': claim,
            'enable_prosody': False
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/analyze",
            json=payload,
            timeout=30
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Verdict: {data['verdict']}")
            print(f"✅ Confidence: {data['confidence']}")
            print(f"✅ Processing time: {processing_time:.2f}s")
            
            if expected_verdict and data['verdict'] != expected_verdict:
                print(f"⚠️  Expected {expected_verdict}, got {data['verdict']}")
                return False
            
            return True
        else:
            print(f"❌ Fact-check failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Fact-check error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TruthLens API Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health check
    total_tests += 1
    if test_health():
        tests_passed += 1
    
    # Test 2: API status
    total_tests += 1
    if test_status():
        tests_passed += 1
    
    # Test 3: Known conspiracy theory (should be False)
    total_tests += 1
    if test_fact_check("The moon landing was faked by NASA", "False"):
        tests_passed += 1
    
    # Test 4: Another known conspiracy theory
    total_tests += 1
    if test_fact_check("Vaccines cause autism", "False"):
        tests_passed += 1
    
    # Test 5: Climate change (should be True or supported)
    total_tests += 1
    if test_fact_check("Climate change is real and caused by human activities"):
        tests_passed += 1
    
    # Test 6: Generic claim (will depend on API availability)
    total_tests += 1
    if test_fact_check("The Earth is round"):
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! TruthLens is working correctly.")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)