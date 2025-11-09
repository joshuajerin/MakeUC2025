#!/usr/bin/env python3
"""
Quick test script to validate EmpathLens setup.
Run this after starting the service to ensure everything works.
"""

import httpx
import json
import sys


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = httpx.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Components: {json.dumps(data['components'], indent=2)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to service: {e}")
        print("   Make sure the service is running: python main.py")
        return False


def test_panic_scenario():
    """Test panic attack scenario"""
    print("\nTesting panic attack scenario...")
    try:
        response = httpx.post(
            f"{BASE_URL}/distress/infer",
            json={
                "message": "I'm having a panic attack, I can't breathe",
                "chat_id": "test_panic_123"
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Panic detection successful")
            print(f"   State: {data['meta']['state']}")
            print(f"   Confidence: {data['meta']['confidence']:.2f}")
            print(f"   Reply: {data['reply_text']}")
            print(f"   Intervention: {data['meta']['intervention_type']}")
            return True
        else:
            print(f"❌ Panic detection failed: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_overwhelmed_scenario():
    """Test overwhelmed scenario"""
    print("\nTesting overwhelmed scenario...")
    try:
        response = httpx.post(
            f"{BASE_URL}/distress/infer",
            json={
                "message": "It's too much, I'm shutting down",
                "chat_id": "test_overwhelmed_123"
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Overwhelm detection successful")
            print(f"   State: {data['meta']['state']}")
            print(f"   Reply: {data['reply_text']}")
            return True
        else:
            print(f"❌ Overwhelm detection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_crisis_scenario():
    """Test crisis keyword detection"""
    print("\nTesting crisis keyword detection...")
    try:
        response = httpx.post(
            f"{BASE_URL}/distress/infer",
            json={
                "message": "I don't want to live anymore",
                "chat_id": "test_crisis_123"
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Crisis detection successful")
            print(f"   State: {data['meta']['state']}")
            print(f"   Reply: {data['reply_text']}")
            
            if data['meta']['state'] == 'crisis_risk':
                print(f"   ✅ Correctly identified as crisis")
                return True
            else:
                print(f"   ⚠️  State should be 'crisis_risk', got '{data['meta']['state']}'")
                return False
        else:
            print(f"❌ Crisis detection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_stop_command():
    """Test stop command"""
    print("\nTesting stop command...")
    try:
        response = httpx.post(
            f"{BASE_URL}/distress/infer",
            json={
                "message": "stop",
                "chat_id": "test_stop_123"
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stop command successful")
            print(f"   Reply: {data['reply_text']}")
            return True
        else:
            print(f"❌ Stop command failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("EmpathLens Service Test Suite")
    print("=" * 60)
    
    tests = [
        test_health,
        test_panic_scenario,
        test_overwhelmed_scenario,
        test_crisis_scenario,
        test_stop_command,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All tests passed! EmpathLens is ready to use.")
        return 0
    else:
        print(f"❌ {failed} test(s) failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

