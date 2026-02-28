#!/usr/bin/env python3
"""
Test the FastAPI endpoints
"""

import sys
import requests
import json
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n" + "=" * 80)
    print("Testing Health Check Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print("✗ Health check failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_recommend_endpoint():
    """Test the recommendation endpoint"""
    print("\n" + "=" * 80)
    print("Testing Recommendation Endpoint")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            "name": "Java Developer Query",
            "payload": {
                "query": "I am hiring for Java developers who can also collaborate effectively with my business team",
                "top_k": 5,
                "balance_skills": True,
                "include_explanation": False
            }
        },
        {
            "name": "Sales Role Query",
            "payload": {
                "query": "I want to hire new graduates for a sales role who are good at communication",
                "top_k": 10,
                "balance_skills": True,
                "include_explanation": True
            }
        },
        {
            "name": "Python Developer Query",
            "payload": {
                "query": "Looking for Python developers with strong problem-solving skills",
                "top_k": 8,
                "balance_skills": True
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 80)
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/recommend",
                json=test_case['payload'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"Query: {data['query'][:60]}...")
                print(f"Total Found: {data['total_found']}")
                print(f"Returned: {data['returned']}")
                print(f"\nTop {min(3, len(data['recommendations']))} Recommendations:")
                
                for j, rec in enumerate(data['recommendations'][:3], 1):
                    print(f"  {j}. {rec['assessment_name']}")
                    print(f"     Type: {rec['test_type']}, Score: {rec['similarity_score']:.3f}")
                    print(f"     URL: {rec['assessment_url']}")
                
                if data.get('explanation'):
                    print(f"\nExplanation: {data['explanation'][:100]}...")
                
                print("✓ Test passed")
                results.append(True)
            else:
                print(f"Response: {response.text}")
                print("✗ Test failed")
                results.append(False)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append(False)
    
    return all(results)


def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 80)
    print("Testing Error Handling")
    print("=" * 80)
    
    # Test cases for error scenarios
    error_cases = [
        {
            "name": "Empty Query",
            "payload": {"query": "", "top_k": 10},
            "expected_status": 422  # Validation error
        },
        {
            "name": "Invalid top_k (too large)",
            "payload": {"query": "test query", "top_k": 100},
            "expected_status": 422
        },
        {
            "name": "Invalid top_k (negative)",
            "payload": {"query": "test query", "top_k": -1},
            "expected_status": 422
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(error_cases, 1):
        print(f"\nError Test {i}: {test_case['name']}")
        print("-" * 80)
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/recommend",
                json=test_case['payload'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Expected: {test_case['expected_status']}")
            
            if response.status_code == test_case['expected_status']:
                print("✓ Error handling correct")
                results.append(True)
            else:
                print(f"Response: {response.text}")
                print("✗ Unexpected status code")
                results.append(False)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append(False)
    
    return all(results)


def test_root_endpoint():
    """Test the root endpoint"""
    print("\n" + "=" * 80)
    print("Testing Root Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✓ Root endpoint passed")
            return True
        else:
            print("✗ Root endpoint failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("API ENDPOINT TESTS")
    print("=" * 80)
    print(f"API Base URL: {API_BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  python scripts/run_api.py")
    print("=" * 80)
    
    # Run tests
    results = {
        "Root Endpoint": test_root_endpoint(),
        "Health Check": test_health_endpoint(),
        "Recommendations": test_recommend_endpoint(),
        "Error Handling": test_error_handling()
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
