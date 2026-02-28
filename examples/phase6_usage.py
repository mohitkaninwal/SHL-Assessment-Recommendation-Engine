#!/usr/bin/env python3
"""
Phase 6 Usage Examples
Demonstrates how to use the FastAPI endpoints
"""

import requests
import json
from typing import Dict, Any


# API Configuration
API_BASE_URL = "http://localhost:8000"


def example_1_health_check():
    """Example 1: Check API health"""
    print("=" * 80)
    print("Example 1: Health Check")
    print("=" * 80)
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))


def example_2_simple_recommendation():
    """Example 2: Simple recommendation request"""
    print("\n" + "=" * 80)
    print("Example 2: Simple Recommendation")
    print("=" * 80)
    
    payload = {
        "query": "I am hiring for Java developers",
        "top_k": 5
    }
    
    print(f"Request:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(
        f"{API_BASE_URL}/recommend",
        json=payload
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: {data['query']}")
        print(f"Total Found: {data['total_found']}")
        print(f"Returned: {data['returned']}")
        print(f"\nRecommendations:")
        
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   Type: {rec['test_type']}")
            print(f"   Score: {rec['similarity_score']:.3f}")
            print(f"   URL: {rec['assessment_url']}")
    else:
        print(f"Error: {response.text}")


def example_3_detailed_recommendation():
    """Example 3: Recommendation with explanation"""
    print("\n" + "=" * 80)
    print("Example 3: Detailed Recommendation with Explanation")
    print("=" * 80)
    
    payload = {
        "query": "I want to hire new graduates for a sales role who are good at communication",
        "top_k": 10,
        "balance_skills": True,
        "include_explanation": True
    }
    
    print(f"Request:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(
        f"{API_BASE_URL}/recommend",
        json=payload
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\nQuery: {data['query'][:60]}...")
        print(f"Total Found: {data['total_found']}")
        print(f"Returned: {data['returned']}")
        
        if data.get('query_analysis'):
            analysis = data['query_analysis']
            print(f"\nQuery Analysis:")
            print(f"  Technical Weight: {analysis.get('technical_weight', 0):.2f}")
            print(f"  Behavioral Weight: {analysis.get('behavioral_weight', 0):.2f}")
            if analysis.get('primary_skills'):
                print(f"  Primary Skills: {', '.join(analysis['primary_skills'][:5])}")
        
        print(f"\nTop {min(5, len(data['recommendations']))} Recommendations:")
        for i, rec in enumerate(data['recommendations'][:5], 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   Type: {rec['test_type']}")
            print(f"   Score: {rec['similarity_score']:.3f}")
            print(f"   URL: {rec['assessment_url']}")
            if rec.get('description'):
                print(f"   Description: {rec['description'][:80]}...")
        
        if data.get('explanation'):
            print(f"\nExplanation:")
            print(f"  {data['explanation'][:200]}...")
    else:
        print(f"Error: {response.text}")


def example_4_multiple_queries():
    """Example 4: Process multiple queries"""
    print("\n" + "=" * 80)
    print("Example 4: Multiple Queries")
    print("=" * 80)
    
    queries = [
        "Python developer with data science skills",
        "Customer service representative with empathy",
        "Project manager with leadership abilities"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"query": query, "top_k": 3}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total_found']} assessments")
            print(f"Top 3:")
            for j, rec in enumerate(data['recommendations'], 1):
                print(f"  {j}. {rec['assessment_name']} (Score: {rec['similarity_score']:.3f})")
        else:
            print(f"Error: {response.status_code}")


def example_5_error_handling():
    """Example 5: Error handling"""
    print("\n" + "=" * 80)
    print("Example 5: Error Handling")
    print("=" * 80)
    
    # Test invalid requests
    invalid_requests = [
        {
            "name": "Empty query",
            "payload": {"query": "", "top_k": 10}
        },
        {
            "name": "Invalid top_k",
            "payload": {"query": "test", "top_k": 100}
        },
        {
            "name": "Missing query",
            "payload": {"top_k": 10}
        }
    ]
    
    for test in invalid_requests:
        print(f"\nTest: {test['name']}")
        print(f"Payload: {test['payload']}")
        
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=test['payload']
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error Response: {response.json()}")


def example_6_client_class():
    """Example 6: Using a client class"""
    print("\n" + "=" * 80)
    print("Example 6: Client Class")
    print("=" * 80)
    
    class SHLRecommendationClient:
        """Client for SHL Recommendation API"""
        
        def __init__(self, base_url: str = API_BASE_URL):
            self.base_url = base_url
        
        def health_check(self) -> Dict[str, Any]:
            """Check API health"""
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        
        def get_recommendations(
            self,
            query: str,
            top_k: int = 10,
            balance_skills: bool = True,
            include_explanation: bool = False
        ) -> Dict[str, Any]:
            """Get recommendations"""
            payload = {
                "query": query,
                "top_k": top_k,
                "balance_skills": balance_skills,
                "include_explanation": include_explanation
            }
            
            response = requests.post(
                f"{self.base_url}/recommend",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    # Use the client
    client = SHLRecommendationClient()
    
    # Health check
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Engine Status: {health['engine_status']}")
    
    # Get recommendations
    print("\nGetting recommendations...")
    results = client.get_recommendations(
        query="I need to assess analytical thinking skills",
        top_k=5
    )
    
    print(f"Query: {results['query']}")
    print(f"Found {results['returned']} recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec['assessment_name']} (Score: {rec['similarity_score']:.3f})")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("PHASE 6 API USAGE EXAMPLES")
    print("=" * 80)
    print(f"API Base URL: {API_BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  python scripts/run_api.py")
    print("=" * 80)
    
    try:
        # Run examples
        example_1_health_check()
        
        # Uncomment to run other examples (requires API to be running)
        # example_2_simple_recommendation()
        # example_3_detailed_recommendation()
        # example_4_multiple_queries()
        # example_5_error_handling()
        # example_6_client_class()
        
        print("\n" + "=" * 80)
        print("Examples completed!")
        print("=" * 80)
        print("\nTo run the API:")
        print("  python scripts/run_api.py")
        print("\nTo test the API:")
        print("  python scripts/test_api.py")
        print("\nInteractive docs:")
        print("  http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API server is running:")
        print("  python scripts/run_api.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
