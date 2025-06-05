#!/usr/bin/env python3
"""
Test script to verify API connections for Creative Writing Benchmark
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_endpoint(endpoint_name, api_key, api_url, model_name):
    """Test a single API endpoint with a simple request"""
    print(f"\n=== Testing {endpoint_name} API ===")
    print(f"URL: {api_url}")
    print(f"Model: {model_name}")
    print(f"API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if api_key else 'NOT SET'}")
    
    if not api_key:
        print("❌ ERROR: API key not set!")
        return False
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,
        "max_tokens": 100,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'API test successful' if you can read this."
            }
        ]
    }
    
    try:
        print("Making API request...")
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ SUCCESS: {content.strip()}")
            return True
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Connection failed")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    print("🧪 Creative Writing Benchmark API Connection Test")
    print("=" * 50)
    
    # Test model configuration
    test_api_key = os.getenv("TEST_API_KEY")
    test_api_url = os.getenv("TEST_API_URL", "https://api.fireworks.ai/inference/v1/chat/completions")
    test_model = "accounts/fireworks/models/deepseek-v3"
    
    # Judge model configuration  
    judge_api_key = os.getenv("JUDGE_API_KEY")
    judge_api_url = os.getenv("JUDGE_API_URL", "https://api.fireworks.ai/inference/v1/chat/completions")
    judge_model = "accounts/fireworks/models/deepseek-v3"
    
    # Test both endpoints
    test_success = test_api_endpoint("TEST", test_api_key, test_api_url, test_model)
    judge_success = test_api_endpoint("JUDGE", judge_api_key, judge_api_url, judge_model)
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"Test API:  {'✅ Working' if test_success else '❌ Failed'}")
    print(f"Judge API: {'✅ Working' if judge_success else '❌ Failed'}")
    
    if test_success and judge_success:
        print("\n🎉 All APIs working! Ready to run the benchmark.")
        return 0
    else:
        print("\n⚠️  Fix the API issues before running the benchmark.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 