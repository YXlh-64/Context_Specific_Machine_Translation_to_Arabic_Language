#!/usr/bin/env python3
"""
Test script for DeepL and OpenRouter integration
Tests both services and compares results
"""

import requests
import json
import os
import sys

# Test configuration
API_BASE_URL = "http://localhost:5002"
TEST_TEXTS = [
    {
        "text": "Hello, how are you?",
        "source": "en",
        "target": "ar",
        "description": "Simple English greeting to Arabic"
    },
    {
        "text": "Bonjour, comment allez-vous?",
        "source": "fr",
        "target": "ar",
        "description": "French greeting to Arabic"
    },
    {
        "text": "مرحباً، كيف حالك؟",
        "source": "ar",
        "target": "en",
        "description": "Arabic greeting to English"
    },
    {
        "text": "The medical report indicates a significant improvement in the patient's condition.",
        "source": "en",
        "target": "ar",
        "description": "Medical text (testing domain handling)"
    }
]

def test_service(service_name, text, source_lang, target_lang, num_variants=1):
    """Test a specific translation service"""
    print(f"\n{'='*70}")
    print(f"Testing {service_name.upper()}")
    print(f"{'='*70}")
    print(f"Text: {text}")
    print(f"Direction: {source_lang} → {target_lang}")
    
    payload = {
        "text": text,
        "source_language": source_lang,
        "target_language": target_lang,
        "translation_service": service_name,
        "num_variants": num_variants
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/translate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS")
            print(f"Service: {result['metadata'].get('service', service_name)}")
            print(f"Variants: {result['metadata'].get('num_variants', len(result['translations']))}")
            print(f"\nTranslation(s):")
            for i, trans in enumerate(result['translations'], 1):
                print(f"  {i}. {trans}")
            return result
        else:
            print(f"\n❌ FAILED (HTTP {response.status_code})")
            print(f"Error: {response.json().get('error', 'Unknown error')}")
            if 'detail' in response.json():
                print(f"Detail: {response.json()['detail']}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION FAILED")
        print(f"Make sure the server is running at {API_BASE_URL}")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None

def check_api_keys():
    """Check which API keys are configured"""
    print("\n" + "="*70)
    print("API KEY CONFIGURATION CHECK")
    print("="*70)
    
    deepl_key = os.getenv('DEEPL_API_KEY')
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    service_type = os.getenv('TRANSLATION_SERVICE', 'deepl')
    
    print(f"Default Service: {service_type}")
    print(f"DeepL API Key: {'✅ Set' if deepl_key else '❌ Not set'}")
    print(f"OpenRouter API Key: {'✅ Set' if openrouter_key else '❌ Not set'}")
    
    if not deepl_key and not openrouter_key:
        print("\n⚠️  WARNING: No API keys configured!")
        print("   Set DEEPL_API_KEY or OPENROUTER_API_KEY in your .env file")
        return False
    
    return True

def main():
    print("="*70)
    print("DEEPL & OPENROUTER INTEGRATION TEST")
    print("="*70)
    
    # Check server is running
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server returned error")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server at {API_BASE_URL}")
        print(f"   Error: {e}")
        print("\n💡 Start the server with: cd app && python -m flask run --port=5002")
        sys.exit(1)
    
    # Check API keys
    if not check_api_keys():
        print("\n💡 Get a free DeepL key at: https://www.deepl.com/pro-api")
        print("   Add to .env file: DEEPL_API_KEY=your_key_here")
    
    # Run tests
    results = {
        'deepl': [],
        'openrouter': []
    }
    
    # Test DeepL
    print("\n" + "="*70)
    print("TESTING DEEPL (FREE)")
    print("="*70)
    
    for test_case in TEST_TEXTS:
        print(f"\n--- {test_case['description']} ---")
        result = test_service(
            'deepl',
            test_case['text'],
            test_case['source'],
            test_case['target'],
            num_variants=1
        )
        if result:
            results['deepl'].append({
                'test': test_case['description'],
                'success': True,
                'translation': result['translation']
            })
        else:
            results['deepl'].append({
                'test': test_case['description'],
                'success': False
            })
    
    # Test OpenRouter (if configured)
    if os.getenv('OPENROUTER_API_KEY'):
        print("\n" + "="*70)
        print("TESTING OPENROUTER (PAID - 3 VARIANTS)")
        print("="*70)
        
        # Test only first case to save credits
        test_case = TEST_TEXTS[0]
        print(f"\n--- {test_case['description']} ---")
        result = test_service(
            'openrouter',
            test_case['text'],
            test_case['source'],
            test_case['target'],
            num_variants=3
        )
        if result:
            results['openrouter'].append({
                'test': test_case['description'],
                'success': True,
                'translations': result['translations']
            })
    else:
        print("\n" + "="*70)
        print("SKIPPING OPENROUTER TESTS (No API key configured)")
        print("="*70)
        print("To test OpenRouter:")
        print("  1. Get API key at: https://openrouter.ai/")
        print("  2. Add to .env: OPENROUTER_API_KEY=your_key_here")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    deepl_success = sum(1 for r in results['deepl'] if r['success'])
    deepl_total = len(results['deepl'])
    print(f"DeepL: {deepl_success}/{deepl_total} tests passed")
    
    if results['openrouter']:
        openrouter_success = sum(1 for r in results['openrouter'] if r['success'])
        openrouter_total = len(results['openrouter'])
        print(f"OpenRouter: {openrouter_success}/{openrouter_total} tests passed")
    
    print("\n" + "="*70)
    if deepl_success == deepl_total:
        print("✅ ALL DEEPL TESTS PASSED!")
        print("\n💡 DeepL is working perfectly. You can use it for free translations.")
        print("   Set TRANSLATION_SERVICE=deepl in .env (or omit - it's the default)")
    else:
        print("❌ SOME TESTS FAILED")
        print("\n💡 Check your DEEPL_API_KEY in .env file")
        print("   Get a free key at: https://www.deepl.com/pro-api")
    
    print("="*70)

if __name__ == "__main__":
    main()
