#!/usr/bin/env python3
"""
Test script for multi-language translation support
Tests all supported language pairs: EN↔AR, FR↔AR
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.rag_service import get_prompting_service

def test_translation_pair(text, source_lang, target_lang, description):
    """Test a single translation"""
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"{'='*70}")
    print(f"Input ({source_lang}): {text}")
    print(f"Target language: {target_lang}")
    print("-" * 70)
    
    service = get_prompting_service()
    
    try:
        result = service.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain="general",
            num_variants=3
        )
        
        translations = result.get('translations', [])
        
        if len(translations) > 0:
            print(f"✅ Success! Got {len(translations)} translation(s):\n")
            for i, trans in enumerate(translations, 1):
                print(f"  {i}. {trans}")
            return True
        else:
            print(f"❌ Failed: No translations returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive multi-language tests"""
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY not set!")
        print("Please set it in your .env file or export it:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Multi-Language Translation Test Suite")
    print("Testing: EN↔AR, FR↔AR")
    print("="*70)
    
    # Test cases for all language pairs
    test_cases = [
        {
            'text': 'Hello, how are you?',
            'source': 'en',
            'target': 'ar',
            'description': 'EN → AR: Greeting'
        },
        {
            'text': 'The patient has a fever.',
            'source': 'en',
            'target': 'ar',
            'description': 'EN → AR: Medical context'
        },
        {
            'text': 'Bonjour, comment allez-vous?',
            'source': 'fr',
            'target': 'ar',
            'description': 'FR → AR: Greeting'
        },
        {
            'text': 'Le patient a de la fièvre.',
            'source': 'fr',
            'target': 'ar',
            'description': 'FR → AR: Medical context'
        },
        {
            'text': 'مرحباً، كيف حالك؟',
            'source': 'ar',
            'target': 'en',
            'description': 'AR → EN: Greeting'
        },
        {
            'text': 'المريض يعاني من حمى.',
            'source': 'ar',
            'target': 'en',
            'description': 'AR → EN: Medical context'
        },
        {
            'text': 'مرحباً، كيف حالك؟',
            'source': 'ar',
            'target': 'fr',
            'description': 'AR → FR: Greeting'
        },
        {
            'text': 'المريض يعاني من حمى.',
            'source': 'ar',
            'target': 'fr',
            'description': 'AR → FR: Medical context'
        },
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*70}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'#'*70}")
        
        success = test_translation_pair(
            test_case['text'],
            test_case['source'],
            test_case['target'],
            test_case['description']
        )
        
        results.append({
            'description': test_case['description'],
            'success': success
        })
        
        # Pause between tests to avoid rate limiting
        if i < len(test_cases):
            import time
            print("\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    # Summary
    print("\n\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal Tests: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {len(results) - successful}")
    print(f"Success Rate: {successful/len(results)*100:.1f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {i}. {status} - {result['description']}")
    
    print("\n" + "="*70)
    
    # Exit with error code if any test failed
    if successful < len(results):
        print("\n⚠️  Some tests failed. Check logs above for details.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
