#!/usr/bin/env python3
"""
Quick test for the "answering instead of translating" fix
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.rag_service import get_prompting_service

def test_greeting_translation():
    """Test that 'Hello, how are you?' is translated, not answered"""
    
    print("="*70)
    print("Testing: Hello, how are you?")
    print("Expected: مرحباً، كيف حالك؟ (or similar)")
    print("NOT Expected: أنا جيد، شكراً (that's an answer, not a translation)")
    print("="*70)
    
    service = get_prompting_service()
    
    result = service.translate(
        text="Hello, how are you?",
        source_lang="en",
        target_lang="ar",
        domain="general",
        num_variants=3
    )
    
    translations = result.get('translations', [])
    
    print(f"\n✅ Got {len(translations)} translation(s):\n")
    
    for i, trans in enumerate(translations, 1):
        print(f"Variant {i}: {trans}")
        
        # Check if it's answering instead of translating
        if 'أنا جيد' in trans or 'أنا بخير' in trans or 'بخير' in trans:
            if 'كيف' not in trans:  # If it doesn't contain "how"
                print(f"  ❌ ERROR: This is an ANSWER, not a TRANSLATION!")
        else:
            print(f"  ✅ Looks good!")
        print()
    
    return result

if __name__ == '__main__':
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY not set!")
        sys.exit(1)
    
    test_greeting_translation()
