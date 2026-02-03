#!/usr/bin/env python3
"""
Test script for Gemma Translation Service
Run this to verify the translation service is working correctly.
"""

import os
import sys

# Add the parent directory to path so we can import the api module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def test_translation():
    """Test the Gemma translation service with a sample sentence"""
    
    print("=" * 60)
    print("GEMMA TRANSLATION SERVICE TEST")
    print("=" * 60)
    
    # Import the service
    print("\n[1] Importing translation service...")
    from api.rag_service import get_prompting_service
    
    # Get the service (this will load the model)
    print("\n[2] Initializing service (loading model)...")
    print("    This may take a few minutes on first run...")
    service = get_prompting_service()
    
    print("\n[3] Model loaded successfully!")
    print(f"    Model: {service.gemma_model_id}")
    
    # Test translations
    test_cases = [
        {
            "text": "Hello, how are you?",
            "source": "en",
            "target": "ar",
            "domain": "general"
        },
        {
            "text": "The patient needs immediate medical attention.",
            "source": "en",
            "target": "ar",
            "domain": "medical"
        },
        {
            "text": "Machine learning models require large datasets for training.",
            "source": "en",
            "target": "ar",
            "domain": "technology"
        },
    ]
    
    print("\n[4] Running translation tests...")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ({test['domain']}) ---")
        print(f"Input ({test['source']}): {test['text']}")
        
        try:
            result = service.translate(
                text=test['text'],
                source_lang=test['source'],
                target_lang=test['target'],
                domain=test['domain'],
                num_variants=1  # Just get 1 variant for quick testing
            )
            
            translations = result.get('translations', [])
            if translations:
                print(f"Output ({test['target']}): {translations[0]}")
            else:
                print("ERROR: No translation returned")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_translation()
