"""
Test script for running Phase 3 with real data from Phase 1 and Phase 2

This script demonstrates the full pipeline:
1. Calls Phase 1 (Glossary) API
2. Calls Phase 2 (RAG) API
3. Constructs prompt using Phase 3
"""

import httpx
import asyncio
import json
from typing import Optional


# API URLs
PHASE1_URL = "http://localhost:8001"
PHASE2_URL = "http://localhost:8002"
PHASE3_URL = "http://localhost:8003"


async def check_service(url: str, name: str) -> bool:
    """Check if a service is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                print(f"✅ {name} is running at {url}")
                return True
    except Exception as e:
        print(f"❌ {name} is not available at {url}: {e}")
    return False


async def call_phase1_glossary(text: str, domain: str = "technology") -> dict:
    """Call Phase 1 Glossary API."""
    print(f"\n📚 Phase 1: Looking up glossary terms...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{PHASE1_URL}/translate/sentence",
                json={
                    "text": text,
                    "source_lang": "en",
                    "target_lang": "ar",
                    "domain": domain
                }
            )
            if response.status_code == 200:
                data = response.json()
                matches = data.get("glossary_matches", [])
                print(f"   Found {len(matches)} glossary terms")
                return data
            else:
                print(f"   Phase 1 returned status {response.status_code}")
                return {}
    except Exception as e:
        print(f"   Phase 1 error: {e}")
        return {}


async def call_phase2_rag(query: str, domain: str = "technology") -> dict:
    """Call Phase 2 RAG API."""
    print(f"\n🔍 Phase 2: Searching for fuzzy matches...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{PHASE2_URL}/search/hybrid",
                json={
                    "query": query,
                    "domain": domain,
                    "source_lang": "en",
                    "target_lang": "ar",
                    "top_k": 5,
                    "semantic_weight": 0.6,
                    "wording_weight": 0.4
                }
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print(f"   Found {len(results)} fuzzy matches")
                return data
            else:
                print(f"   Phase 2 returned status {response.status_code}")
                return {}
    except Exception as e:
        print(f"   Phase 2 error: {e}")
        return {}


async def call_phase3_construct(
    source_sentence: str,
    glossary_matches: list,
    fuzzy_matches: list,
    domain: str = "technology",
    prompt_format: str = "xml"
) -> dict:
    """Call Phase 3 Prompt Construction API."""
    print(f"\n🔧 Phase 3: Constructing prompt ({prompt_format} format)...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{PHASE3_URL}/prompt/construct",
                json={
                    "source_sentence": source_sentence,
                    "glossary_matches": glossary_matches,
                    "fuzzy_matches": fuzzy_matches,
                    "domain": domain,
                    "source_lang": "en",
                    "target_lang": "ar",
                    "prompt_format": prompt_format,
                    "include_system_message": True
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"   Prompt constructed: {data.get('token_count', 0)} tokens")
                return data
            else:
                print(f"   Phase 3 returned status {response.status_code}")
                return {}
    except Exception as e:
        print(f"   Phase 3 error: {e}")
        return {}


async def call_full_pipeline(source_sentence: str, domain: str = "technology") -> dict:
    """Call the full pipeline endpoint."""
    print(f"\n🚀 Full Pipeline: Running complete pipeline...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PHASE3_URL}/pipeline/full",
                json={
                    "source_sentence": source_sentence,
                    "domain": domain,
                    "source_lang": "en",
                    "target_lang": "ar",
                    "generate_translation": False  # Set True if LLM configured
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"   Pipeline complete in {data.get('processing_time_ms', 0):.1f}ms")
                return data
            else:
                print(f"   Pipeline returned status {response.status_code}")
                return {}
    except Exception as e:
        print(f"   Pipeline error: {e}")
        return {}


async def main():
    """Run the full pipeline test."""
    print("=" * 70)
    print("Phase 3 Prompt Construction - Pipeline Test")
    print("=" * 70)
    
    # Test sentence
    test_sentence = "Machine learning algorithms are transforming the healthcare industry."
    domain = "technology"
    
    print(f"\n📝 Test Sentence: '{test_sentence}'")
    print(f"📁 Domain: {domain}")
    
    # Check services
    print("\n" + "-" * 50)
    print("Checking Services...")
    print("-" * 50)
    
    phase1_ok = await check_service(PHASE1_URL, "Phase 1 (Glossary)")
    phase2_ok = await check_service(PHASE2_URL, "Phase 2 (RAG)")
    phase3_ok = await check_service(PHASE3_URL, "Phase 3 (Prompt Construction)")
    
    if phase3_ok:
        print("\n" + "-" * 50)
        print("Testing Phase 3 Directly...")
        print("-" * 50)
        
        # Test with sample data
        sample_glossary = [
            {"source_term": "machine learning", "target_term": "التعلم الآلي", "domain": "technology"},
            {"source_term": "healthcare", "target_term": "الرعاية الصحية", "domain": "technology"}
        ]
        sample_fuzzy = [
            {"source": "ML is changing healthcare.", "target": "التعلم الآلي يغير الرعاية الصحية.", "similarity_percentage": 75.0}
        ]
        
        # Test prompt construction
        result = await call_phase3_construct(
            test_sentence,
            sample_glossary,
            sample_fuzzy,
            domain,
            "xml"
        )
        
        if result:
            print(f"\n✅ Phase 3 Direct Test: SUCCESS")
            print(f"   Token Count: {result.get('token_count', 0)}")
            print(f"   Format: {result.get('format', 'unknown')}")
    
    if phase1_ok and phase2_ok and phase3_ok:
        print("\n" + "-" * 50)
        print("Testing Full Pipeline (All 3 Phases)...")
        print("-" * 50)
        
        # Step-by-step test
        # Phase 1
        glossary_result = await call_phase1_glossary(test_sentence, domain)
        glossary_matches = glossary_result.get("glossary_matches", [])
        
        # Phase 2
        rag_result = await call_phase2_rag(test_sentence, domain)
        fuzzy_matches = []
        for r in rag_result.get("results", []):
            fuzzy_matches.append({
                "source": r.get("source", ""),
                "target": r.get("target", ""),
                "similarity_percentage": r.get("similarity_percentage", 0)
            })
        
        # Phase 3
        prompt_result = await call_phase3_construct(
            test_sentence,
            glossary_matches,
            fuzzy_matches,
            domain,
            "xml"
        )
        
        if prompt_result:
            print("\n" + "=" * 70)
            print("PIPELINE RESULTS")
            print("=" * 70)
            print(f"Glossary Terms: {len(glossary_matches)}")
            print(f"Fuzzy Matches: {len(fuzzy_matches)}")
            print(f"Token Count: {prompt_result.get('token_count', 0)}")
            
            print("\n--- System Message ---")
            sys_msg = prompt_result.get("system_message", "")
            if sys_msg:
                print(sys_msg[:200] + "..." if len(sys_msg) > 200 else sys_msg)
            
            print("\n--- Constructed Prompt (preview) ---")
            prompt = prompt_result.get("prompt", "")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        # Test full pipeline endpoint
        print("\n" + "-" * 50)
        print("Testing /pipeline/full endpoint...")
        print("-" * 50)
        
        full_result = await call_full_pipeline(test_sentence, domain)
        if full_result:
            print(f"\n✅ Full Pipeline Test: SUCCESS")
            print(f"   Glossary Count: {full_result.get('glossary_count', 0)}")
            print(f"   Fuzzy Count: {full_result.get('fuzzy_count', 0)}")
            print(f"   Token Count: {full_result.get('token_count', 0)}")
            print(f"   Processing Time: {full_result.get('processing_time_ms', 0):.1f}ms")
    
    else:
        print("\n⚠️ Not all services are running. Start them with:")
        if not phase1_ok:
            print("   Phase 1: cd PROJECT/glossary-system && uvicorn app.main:app --port 8001")
        if not phase2_ok:
            print("   Phase 2: cd PROJECT/RAG-SYSTEM && uvicorn app.main:app --port 8002")
        if not phase3_ok:
            print("   Phase 3: cd PROJECT/prompt-construction && uvicorn app.main:app --port 8003")
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
