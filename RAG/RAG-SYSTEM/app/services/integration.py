"""
Integration Module
Integration with Phase 1 (Glossary System)
"""

import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.services.retrieval_service import SemanticRetriever
from app.services.pipeline import retrieve_fuzzy_matches

logger = logging.getLogger(__name__)


def integrate_with_phase1(
    phase1_output: Dict,
    retriever: SemanticRetriever
) -> Dict:
    """
    Integrate Phase 1 (glossary) output with Phase 2 (semantic retrieval)
    
    Takes the output from Phase 1 glossary lookup and enriches it with
    semantic translation memory matches.
    
    Phase 1 Output Structure:
    {
        'source_sentence': str,
        'glossary_matches': [
            {'source_term': str, 'target_term': str, 'domain': str, ...}
        ],
        'domain': str,
        'source_lang': str,
        'target_lang': str
    }
    
    Args:
        phase1_output: Output from Phase 1 glossary system
        retriever: SemanticRetriever instance
        
    Returns:
        Combined output with glossary AND fuzzy matches
    """
    # Extract Phase 1 data
    source_sentence = phase1_output.get('source_sentence', '')
    glossary_matches = phase1_output.get('glossary_matches', [])
    domain = phase1_output.get('domain')
    source_lang = phase1_output.get('source_lang', 'en')
    target_lang = phase1_output.get('target_lang', 'ar')
    
    if not source_sentence:
        logger.warning("No source sentence in Phase 1 output")
        return phase1_output
    
    logger.info(f"Integrating Phase 2 for: {source_sentence[:50]}...")
    
    # Execute Phase 2 retrieval
    fuzzy_matches = retrieve_fuzzy_matches(
        retriever=retriever,
        query=source_sentence,
        domain=domain,
        source_lang=source_lang,
        target_lang=target_lang,
        top_k=settings.DEFAULT_TOP_K,
        enable_hybrid=True,
        enable_diversity=True,
        enable_domain_boost=True if domain else False
    )
    
    # Combine outputs
    phase2_output = {
        # Original Phase 1 data
        'source_sentence': source_sentence,
        'domain': domain,
        'source_lang': source_lang,
        'target_lang': target_lang,
        
        # Phase 1: Glossary matches (exact term matches)
        'glossary_matches': glossary_matches,
        'glossary_count': len(glossary_matches),
        
        # Phase 2: Fuzzy/semantic matches (similar sentences)
        'fuzzy_matches': fuzzy_matches,
        'fuzzy_count': len(fuzzy_matches),
        
        # Combined statistics
        'total_matches': len(glossary_matches) + len(fuzzy_matches),
        'has_glossary_support': len(glossary_matches) > 0,
        'has_fuzzy_support': len(fuzzy_matches) > 0
    }
    
    logger.info(f"Phase 2 complete: {len(glossary_matches)} glossary, {len(fuzzy_matches)} fuzzy matches")
    
    return phase2_output


def format_for_prompt(phase2_output: Dict) -> str:
    """
    Format Phase 2 output for LLM prompt construction (Phase 3)
    
    Creates a structured text format that can be included in translation prompts.
    
    Args:
        phase2_output: Combined output from integrate_with_phase1
        
    Returns:
        Formatted string for prompt inclusion
    """
    sections = []
    
    # Source sentence
    source = phase2_output.get('source_sentence', '')
    sections.append(f"Source Sentence: {source}")
    sections.append("")
    
    # Glossary terms section
    glossary_matches = phase2_output.get('glossary_matches', [])
    if glossary_matches:
        sections.append("=== Glossary Terms ===")
        for match in glossary_matches:
            source_term = match.get('source_term', '')
            target_term = match.get('target_term', '')
            sections.append(f"• {source_term} → {target_term}")
        sections.append("")
    
    # Similar translations section
    fuzzy_matches = phase2_output.get('fuzzy_matches', [])
    if fuzzy_matches:
        sections.append("=== Similar Translations ===")
        for match in fuzzy_matches[:5]:  # Top 5 for prompt
            similarity = match.get('similarity_percentage', 0)
            source_text = match.get('source', '')
            target_text = match.get('target', '')
            sections.append(f"[{similarity}%] Source: {source_text}")
            sections.append(f"       Target: {target_text}")
            sections.append("")
    
    return "\n".join(sections)


def create_translation_context(
    source_sentence: str,
    domain: str,
    source_lang: str,
    target_lang: str,
    retriever: SemanticRetriever,
    glossary_matches: List[Dict] = None
) -> Dict:
    """
    Create complete translation context from scratch
    
    Convenience function that combines glossary lookup with semantic retrieval.
    
    Args:
        source_sentence: Sentence to translate
        domain: Translation domain
        source_lang: Source language
        target_lang: Target language
        retriever: SemanticRetriever instance
        glossary_matches: Pre-fetched glossary matches (optional)
        
    Returns:
        Complete context for translation
    """
    # Build Phase 1 format
    phase1_output = {
        'source_sentence': source_sentence,
        'glossary_matches': glossary_matches or [],
        'domain': domain,
        'source_lang': source_lang,
        'target_lang': target_lang
    }
    
    # Execute Phase 2 integration
    return integrate_with_phase1(phase1_output, retriever)


async def integrate_with_phase1_async(
    phase1_output: Dict,
    retriever: SemanticRetriever
) -> Dict:
    """Async version of integrate_with_phase1"""
    # For now, just wrap the sync version
    # Can be optimized with async Qdrant client later
    return integrate_with_phase1(phase1_output, retriever)


if __name__ == "__main__":
    # Test integration
    logging.basicConfig(level=logging.INFO)
    
    from app.services.setup_qdrant import get_qdrant_client
    from app.services.retrieval_service import SemanticRetriever
    
    client = get_qdrant_client()
    retriever = SemanticRetriever(client)
    
    # Simulate Phase 1 output
    phase1_output = {
        'source_sentence': "Patients with severe symptoms require immediate care",
        'glossary_matches': [
            {'source_term': 'patients', 'target_term': 'المرضى', 'domain': 'health'},
            {'source_term': 'symptoms', 'target_term': 'الأعراض', 'domain': 'health'}
        ],
        'domain': 'health',
        'source_lang': 'en',
        'target_lang': 'ar'
    }
    
    # Execute Phase 2
    result = integrate_with_phase1(phase1_output, retriever)
    
    print("\n=== Phase 2 Integration Result ===")
    print(f"Glossary matches: {result['glossary_count']}")
    print(f"Fuzzy matches: {result['fuzzy_count']}")
    
    print("\n=== Formatted for Prompt ===")
    print(format_for_prompt(result))
