"""
Pipeline Module
Complete retrieval pipeline with all stages
"""

import logging
from typing import List, Dict, Optional

from app.core.config import settings
from app.services.retrieval_service import SemanticRetriever

logger = logging.getLogger(__name__)


def retrieve_fuzzy_matches(
    retriever: SemanticRetriever,
    query: str,
    domain: str = None,
    source_lang: str = "en",
    target_lang: str = "ar",
    top_k: int = None,
    enable_hybrid: bool = True,
    enable_diversity: bool = True,
    enable_domain_boost: bool = True,
    enable_complexity_filter: bool = True
) -> List[Dict]:
    """
    Complete retrieval pipeline for finding fuzzy translation matches
    
    This is the main entry point for Phase 2 semantic retrieval.
    
    Pipeline stages:
    1. Multi-stage search (semantic + wording)
    2. Score combination
    3. Domain boosting (if enabled)
    4. Complexity filtering (if enabled)
    5. Diversity re-ranking (if enabled)
    
    Args:
        retriever: SemanticRetriever instance
        query: Source sentence to find matches for
        domain: Domain for filtering/boosting
        source_lang: Source language code
        target_lang: Target language code
        top_k: Number of results to return
        enable_hybrid: Use hybrid (semantic + wording) search
        enable_diversity: Apply MMR diversity re-ranking
        enable_domain_boost: Boost same-domain matches
        enable_complexity_filter: Filter by sentence complexity
        
    Returns:
        List of matching translations with rich metadata
    """
    if top_k is None:
        top_k = settings.DEFAULT_TOP_K
    
    logger.info(f"Retrieving matches for query: {query[:50]}...")
    
    # Stage 1 & 2: Multi-stage search
    if enable_diversity:
        # Use diversity-aware search (includes hybrid)
        results = retriever.search_with_diversity(
            query=query,
            top_k=top_k * 2,  # Get extra for filtering
            domain=domain if enable_domain_boost else None,
            source_lang=source_lang,
            target_lang=target_lang
        )
    elif enable_hybrid:
        # Use hybrid search
        results = retriever.search_hybrid(
            query=query,
            top_k=top_k * 2,
            domain=domain if enable_domain_boost else None,
            source_lang=source_lang,
            target_lang=target_lang
        )
    else:
        # Use simple semantic search
        results = retriever.search_semantic(
            query=query,
            top_k=top_k * 2,
            domain=domain,
            source_lang=source_lang,
            target_lang=target_lang
        )
    
    # Stage 3: Complexity filtering
    if enable_complexity_filter:
        results = filter_by_complexity(query, results)
    
    # Limit to top_k
    results = results[:top_k]
    
    # Format final results
    formatted_results = format_pipeline_results(query, results)
    
    logger.info(f"Retrieved {len(formatted_results)} matches")
    
    return formatted_results


def filter_by_complexity(query: str, results: List[Dict]) -> List[Dict]:
    """
    Filter results by sentence complexity similarity
    
    Removes matches that are too short/long compared to query
    """
    query_length = len(query.split())
    filtered = []
    
    for result in results:
        source_length = result.get('source_length', 0)
        
        # Check length ratio
        if source_length > 0:
            ratio = max(query_length, source_length) / max(min(query_length, source_length), 1)
            
            if ratio <= settings.MAX_LENGTH_RATIO:
                result['length_ratio'] = round(ratio, 2)
                filtered.append(result)
    
    return filtered


def format_pipeline_results(query: str, results: List[Dict]) -> List[Dict]:
    """Format results for output"""
    formatted = []
    
    for i, result in enumerate(results):
        formatted.append({
            "rank": i + 1,
            "id": result.get('id'),
            "similarity_percentage": result.get('similarity_percentage', 0),
            "score": result.get('score', 0),
            "source": result.get('source', ''),
            "target": result.get('target', ''),
            "domain": result.get('domain', ''),
            "language_pair": result.get('language_pair', ''),
            "source_lang": result.get('source_lang', ''),
            "target_lang": result.get('target_lang', ''),
            "source_length": result.get('source_length', 0),
            "target_length": result.get('target_length', 0),
            "search_type": result.get('search_type', 'unknown'),
            "metadata": {
                "semantic_score": result.get('semantic_score'),
                "wording_score": result.get('wording_score'),
                "domain_boosted": result.get('domain_boosted', False),
                "mmr_selected": result.get('mmr_selected', False),
                "length_ratio": result.get('length_ratio')
            }
        })
    
    return formatted


def batch_retrieve_fuzzy_matches(
    retriever: SemanticRetriever,
    queries: List[str],
    domain: str = None,
    source_lang: str = "en",
    target_lang: str = "ar",
    top_k: int = None
) -> Dict[str, List[Dict]]:
    """
    Retrieve matches for multiple queries
    
    Args:
        retriever: SemanticRetriever instance
        queries: List of source sentences
        domain: Domain for filtering
        source_lang: Source language code
        target_lang: Target language code
        top_k: Number of results per query
        
    Returns:
        Dict mapping queries to their results
    """
    results = {}
    
    for query in queries:
        results[query] = retrieve_fuzzy_matches(
            retriever=retriever,
            query=query,
            domain=domain,
            source_lang=source_lang,
            target_lang=target_lang,
            top_k=top_k
        )
    
    return results


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)
    
    from app.services.setup_qdrant import get_qdrant_client
    from app.services.retrieval_service import SemanticRetriever
    
    client = get_qdrant_client()
    retriever = SemanticRetriever(client)
    
    test_query = "Patients with severe symptoms require immediate care"
    
    results = retrieve_fuzzy_matches(
        retriever=retriever,
        query=test_query,
        domain="health",
        source_lang="en",
        target_lang="ar",
        top_k=5
    )
    
    print(f"\nResults for: {test_query}")
    for r in results:
        print(f"  [{r['rank']}] {r['similarity_percentage']}% - {r['source'][:50]}...")
