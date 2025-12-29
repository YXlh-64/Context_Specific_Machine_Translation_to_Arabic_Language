"""
Evaluation Module
Quality evaluation metrics for RAG retrieval
"""

import logging
import time
from typing import List, Dict, Callable
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def evaluate_retrieval_quality(
    retriever,  # SemanticRetriever
    test_set: List[Dict],
    top_k: int = 7
) -> Dict:
    """
    Evaluate retrieval quality on a test set
    
    Args:
        retriever: SemanticRetriever instance
        test_set: List of test cases with 'query' and 'expected_domain'
        top_k: Number of results to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    from app.services.pipeline import retrieve_fuzzy_matches
    
    total_queries = len(test_set)
    total_time = 0
    results_counts = []
    domain_accuracy = []
    similarity_scores = []
    
    for test_case in test_set:
        query = test_case.get('query', '')
        expected_domain = test_case.get('expected_domain')
        
        start_time = time.time()
        
        try:
            results = retrieve_fuzzy_matches(
                retriever=retriever,
                query=query,
                domain=expected_domain,
                top_k=top_k
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            results_counts.append(len(results))
            
            if results:
                # Check domain accuracy
                if expected_domain:
                    domain_matches = sum(1 for r in results if r.get('domain') == expected_domain)
                    domain_accuracy.append(domain_matches / len(results))
                
                # Collect similarity scores
                scores = [r.get('score', 0) for r in results]
                similarity_scores.extend(scores)
                
        except Exception as e:
            logger.error(f"Evaluation error for query '{query[:30]}...': {e}")
            results_counts.append(0)
    
    # Calculate metrics
    metrics = {
        "total_queries": total_queries,
        "avg_response_time_ms": (total_time / total_queries * 1000) if total_queries > 0 else 0,
        "avg_results_count": np.mean(results_counts) if results_counts else 0,
        "min_results_count": min(results_counts) if results_counts else 0,
        "max_results_count": max(results_counts) if results_counts else 0,
        "queries_with_results": sum(1 for c in results_counts if c > 0),
        "queries_without_results": sum(1 for c in results_counts if c == 0),
        "recall_rate": sum(1 for c in results_counts if c > 0) / total_queries if total_queries > 0 else 0
    }
    
    if domain_accuracy:
        metrics["domain_accuracy"] = np.mean(domain_accuracy)
    
    if similarity_scores:
        metrics["avg_similarity"] = np.mean(similarity_scores)
        metrics["min_similarity"] = min(similarity_scores)
        metrics["max_similarity"] = max(similarity_scores)
        metrics["similarity_std"] = np.std(similarity_scores)
    
    return metrics


def evaluate_diversity(results: List[Dict]) -> Dict:
    """
    Evaluate diversity of retrieval results
    
    Args:
        results: List of retrieval results
        
    Returns:
        Diversity metrics
    """
    if not results:
        return {"diversity_score": 0}
    
    # Unique sources
    sources = [r.get('source', '') for r in results]
    unique_sources = len(set(sources))
    
    # Domain distribution
    domains = [r.get('domain', 'unknown') for r in results]
    unique_domains = len(set(domains))
    
    # Length distribution
    lengths = [r.get('source_length', 0) for r in results]
    length_variance = np.var(lengths) if lengths else 0
    
    return {
        "total_results": len(results),
        "unique_sources": unique_sources,
        "uniqueness_ratio": unique_sources / len(results) if results else 0,
        "unique_domains": unique_domains,
        "domain_diversity": unique_domains / len(results) if results else 0,
        "length_variance": float(length_variance)
    }


def evaluate_relevance(query: str, results: List[Dict], model=None) -> Dict:
    """
    Evaluate relevance of results to query using embeddings
    
    Args:
        query: Original query
        results: Retrieved results
        model: Embedding model (optional)
        
    Returns:
        Relevance metrics
    """
    if not results:
        return {"relevance_score": 0}
    
    if model is None:
        from app.services.embedding_service import get_model
        model = get_model()
    
    # Get query embedding
    query_embedding = model.encode(query, normalize_embeddings=True)
    
    # Calculate relevance for each result
    relevance_scores = []
    for result in results:
        source = result.get('source', '')
        if source:
            source_embedding = model.encode(source, normalize_embeddings=True)
            similarity = np.dot(query_embedding, source_embedding)
            relevance_scores.append(float(similarity))
    
    if not relevance_scores:
        return {"relevance_score": 0}
    
    return {
        "avg_relevance": np.mean(relevance_scores),
        "min_relevance": min(relevance_scores),
        "max_relevance": max(relevance_scores),
        "relevance_std": np.std(relevance_scores),
        "highly_relevant_count": sum(1 for s in relevance_scores if s > 0.8),
        "moderately_relevant_count": sum(1 for s in relevance_scores if 0.5 <= s <= 0.8),
        "low_relevance_count": sum(1 for s in relevance_scores if s < 0.5)
    }


def create_test_set() -> List[Dict]:
    """
    Create a test set for evaluation
    
    Returns:
        List of test cases
    """
    return [
        {
            "query": "Patients with severe symptoms require immediate care",
            "expected_domain": "health",
            "description": "Medical emergency scenario"
        },
        {
            "query": "The new smartphone features advanced artificial intelligence",
            "expected_domain": "technology",
            "description": "Technology product description"
        },
        {
            "query": "Agricultural practices have evolved significantly over centuries",
            "expected_domain": "agriculture",
            "description": "Agricultural history"
        },
        {
            "query": "The stock market experienced significant volatility",
            "expected_domain": "finance",
            "description": "Financial market analysis"
        },
        {
            "query": "The ancient civilization developed sophisticated writing systems",
            "expected_domain": "history",
            "description": "Historical development"
        },
        {
            "query": "The contract stipulates specific terms and conditions",
            "expected_domain": "legal",
            "description": "Legal document"
        },
        {
            "query": "Chronic bronchitis is a long-term disease of the lungs",
            "expected_domain": "health",
            "description": "Medical condition description"
        },
        {
            "query": "Machine learning algorithms analyze large datasets",
            "expected_domain": "technology",
            "description": "AI/ML description"
        }
    ]


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation report"""
    print("\n" + "=" * 60)
    print("          RAG SYSTEM EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n📊 Query Statistics:")
    print(f"   Total queries evaluated: {metrics.get('total_queries', 0)}")
    print(f"   Queries with results: {metrics.get('queries_with_results', 0)}")
    print(f"   Queries without results: {metrics.get('queries_without_results', 0)}")
    print(f"   Recall rate: {metrics.get('recall_rate', 0):.2%}")
    
    print(f"\n⏱️ Performance:")
    print(f"   Average response time: {metrics.get('avg_response_time_ms', 0):.2f} ms")
    
    print(f"\n📈 Results Statistics:")
    print(f"   Average results per query: {metrics.get('avg_results_count', 0):.1f}")
    print(f"   Min results: {metrics.get('min_results_count', 0)}")
    print(f"   Max results: {metrics.get('max_results_count', 0)}")
    
    if 'domain_accuracy' in metrics:
        print(f"\n🎯 Domain Accuracy:")
        print(f"   Domain match rate: {metrics.get('domain_accuracy', 0):.2%}")
    
    if 'avg_similarity' in metrics:
        print(f"\n🔍 Similarity Scores:")
        print(f"   Average: {metrics.get('avg_similarity', 0):.3f}")
        print(f"   Min: {metrics.get('min_similarity', 0):.3f}")
        print(f"   Max: {metrics.get('max_similarity', 0):.3f}")
        print(f"   Std Dev: {metrics.get('similarity_std', 0):.3f}")
    
    print("\n" + "=" * 60)


def run_full_evaluation(retriever) -> Dict:
    """
    Run complete evaluation suite
    
    Args:
        retriever: SemanticRetriever instance
        
    Returns:
        Complete evaluation results
    """
    test_set = create_test_set()
    
    # Basic retrieval evaluation
    retrieval_metrics = evaluate_retrieval_quality(retriever, test_set)
    
    # Diversity evaluation (on sample queries)
    from app.services.pipeline import retrieve_fuzzy_matches
    
    diversity_metrics_list = []
    relevance_metrics_list = []
    
    for test_case in test_set[:3]:  # Sample
        results = retrieve_fuzzy_matches(
            retriever=retriever,
            query=test_case['query'],
            domain=test_case.get('expected_domain'),
            top_k=7
        )
        
        diversity_metrics_list.append(evaluate_diversity(results))
        relevance_metrics_list.append(evaluate_relevance(test_case['query'], results))
    
    # Aggregate diversity metrics
    if diversity_metrics_list:
        avg_diversity = {
            "avg_uniqueness_ratio": np.mean([m.get('uniqueness_ratio', 0) for m in diversity_metrics_list]),
            "avg_domain_diversity": np.mean([m.get('domain_diversity', 0) for m in diversity_metrics_list])
        }
    else:
        avg_diversity = {}
    
    # Aggregate relevance metrics
    if relevance_metrics_list:
        avg_relevance = {
            "overall_relevance": np.mean([m.get('avg_relevance', 0) for m in relevance_metrics_list])
        }
    else:
        avg_relevance = {}
    
    return {
        "retrieval": retrieval_metrics,
        "diversity": avg_diversity,
        "relevance": avg_relevance
    }


if __name__ == "__main__":
    # Run evaluation
    logging.basicConfig(level=logging.INFO)
    
    try:
        from app.services.setup_qdrant import get_qdrant_client
        from app.services.retrieval_service import SemanticRetriever
        
        client = get_qdrant_client()
        retriever = SemanticRetriever(client)
        
        # Run evaluation
        test_set = create_test_set()
        metrics = evaluate_retrieval_quality(retriever, test_set)
        
        print_evaluation_report(metrics)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error running evaluation: {e}")
