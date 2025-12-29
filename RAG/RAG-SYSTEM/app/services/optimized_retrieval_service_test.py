import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
from functools import partial

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.optimized_embedding_service import (
    get_embedding_service,
    generate_single_embedding
)

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for fault tolerance."""
    failures: int = 0
    last_failure: float = 0.0
    state: str = "closed"  # closed, open, half-open
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= 5:
            self.state = "open"
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if (time.time() - self.last_failure) > 30:
                self.state = "half-open"
                return True
            return False
        return True  # half-open


class ResultCache:
    """LRU cache for search results."""
    
    def __init__(self, max_size: int = 5000, ttl_seconds: int = 600):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def _generate_key(
        self,
        query: str,
        domain: str,
        src_lang: str,
        tgt_lang: str,
        top_k: int,
        search_type: str
    ) -> str:
        """Generate cache key."""
        key_data = f"{query}:{domain}:{src_lang}:{tgt_lang}:{top_k}:{search_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        domain: str,
        src_lang: str,
        tgt_lang: str,
        top_k: int,
        search_type: str
    ) -> Optional[List[Dict]]:
        """Get cached results."""
        key = self._generate_key(query, domain, src_lang, tgt_lang, top_k, search_type)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry, created_at = self._cache[key]
            
            if (time.time() - created_at) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            self._cache.move_to_end(key)
            self._hits += 1
            return entry
    
    def put(
        self,
        query: str,
        domain: str,
        src_lang: str,
        tgt_lang: str,
        top_k: int,
        search_type: str,
        results: List[Dict]
    ) -> None:
        """Store results in cache."""
        key = self._generate_key(query, domain, src_lang, tgt_lang, top_k, search_type)
        
        with self._lock:
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (results, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2)
            }
    
    def clear(self) -> int:
        """Clear cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count


class OptimizedRetriever:
    """
    Production-grade semantic retrieval service.
    Features: RRF Fusion, Circuit Breakers, Async Caching.
    """
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = None,
        cache_size: int = 5000,
        cache_ttl: int = 600
    ):
        self.client = client
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self._embedding_service = get_embedding_service()
        
        self._cache = ResultCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._circuit_breaker = CircuitBreakerState()
        
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="retriever_worker"
        )
        
        self._total_searches = 0
        self._total_time_ms = 0.0
        logger.info(f"OptimizedRetriever initialized: collection={self.collection_name}")
    
    def search_semantic(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        min_score: float = None,
        use_cache: bool = True,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Semantic search with caching."""
        top_k = top_k or settings.DEFAULT_TOP_K
        min_score = min_score or settings.SIMILARITY_THRESHOLD
        
        if use_cache:
            cached = self._cache.get(
                query, domain or "", source_lang or "", 
                target_lang or "", top_k, "semantic"
            )
            if cached is not None:
                return cached
        
        if not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open, skipping search")
            return []
        
        start = time.time()
        
        try:
            if query_embedding is None:
                query_embedding = self._embedding_service.encode_single(query)
            
            search_filter = self._build_filter(domain, source_lang, target_lang)
            
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                using="cross_lingual",
                query_filter=search_filter,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True
            )
            
            results = self._format_results(response.points, "semantic")
            
            elapsed = (time.time() - start) * 1000
            self._total_searches += 1
            self._total_time_ms += elapsed
            
            if use_cache:
                self._cache.put(
                    query, domain or "", source_lang or "",
                    target_lang or "", top_k, "semantic", results
                )
            
            self._circuit_breaker.record_success()
            return results
            
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        semantic_weight: float = None,
        wording_weight: float = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        Fixes score distribution mismatches by relying on Rank rather than Score.
        """
        top_k = top_k or settings.DEFAULT_TOP_K
        
        if use_cache:
            cached = self._cache.get(
                query, domain or "", source_lang or "",
                target_lang or "", top_k, "hybrid"
            )
            if cached is not None:
                return cached
        
        query_embedding = self._embedding_service.encode_single(query)
        
        # Get more candidates for RRF intersection (at least 60)
        candidate_k = max(60, top_k * 2)
        
        # Search Semantic (min_score=0.0 to capture weak matches that might be strong in wording)
        semantic_results = self.search_semantic(
            query, candidate_k, domain, source_lang, target_lang,
            min_score=0.0, 
            use_cache=False,
            query_embedding=query_embedding
        )
        
        # Search Wording (min_score=None to capture keyword matches)
        wording_results = self._search_wording(
            query, candidate_k, domain, source_lang, target_lang,
            min_score=None,
            query_embedding=query_embedding
        )
        
        # Combine using RRF
        combined = self._combine_results_rrf(
            semantic_results, 
            wording_results,
            k_constant=60
        )
        
        # Apply domain boosting
        if domain:
            combined = self._apply_domain_boost(combined, domain)
            # Re-sort because boosting modifies scores
            combined = sorted(combined, key=lambda x: x['score'], reverse=True)
        
        results = combined[:top_k]
        
        if use_cache:
            self._cache.put(
                query, domain or "", source_lang or "",
                target_lang or "", top_k, "hybrid", results
            )
        
        return results
    
    def search_with_diversity(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        diversity_lambda: float = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """Search with MMR diversity re-ranking."""
        top_k = top_k or settings.DEFAULT_TOP_K
        diversity_lambda = diversity_lambda or settings.DIVERSITY_LAMBDA
        
        query_embedding = self._embedding_service.encode_single(query)
        
        # Use hybrid search for candidates
        candidates = self.search_hybrid(
            query, max(50, top_k * 10), domain, source_lang, target_lang,
            use_cache=False
        )
        
        if len(candidates) <= top_k:
            return candidates
        
        return self._mmr_rerank_optimized(
            query, candidates, top_k, diversity_lambda, 
            query_embedding=query_embedding
        )
    
    async def search_semantic_async(self, *args, **kwargs) -> List[Dict]:
        """Async semantic search."""
        loop = asyncio.get_event_loop()
        func = partial(self.search_semantic, *args, **kwargs)
        return await loop.run_in_executor(
            self._executor,
            func
        )
    
    async def search_hybrid_async(self, *args, **kwargs) -> List[Dict]:
        """Async hybrid search."""
        loop = asyncio.get_event_loop()
        func = partial(self.search_hybrid, *args, **kwargs)
        return await loop.run_in_executor(
            self._executor,
            func
        )
    
    def _search_wording(
        self,
        query: str,
        top_k: int,
        domain: str,
        source_lang: str,
        target_lang: str,
        query_embedding: Optional[np.ndarray] = None,
        min_score: float = None
    ) -> List[Dict]:
        """Search using source wording similarity."""
        try:
            if query_embedding is None:
                query_embedding = self._embedding_service.encode_single(query)
            
            search_filter = self._build_filter(domain, source_lang, target_lang)
            
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                using="source_semantic",
                query_filter=search_filter,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True
            )
            
            return self._format_results(response.points, "wording")
            
        except Exception as e:
            logger.error(f"Wording search failed: {e}")
            return []
    
    def _build_filter(
        self,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None
    ) -> Optional[Filter]:
        """Build Qdrant filter."""
        conditions = []
        
        if domain:
            conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        
        if source_lang:
            conditions.append(
                FieldCondition(key="source_lang", match=MatchValue(value=source_lang))
            )
        
        if target_lang:
            conditions.append(
                FieldCondition(key="target_lang", match=MatchValue(value=target_lang))
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _format_results(self, results: list, search_type: str) -> List[Dict]:
        """Format Qdrant results and preserve payload for testing."""
        formatted = []
        
        for i, result in enumerate(results):
            # Extract payload safely
            payload = result.payload or {}
            
            formatted.append({
                "rank": i + 1,
                "id": result.id,
                "score": float(result.score),
                "similarity_percentage": round(result.score * 100, 2),
                "search_type": search_type,
                # Essential fields
                "source": payload.get("source", ""),
                "target": payload.get("target", ""),
                "domain": payload.get("domain", ""),
                "language_pair": payload.get("language_pair", ""),
                "source_lang": payload.get("source_lang", ""),
                "target_lang": payload.get("target_lang", ""),
                # Pass full metadata for debugging/testing
                "metadata": payload
            })
        
        return formatted
    
    def _combine_results_rrf(
        self,
        semantic_results: List[Dict],
        wording_results: List[Dict],
        k_constant: int = 60,
        weight_semantic: float = 1.0,
        weight_wording: float = 1.0
    ) -> List[Dict]:
        """
        Combines results using Reciprocal Rank Fusion (RRF).
        Output includes 'score' (normalized 0-1) and 'rrf_score' (raw).
        """
        combined_dict = {}

        def process_results(results_list: List[Dict], weight: float, source_type: str):
            for rank, result in enumerate(results_list):
                # Convert ID to string to ensure matching works (UUID vs Int)
                doc_id = str(result['id']) 
                
                # RRF Formula: weight * (1 / (k + rank))
                rrf_score = weight * (1 / (k_constant + rank + 1))
                
                if doc_id not in combined_dict:
                    combined_dict[doc_id] = {
                        **result,
                        'id': result['id'], # Preserve original ID type
                        'rrf_score': 0.0,
                        'semantic_rank': None,
                        'wording_rank': None,
                        # Ensure metadata exists
                        'metadata': result.get('metadata', {})
                    }
                
                combined_dict[doc_id]['rrf_score'] += rrf_score
                combined_dict[doc_id][f'{source_type}_rank'] = rank + 1

        process_results(semantic_results, weight_semantic, "semantic")
        process_results(wording_results, weight_wording, "wording")

        results = list(combined_dict.values())
        results = sorted(results, key=lambda x: x['rrf_score'], reverse=True)

        # Normalize scores for UI (0.0 to 1.0)
        if results:
            max_score = results[0]['rrf_score']
            for res in results:
                # Avoid division by zero
                normalized = res['rrf_score'] / max_score if max_score > 0 else 0
                res['score'] = normalized
                res['similarity_percentage'] = round(normalized * 100, 2)
                res['search_type'] = 'hybrid_rrf'
                
                # Update metadata for debugging visibility
                if 'metadata' not in res or res['metadata'] is None:
                    res['metadata'] = {}
                
                res['metadata'].update({
                    'rrf_score': round(res['rrf_score'], 4),
                    'semantic_rank': res['semantic_rank'],
                    'wording_rank': res['wording_rank']
                })

        return results

    def _apply_domain_boost(
        self, 
        results: List[Dict], 
        target_domain: str
    ) -> List[Dict]:
        """
        Apply score boost to same-domain matches.
        Uses 'score' key (compatible with RRF).
        """
        for result in results:
            if result.get('domain') == target_domain:
                # Use 'score', NOT 'combined_score'
                current_score = result.get('score', 0.0)
                
                # Boost and Cap at 1.0
                boosted = min(current_score * settings.DOMAIN_BOOST_FACTOR, 1.0)
                
                result['score'] = boosted
                result['similarity_percentage'] = round(boosted * 100, 2)
                result['domain_boosted'] = True
            else:
                result['domain_boosted'] = False
        
        return results
    
    def _mmr_rerank_optimized(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        lambda_param: float,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Optimized MMR re-ranking."""
        if not candidates:
            return []
        
        if query_embedding is None:
            query_embedding = self._embedding_service.encode_single(query)
        
        sources = [c['source'] for c in candidates]
        candidate_embeddings = self._embedding_service.encode_batch(sources)
        
        selected_indices = []
        selected_embeddings = []
        
        # Normalize relevance scores to [0, 1] for MMR balance
        scores = [c['score'] for c in candidates]
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 1
        score_range = max_s - min_s if max_s > min_s else 1.0
        
        for _ in range(min(top_k, len(candidates))):
            best_idx = None
            best_score = -float('inf')
            
            for i in range(len(candidates)):
                if i in selected_indices:
                    continue
                
                # Normalized relevance
                relevance = (candidates[i]['score'] - min_s) / score_range
                
                if selected_embeddings:
                    selected_emb_array = np.array(selected_embeddings)
                    sims = cosine_similarity(
                        [candidate_embeddings[i]],
                        selected_emb_array
                    )[0]
                    max_sim = max(sims)
                else:
                    max_sim = 0
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
        
        results = []
        for rank, idx in enumerate(selected_indices):
            result = candidates[idx].copy()
            result['rank'] = rank + 1
            result['mmr_selected'] = True
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        avg_time = (
            self._total_time_ms / self._total_searches
            if self._total_searches > 0 else 0
        )
        
        try:
            info = self.client.get_collection(self.collection_name)
            collection_stats = {
                "points_count": info.points_count,
                "vectors_count": getattr(info, 'vectors_count', 0),
                "status": info.status.value if info.status else "unknown"
            }
        except Exception as e:
            collection_stats = {"error": str(e)}
        
        return {
            "collection": self.collection_name,
            "collection_stats": collection_stats,
            "total_searches": self._total_searches,
            "total_time_ms": round(self._total_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
            "cache": self._cache.get_stats(),
            "circuit_breaker": {
                "state": self._circuit_breaker.state,
                "failures": self._circuit_breaker.failures
            }
        }
    
    def clear_cache(self) -> int:
        """Clear result cache."""
        return self._cache.clear()


# Backward compatible alias
SemanticRetriever = OptimizedRetriever