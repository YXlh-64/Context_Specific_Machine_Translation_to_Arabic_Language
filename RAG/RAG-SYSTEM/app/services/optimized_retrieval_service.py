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
    """Production-grade semantic retrieval service."""
    
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
        Hybrid search combining semantic and wording similarity.
        
        Uses weighted combination of scores for better precision than RRF 
        when using the same embedding model for both vectors.
        """
        top_k = top_k or settings.DEFAULT_TOP_K
        semantic_weight = semantic_weight or settings.SEMANTIC_WEIGHT
        wording_weight = wording_weight or settings.WORDING_WEIGHT
        
        if use_cache:
            cached = self._cache.get(
                query, domain or "", source_lang or "",
                target_lang or "", top_k, "hybrid"
            )
            if cached is not None:
                return cached
        
        query_embedding = self._embedding_service.encode_single(query)
        
        # Get more candidates for re-ranking
        candidate_k = max(100, top_k * 10)
        
        # Search Semantic (No threshold for candidates to ensure we get enough for hybrid)
        semantic_results = self.search_semantic(
            query, candidate_k, domain, source_lang, target_lang,
            min_score=None, 
            use_cache=False,
            query_embedding=query_embedding
        )
        
        # Search Wording (No threshold for candidates)
        wording_results = self._search_wording(
            query, candidate_k, domain, source_lang, target_lang,
            min_score=None,
            query_embedding=query_embedding
        )
        
        # Combine and re-rank using weighted average
        combined = self._combine_results(
            semantic_results, 
            wording_results,
            semantic_weight,
            wording_weight
        )
        
        # Apply domain boosting
        if domain:
            combined = self._apply_domain_boost(combined, domain)
        
        # Sort and limit
        combined = sorted(combined, key=lambda x: x['combined_score'], reverse=True)
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
        
        # Use the improved hybrid search for candidates
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
                score_threshold=min_score, # Now accepts None
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
        """Format Qdrant results."""
        formatted = []
        
        for i, result in enumerate(results):
            formatted.append({
                "rank": i + 1,
                "id": result.id,
                "score": float(result.score),
                "similarity_percentage": round(result.score * 100, 2),
                "search_type": search_type,
                "source": result.payload.get("source", ""),
                "target": result.payload.get("target", ""),
                "domain": result.payload.get("domain", ""),
                "language_pair": result.payload.get("language_pair", ""),
                "source_lang": result.payload.get("source_lang", ""),
                "target_lang": result.payload.get("target_lang", ""),
                "source_length": result.payload.get("source_length", 0),
                "target_length": result.payload.get("target_length", 0)
            })
        
        return formatted
    
    def _combine_results(
        self,
        semantic_results: List[Dict],
        wording_results: List[Dict],
        semantic_weight: float,
        wording_weight: float
    ) -> List[Dict]:
        """
        Combine semantic and wording results using weighted average.
        
        Handles missing candidates by using a floor score.
        """
        combined_dict = {}
        
        # Get min scores for floor calculation
        min_semantic = min([r['score'] for r in semantic_results]) if semantic_results else 0.0
        min_wording = min([r['score'] for r in wording_results]) if wording_results else 0.0
        
        # Process Semantic
        for result in semantic_results:
            item_id = result['id']
            combined_dict[item_id] = {
                **result,
                'semantic_score': result['score'],
                'wording_score': min_wording * 0.5 # Conservative floor
            }
            
        # Process Wording
        for result in wording_results:
            item_id = result['id']
            if item_id in combined_dict:
                combined_dict[item_id]['wording_score'] = result['score']
            else:
                combined_dict[item_id] = {
                    **result,
                    'semantic_score': min_semantic * 0.5, # Conservative floor
                    'wording_score': result['score']
                }
                
        # Calculate final scores
        results = []
        for item in combined_dict.values():
            combined_score = (
                semantic_weight * item['semantic_score'] +
                wording_weight * item['wording_score']
            )
            item['combined_score'] = combined_score
            item['score'] = combined_score
            item['similarity_percentage'] = round(combined_score * 100, 2)
            item['search_type'] = 'hybrid'
            results.append(item)
            
        return results

    def _apply_domain_boost(
        self, 
        results: List[Dict], 
        target_domain: str
    ) -> List[Dict]:
        """Apply score boost to same-domain matches."""
        for result in results:
            if result.get('domain') == target_domain:
                result['combined_score'] *= settings.DOMAIN_BOOST_FACTOR
                result['score'] = result['combined_score']
                result['similarity_percentage'] = round(result['combined_score'] * 100, 2)
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
    
    def get_all_pairs_by_domain(self, domain: str = None) -> List[Dict]:
        """
        Get all translation pairs for a specific domain.
        
        Uses Qdrant scroll API to retrieve all points with optional domain filter.
        Returns formatted results similar to search results.
        """
        try:
            search_filter = self._build_filter(domain=domain)
            
            all_points = []
            next_page_offset = None
            
            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=search_filter,
                    limit=1000,  # Batch size
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False  # We don't need vectors for this
                )
                
                points = response[0]  # points
                next_page_offset = response[1]  # next_page_offset
                
                all_points.extend(points)
                
                # If no more points, break
                if next_page_offset is None or len(points) == 0:
                    break
            
            # Format results similar to search results
            formatted_results = []
            for i, point in enumerate(all_points):
                formatted_results.append({
                    "id": point.id,
                    "source": point.payload.get("source", ""),
                    "target": point.payload.get("target", ""),
                    "domain": point.payload.get("domain", ""),
                    "language_pair": point.payload.get("language_pair", ""),
                    "source_lang": point.payload.get("source_lang", ""),
                    "target_lang": point.payload.get("target_lang", ""),
                    "source_length": point.payload.get("source_length", 0),
                    "target_length": point.payload.get("target_length", 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get pairs by domain: {e}")
            return []
    
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