"""
Optimized Glossary Service - Production Grade

Performance Optimizations:
1. Connection pooling for SQLite
2. Multi-layer caching (L1: LRU, L2: Redis)
3. Async-compatible operations
4. Batch query optimization
5. Smart n-gram filtering

Resource Management:
- Bounded memory usage with LRU eviction
- Connection recycling
- Graceful degradation when Redis unavailable

Time Complexity:
- Cache hit: O(1)
- Database query: O(log n) with FTS5 index
- N-gram generation: O(n) where n = tokens
"""

import time
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from app.core.connection_pool import get_pooled_connection, get_connection_pool
from app.core.lru_cache import get_glossary_cache, GlossaryTermCache, memoize
from app.utils.text_processor import normalize_text, tokenize, generate_ngrams
from app.models.schemas import GlossaryMatch
from app.services.cache_service import CacheService
from app.core.config import settings

logger = logging.getLogger(__name__)

# Thread pool for async database operations
_db_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db_worker")


@dataclass
class LookupMetrics:
    """Metrics for a single glossary lookup."""
    preprocessing_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    db_query_ms: float = 0.0
    post_processing_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit: bool = False
    terms_found: int = 0
    ngrams_generated: int = 0


class OptimizedGlossaryService:
    """
    Production-ready glossary lookup service with extreme optimization.
    
    Architecture:
    - Multi-layer caching (L1 LRU + L2 Redis)
    - Connection pooling for database
    - Async-compatible design
    - Batch query optimization
    
    Performance Characteristics:
    - L1 cache hit: ~0.1ms
    - L2 cache hit: ~2ms  
    - Database query: ~10-50ms
    """
    
    # Configurable limits
    MAX_NGRAM_SIZE = 10
    MIN_NGRAM_SIZE = 1
    MAX_TERMS_PER_QUERY = 1000
    MAX_MATCHES_RETURNED = 50
    
    def __init__(self):
        """Initialize service with optimized components."""
        self.redis_cache = CacheService()
        self.term_cache = get_glossary_cache()
        
        # Set Redis client for L2 cache
        if self.redis_cache.is_connected:
            self.term_cache.set_redis_client(self.redis_cache.redis)
        
        logger.info("OptimizedGlossaryService initialized")
    
    def process_request(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Synchronous entry point for single sentence processing.
        
        Optimizations:
        1. Early return on cache hit
        2. Efficient n-gram generation with size filtering
        3. Batch database queries
        4. Smart overlap removal
        
        Args:
            text: Source sentence
            src_lang: Source language code
            tgt_lang: Target language code
            domain: Glossary domain
            
        Returns:
            Response dict with matches and metrics
        """
        metrics = LookupMetrics()
        start_time = time.time()
        
        # =====================================================
        # PHASE 1: Preprocessing (O(n) where n = characters)
        # =====================================================
        preprocess_start = time.time()
        
        clean_text = normalize_text(text, src_lang)
        tokens = tokenize(clean_text, src_lang)
        
        # Enforce text length limit
        if len(tokens) > settings.MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated: {len(tokens)} -> {settings.MAX_TEXT_LENGTH} tokens")
            tokens = tokens[:settings.MAX_TEXT_LENGTH]
        
        metrics.preprocessing_ms = (time.time() - preprocess_start) * 1000
        
        # Early return for empty input
        if not tokens:
            return self._build_response([], text, domain, metrics, start_time)
        
        # =====================================================
        # PHASE 2: N-gram Generation (O(n * m) where m = max_ngram)
        # =====================================================
        ngram_data = generate_ngrams(tokens, max_n=self.MAX_NGRAM_SIZE)
        
        if not ngram_data:
            return self._build_response([], text, domain, metrics, start_time)
        
        # Extract unique terms, limit for performance
        ngram_texts = list(set(item['text'] for item in ngram_data))[:self.MAX_TERMS_PER_QUERY]
        metrics.ngrams_generated = len(ngram_texts)
        
        # =====================================================
        # PHASE 3: Cache Lookup (O(1))
        # =====================================================
        cache_start = time.time()
        
        cached_matches = self.term_cache.get(domain, src_lang, tgt_lang, ngram_texts)
        
        if cached_matches is not None:
            metrics.cache_hit = True
            metrics.cache_lookup_ms = (time.time() - cache_start) * 1000
            metrics.terms_found = len(cached_matches)
            
            final_matches = self._convert_to_glossary_matches(cached_matches)
            return self._build_response(final_matches, text, domain, metrics, start_time)
        
        metrics.cache_lookup_ms = (time.time() - cache_start) * 1000
        
        # =====================================================
        # PHASE 4: Database Query (O(log n) with FTS5)
        # =====================================================
        db_start = time.time()
        
        matched_rows = self._query_database_optimized(ngram_texts, src_lang, tgt_lang, domain)
        
        metrics.db_query_ms = (time.time() - db_start) * 1000
        
        # =====================================================
        # PHASE 5: Post-processing with Overlap Removal
        # =====================================================
        postprocess_start = time.time()
        
        matches = self._process_matches_optimized(matched_rows, tokens)
        
        # Convert to serializable format for caching
        matches_list = [
            {
                'source_term': m.source_term,
                'target_term': m.target_term,
                'n_gram_size': m.n_gram_size
            }
            for m in matches
        ]
        
        # Cache results (write-through)
        self.term_cache.put(domain, src_lang, tgt_lang, ngram_texts, matches_list)
        
        metrics.post_processing_ms = (time.time() - postprocess_start) * 1000
        metrics.terms_found = len(matches)
        
        # Convert final matches
        final_matches = self._convert_to_glossary_matches(matches_list)
        
        return self._build_response(final_matches, text, domain, metrics, start_time)
    
    async def process_request_async(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Async entry point for non-blocking operation.
        
        Runs database operations in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _db_executor,
            self.process_request,
            text, src_lang, tgt_lang, domain
        )
    
    def _query_database_optimized(
        self, 
        terms: List[str], 
        src: str, 
        tgt: str, 
        domain: str,
        allow_reverse: bool = True
    ) -> List[dict]:
        """
        Optimized database query with connection pooling and bidirectional support.
        
        Improvements:
        - Uses connection pool instead of creating new connections
        - Optimized FTS5 query construction
        - Adaptive batch sizing based on term count
        - Bidirectional lookup: if no matches in forward direction, try reverse
        
        Args:
            terms: N-gram terms to search
            src: Source language
            tgt: Target language
            domain: Domain filter
            allow_reverse: Whether to try reverse lookup if forward fails
            
        Returns:
            List of matching rows, each with 'is_reverse' flag
        """
        if not terms:
            return []
        
        all_rows = []
        
        try:
            with get_pooled_connection() as conn:
                cursor = conn.cursor()
                
                # Escape FTS5 special characters and build query
                escaped_terms = []
                for term in terms:
                    escaped = term.replace('"', '""')
                    escaped_terms.append(f'"{escaped}"')
                
                # Adaptive batch sizing: smaller batches for more terms
                batch_size = max(20, 100 - len(terms))
                
                # Forward query
                for i in range(0, len(escaped_terms), batch_size):
                    batch = escaped_terms[i:i + batch_size]
                    fts_query = ' OR '.join(batch)
                    
                    query = """
                        SELECT g.source_term, g.target_term, g.n_gram_size
                        FROM glossary_terms g
                        INNER JOIN glossary_fts fts ON g.id = fts.rowid
                        WHERE g.domain = ? 
                        AND g.source_lang = ? 
                        AND g.target_lang = ?
                        AND fts.source_term MATCH ?
                        ORDER BY g.n_gram_size DESC
                        LIMIT 100
                    """
                    
                    cursor.execute(query, [domain, src, tgt, fts_query])
                    rows = cursor.fetchall()
                    for row in rows:
                        all_rows.append(dict(row) | {'is_reverse': False})
                
                # If no results and allow_reverse, try reverse
                if not all_rows and allow_reverse:
                    for i in range(0, len(escaped_terms), batch_size):
                        batch = escaped_terms[i:i + batch_size]
                        fts_query = ' OR '.join(batch)
                        
                        reverse_query = """
                            SELECT g.source_term, g.target_term, g.n_gram_size
                            FROM glossary_terms g
                            INNER JOIN glossary_fts_target fts ON g.id = fts.rowid
                            WHERE g.domain = ? 
                            AND g.source_lang = ? 
                            AND g.target_lang = ?
                            AND fts.target_term MATCH ?
                            ORDER BY g.n_gram_size DESC
                            LIMIT 100
                        """
                        
                        cursor.execute(reverse_query, [domain, tgt, src, fts_query])
                        rows = cursor.fetchall()
                        for row in rows:
                            all_rows.append(dict(row) | {'is_reverse': True})
                
        except FileNotFoundError:
            logger.error("Database file not found")
        except Exception as e:
            logger.error(f"Database query failed: {e}")
        
        return all_rows
    
    def _process_matches_optimized(
        self, 
        db_rows: List[Any], 
        tokens: List[str]
    ) -> List[GlossaryMatch]:
        """
        Optimized overlap removal with set-based tracking.
        
        Algorithm:
        1. Sort by n-gram size (DESC)
        2. Use set for O(1) overlap checking
        3. Greedy selection of non-overlapping terms
        
        Time Complexity: O(m * k) where m = matches, k = avg term size
        Space Complexity: O(n) where n = tokens
        """
        if not db_rows:
            return []
        
        # Convert to dicts if not already
        matches = [dict(row) if not isinstance(row, dict) else row for row in db_rows]
        
        # Sort by size (DESC) 
        matches.sort(key=lambda x: x['n_gram_size'], reverse=True)
        
        # Build token position index for O(1) lookup
        token_str = " ".join(tokens).lower()
        
        selected = []
        covered_positions: Set[int] = set()
        
        for match in matches:
            if len(selected) >= self.MAX_MATCHES_RETURNED:
                break
            
            # For reverse matches, the term to match is target_term (since input is in target_lang for reverse)
            term = match['target_term'].lower() if match.get('is_reverse') else match['source_term'].lower()
            term_len = match['n_gram_size']
            
            # Find term position in token string
            pos = token_str.find(term)
            if pos == -1:
                continue
            
            # Calculate token indices covered by this match
            # Approximate: each token position ~ len(term) / term_len
            start_idx = token_str[:pos].count(' ')
            end_idx = start_idx + term_len
            
            term_indices = set(range(start_idx, end_idx))
            
            # Check for overlap with already selected terms
            if not term_indices.intersection(covered_positions):
                # Swap terms if reverse match
                if match.get('is_reverse'):
                    source_term = match['target_term']
                    target_term = match['source_term']
                else:
                    source_term = match['source_term']
                    target_term = match['target_term']
                
                selected.append(GlossaryMatch(
                    source_term=source_term,
                    target_term=target_term,
                    n_gram_size=match['n_gram_size']
                ))
                covered_positions.update(term_indices)
        
        return selected
    
    def _convert_to_glossary_matches(
        self, 
        matches_list: List[Dict]
    ) -> List[GlossaryMatch]:
        """Convert dict matches to GlossaryMatch objects."""
        return [
            GlossaryMatch(
                source_term=m['source_term'],
                target_term=m['target_term'],
                n_gram_size=m.get('n_gram_size', 1)
            )
            for m in matches_list
        ]
    
    def _build_response(
        self, 
        matches: List[GlossaryMatch], 
        text: str, 
        domain: str, 
        metrics: LookupMetrics,
        start_time: float
    ) -> Dict[str, Any]:
        """Build standardized response with metrics."""
        metrics.total_ms = (time.time() - start_time) * 1000
        
        return {
            "glossary_matches": matches,
            "match_count": len(matches),
            "source_sentence": text,
            "domain": domain,
            "processing_time_ms": round(metrics.total_ms, 2),
            "metrics": {
                "preprocessing_ms": round(metrics.preprocessing_ms, 2),
                "cache_lookup_ms": round(metrics.cache_lookup_ms, 2),
                "cache_hit": metrics.cache_hit,
                "db_query_ms": round(metrics.db_query_ms, 2),
                "post_processing_ms": round(metrics.post_processing_ms, 2),
                "ngrams_generated": metrics.ngrams_generated,
                "terms_found": metrics.terms_found
            }
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        pool_stats = get_connection_pool().get_stats()
        cache_stats = self.term_cache.get_stats()
        
        return {
            "connection_pool": pool_stats,
            "term_cache": cache_stats,
            "redis_connected": self.redis_cache.is_connected
        }
    
    def invalidate_domain_cache(self, domain: str) -> int:
        """Invalidate all cached entries for a domain."""
        return self.term_cache.invalidate_domain(domain)


# Backward compatible alias
GlossaryService = OptimizedGlossaryService
