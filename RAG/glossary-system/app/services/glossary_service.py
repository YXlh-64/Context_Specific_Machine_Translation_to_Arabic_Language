"""
Glossary Service for Single Sentence Mode (Mode 1)

This service handles glossary lookups for individual sentences.
Implements standard lookup using SQLite/FTS5.

Flow:
1. Standard lookup: Normalize -> N-grams -> SQLite/FTS5 Query
2. Result Processing: Overlap removal and sorting
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional

from app.core.database import get_db_connection
from app.utils.text_processor import normalize_text, tokenize, generate_ngrams
from app.models.schemas import GlossaryMatch
from app.services.cache_service import CacheService
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class GlossaryService:
    """
    Production-ready glossary lookup service.
    
    Handles single sentence translations with:
    - Standard SQLite/FTS5 glossary lookup
    - Overlap removal algorithm
    """
    
    def __init__(self):
        """Initialize service with cache dependencies."""
        self.cache = CacheService()
    
    def process_request(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Main entry point for Single Sentence Mode.
        
        1. Standard glossary lookup
        2. Result Processing & Overlap Removal
        
        Args:
            text: Source sentence to process
            src_lang: Source language code
            tgt_lang: Target language code
            domain: Domain for glossary lookup
            
        Returns:
            Response dict with glossary_matches, match_count, etc.
        """
        start_time = time.time()
        
        # =====================================================
        # Glossary Lookup
        # =====================================================
        
        # 1. Preprocessing
        clean_text = normalize_text(text, src_lang)
        tokens = tokenize(clean_text, src_lang)
        
        if len(tokens) > settings.MAX_TEXT_LENGTH:
            logger.warning(f"Text exceeds max length ({len(tokens)} > {settings.MAX_TEXT_LENGTH})")
            tokens = tokens[:settings.MAX_TEXT_LENGTH]

        # 2. N-gram Generation
        ngram_data = generate_ngrams(tokens)
        if not ngram_data:
            return self._build_response([], text, domain, start_time)

        # Extract just the text strings for the SQL query
        ngram_texts = list(set([item['text'] for item in ngram_data]))
        
        # 3. Database Query
        matched_rows = self._query_database(ngram_texts, src_lang, tgt_lang, domain)
        
        # 4. Result Processing & Overlap Removal
        matches = self._process_matches(matched_rows, tokens)
        
        # Convert to list of dicts for agent compatibility
        matches_list = [
            {
                'source_term': m.source_term,
                'target_term': m.target_term,
                'n_gram_size': m.n_gram_size
            }
            for m in matches
        ]
        
        # 5. Build Response
        final_matches = [
            GlossaryMatch(
                source_term=m['source_term'],
                target_term=m['target_term'],
                n_gram_size=m['n_gram_size']
            )
            for m in matches_list
        ]
        
        return self._build_response(final_matches, text, domain, start_time)

    def _query_database(
        self, 
        terms: List[str], 
        src: str, 
        tgt: str, 
        domain: str
    ) -> List[dict]:
        """
        Query SQLite FTS5 with domain filter.
        
        Uses FTS5 MATCH for fast text search with JOIN to filter by domain/language.
        
        Args:
            terms: List of n-gram terms to search
            src: Source language code
            tgt: Target language code
            domain: Domain filter
            
        Returns:
            List of matching database rows
        """
        if not terms:
            return []
            
        all_rows = []
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build FTS5 match query - use OR for multiple terms
                # Escape special FTS5 characters and wrap terms in quotes
                escaped_terms = []
                for term in terms:
                    # Escape double quotes in terms
                    escaped = term.replace('"', '""')
                    escaped_terms.append(f'"{escaped}"')
                
                # Batch processing for large term sets
                batch_size = 50
                
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
                    """
                    
                    params = [domain, src, tgt, fts_query]
                    cursor.execute(query, params)
                    all_rows.extend(cursor.fetchall())
                    
        except FileNotFoundError as e:
            logger.error(f"Database not found: {str(e)}")
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
                
        return all_rows

    def _process_matches(
        self, 
        db_rows: List[Any], 
        tokens: List[str]
    ) -> List[GlossaryMatch]:
        """
        Process database results with overlap removal.
        
        Implements the overlap removal algorithm:
        - Sort by n-gram size (DESC) 
        - Track covered token indices
        - Skip terms that overlap with already selected terms
        
        Args:
            db_rows: Raw database query results
            tokens: Original tokenized text
            
        Returns:
            List of GlossaryMatch objects (max 5)
        """
        # Convert DB rows to dicts
        matches = [dict(row) for row in db_rows]
        
        # Sort by Size (DESC) then 
        matches.sort(key=lambda x: x['n_gram_size'], reverse=True)
        
        selected_terms = []
        covered_indices = set()
        
        for match in matches:
            term_len = match['n_gram_size']
            
            # Find where this specific term exists in original token list
            found_indices = []
            
            # Sliding window to find match indices in source tokens
            for i in range(len(tokens) - term_len + 1):
                # Compare token chunk (joined) with match term
                chunk = " ".join(tokens[i : i + term_len])
                if chunk == match['source_term']:
                    found_indices.append(set(range(i, i + term_len)))

            # Check for overlap
            for indices_set in found_indices:
                if not indices_set.intersection(covered_indices):
                    # No overlap, select this term
                    selected_terms.append(GlossaryMatch(
                        source_term=match['source_term'],
                        target_term=match['target_term'],
                        n_gram_size=match['n_gram_size']
                    ))
                    covered_indices.update(indices_set)
                    break  # Only select one instance per term
                    
            if len(selected_terms) >= 5:
                break
                
        return selected_terms[:5]

    def _build_response(
        self, 
        matches: List[GlossaryMatch], 
        text: str, 
        domain: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """
        Build standardized response dictionary.
        
        Args:
            matches: List of GlossaryMatch objects
            text: Original source text
            domain: Domain used for lookup
            start_time: Processing start timestamp
            
        Returns:
            Response dictionary
        """
        return {
            "glossary_matches": matches,
            "match_count": len(matches),
            "source_sentence": text,
            "domain": domain,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
