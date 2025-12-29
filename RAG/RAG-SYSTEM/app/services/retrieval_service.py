"""
Semantic Retrieval Module
Advanced semantic retrieval system with multi-stage search
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, 
    FieldCondition, 
    MatchValue,
    Range
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.embedding_service import get_model, generate_single_embedding

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Advanced semantic retrieval system with multi-stage search"""
    
    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer = None,
        collection_name: str = None
    ):
        """
        Initialize the semantic retriever
        
        Args:
            client: Qdrant client instance
            model: SentenceTransformer model (optional, will load default)
            collection_name: Qdrant collection name
        """
        self.client = client
        self.model = model if model is not None else get_model()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        
        logger.info(f"SemanticRetriever initialized for collection: {self.collection_name}")
    
    def search_semantic(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        min_score: float = None
    ) -> List[Dict]:
        """
        Search using cross-lingual semantic embedding
        
        This finds translations with similar MEANING, regardless of wording.
        Best for: Finding conceptually similar translations.
        
        Args:
            query: Source sentence to find matches for
            top_k: Number of results to return
            domain: Filter by domain (optional)
            source_lang: Filter by source language
            target_lang: Filter by target language
            min_score: Minimum similarity score threshold
            
        Returns:
            List of matching translations with scores
        """
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K
        if min_score is None:
            min_score = settings.SIMILARITY_THRESHOLD
        
        # Generate query embedding
        query_embedding = generate_single_embedding(query, self.model)
        
        # Build filter
        search_filter = self._build_filter(domain, source_lang, target_lang)
        
        # Search using cross-lingual vector (new query_points API)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            using="cross_lingual",
            query_filter=search_filter,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True
        )
        
        return self._format_results(response.points, "semantic")
    
    def search_wording(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        min_score: float = None
    ) -> List[Dict]:
        """
        Search using source sentence embedding
        
        This finds translations with similar WORDING/phrasing.
        Best for: Finding translations with similar sentence structure.
        
        Args:
            query: Source sentence to find matches for
            top_k: Number of results to return
            domain: Filter by domain (optional)
            source_lang: Filter by source language
            target_lang: Filter by target language
            min_score: Minimum similarity score threshold
            
        Returns:
            List of matching translations with scores
        """
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K
        if min_score is None:
            min_score = settings.SIMILARITY_THRESHOLD
        
        # Generate query embedding
        query_embedding = generate_single_embedding(query, self.model)
        
        # Build filter
        search_filter = self._build_filter(domain, source_lang, target_lang)
        
        # Search using source_semantic vector (new query_points API)
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
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        semantic_weight: float = None,
        wording_weight: float = None,
        min_score: float = None
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and wording similarity
        
        This combines both meaning and wording matches for best results.
        Uses weighted combination of scores.
        
        Args:
            query: Source sentence to find matches for
            top_k: Number of results to return
            domain: Filter by domain (optional)
            source_lang: Filter by source language
            target_lang: Filter by target language
            semantic_weight: Weight for semantic scores (0-1)
            wording_weight: Weight for wording scores (0-1)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of matching translations with combined scores
        """
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K
        if semantic_weight is None:
            semantic_weight = settings.SEMANTIC_WEIGHT
        if wording_weight is None:
            wording_weight = settings.WORDING_WEIGHT
        if min_score is None:
            min_score = settings.SIMILARITY_THRESHOLD
        
        # Get more candidates for re-ranking
        candidate_k = top_k * 3
        
        # Get semantic matches
        semantic_results = self.search_semantic(
            query, candidate_k, domain, source_lang, target_lang, min_score * 0.8
        )
        
        # Get wording matches
        wording_results = self.search_wording(
            query, candidate_k, domain, source_lang, target_lang, min_score * 0.8
        )
        
        # Combine and re-rank
        combined = self._combine_results(
            semantic_results, 
            wording_results,
            semantic_weight,
            wording_weight
        )
        
        # Apply domain boosting
        if domain:
            combined = self._apply_domain_boost(combined, domain)
        
        # Filter by minimum score and return top_k
        combined = [r for r in combined if r['combined_score'] >= min_score]
        combined = sorted(combined, key=lambda x: x['combined_score'], reverse=True)
        
        return combined[:top_k]
    
    def search_with_diversity(
        self,
        query: str,
        top_k: int = None,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        diversity_lambda: float = None
    ) -> List[Dict]:
        """
        Search with MMR (Maximal Marginal Relevance) for diversity
        
        This ensures results are both relevant AND diverse.
        Avoids returning many near-duplicate results.
        
        Args:
            query: Source sentence to find matches for
            top_k: Number of results to return
            domain: Filter by domain (optional)
            source_lang: Filter by source language
            target_lang: Filter by target language
            diversity_lambda: Balance between relevance (1) and diversity (0)
            
        Returns:
            List of diverse matching translations
        """
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K
        if diversity_lambda is None:
            diversity_lambda = settings.DIVERSITY_LAMBDA
        
        # Get more candidates for diversity selection
        candidates = self.search_hybrid(
            query, 
            top_k * 5,  # Get 5x candidates
            domain, 
            source_lang, 
            target_lang
        )
        
        if len(candidates) <= top_k:
            return candidates
        
        # Apply MMR diversity re-ranking
        diverse_results = self._mmr_rerank(
            query,
            candidates,
            top_k,
            diversity_lambda
        )
        
        return diverse_results
    
    def _build_filter(
        self,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        min_length: int = None,
        max_length: int = None
    ) -> Optional[Filter]:
        """Build Qdrant filter from parameters"""
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
        
        if min_length is not None or max_length is not None:
            range_filter = {}
            if min_length is not None:
                range_filter["gte"] = min_length
            if max_length is not None:
                range_filter["lte"] = max_length
            conditions.append(
                FieldCondition(key="source_length", range=Range(**range_filter))
            )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def _format_results(self, results: list, search_type: str) -> List[Dict]:
        """Format Qdrant search results"""
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
        """Combine semantic and wording results with weighted scores"""
        # Create lookup by ID
        combined_dict = {}
        
        for result in semantic_results:
            result_id = result['id']
            combined_dict[result_id] = {
                **result,
                'semantic_score': result['score'],
                'wording_score': 0.0
            }
        
        for result in wording_results:
            result_id = result['id']
            if result_id in combined_dict:
                combined_dict[result_id]['wording_score'] = result['score']
            else:
                combined_dict[result_id] = {
                    **result,
                    'semantic_score': 0.0,
                    'wording_score': result['score']
                }
        
        # Calculate combined scores
        results = []
        for result_id, result in combined_dict.items():
            combined_score = (
                semantic_weight * result['semantic_score'] +
                wording_weight * result['wording_score']
            )
            result['combined_score'] = combined_score
            result['score'] = combined_score
            result['similarity_percentage'] = round(combined_score * 100, 2)
            result['search_type'] = 'hybrid'
            results.append(result)
        
        return results
    
    def _apply_domain_boost(self, results: List[Dict], target_domain: str) -> List[Dict]:
        """Apply score boost to same-domain matches"""
        for result in results:
            if result.get('domain') == target_domain:
                boost_factor = settings.DOMAIN_BOOST_FACTOR
                result['combined_score'] *= boost_factor
                result['score'] = result['combined_score']
                result['similarity_percentage'] = round(result['combined_score'] * 100, 2)
                result['domain_boosted'] = True
            else:
                result['domain_boosted'] = False
        
        return results
    
    def _mmr_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        lambda_param: float
    ) -> List[Dict]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity
        
        MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
        """
        if len(candidates) == 0:
            return []
        
        # Get query embedding
        query_embedding = generate_single_embedding(query, self.model)
        
        # Get embeddings for all candidates
        candidate_embeddings = []
        for candidate in candidates:
            # Generate embedding for source text
            emb = generate_single_embedding(candidate['source'], self.model)
            candidate_embeddings.append(emb)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Calculate similarity to query
        query_similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # MMR selection
        selected_indices = []
        selected_embeddings = []
        
        for _ in range(min(top_k, len(candidates))):
            best_idx = None
            best_score = -float('inf')
            
            for i, candidate in enumerate(candidates):
                if i in selected_indices:
                    continue
                
                # Relevance to query
                relevance = query_similarities[i]
                
                # Diversity: max similarity to already selected
                if selected_embeddings:
                    selected_emb_array = np.array(selected_embeddings)
                    similarities_to_selected = cosine_similarity(
                        [candidate_embeddings[i]], 
                        selected_emb_array
                    )[0]
                    max_sim_to_selected = max(similarities_to_selected)
                else:
                    max_sim_to_selected = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
        
        # Return selected candidates in order
        diverse_results = []
        for rank, idx in enumerate(selected_indices):
            result = candidates[idx].copy()
            result['rank'] = rank + 1
            result['mmr_selected'] = True
            diverse_results.append(result)
        
        return diverse_results
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": getattr(info, 'vectors_count', 0),
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value if info.status else "unknown"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Test the retriever
    logging.basicConfig(level=logging.INFO)
    
    from app.services.setup_qdrant import get_qdrant_client
    
    client = get_qdrant_client()
    retriever = SemanticRetriever(client)
    
    print("Retriever stats:", retriever.get_stats())
