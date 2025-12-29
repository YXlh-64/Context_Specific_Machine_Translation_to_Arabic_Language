"""
Embedding Generation Module
Generate LaBSE embeddings for translation pairs
"""

import logging
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Tuple, Optional
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global model instance (singleton pattern)
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Get or initialize the LaBSE model (singleton)"""
    global _model
    
    if _model is None:
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        _model = SentenceTransformer(settings.MODEL_NAME, device=device)
        _model.eval()
        
        logger.info("Model loaded successfully")
    
    return _model


def generate_embeddings_batch(
    sentences: List[str],
    model: SentenceTransformer = None,
    batch_size: int = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for list of sentences in batches
    
    Args:
        sentences: List of text strings
        model: SentenceTransformer model (uses global if None)
        batch_size: Number of sentences per batch
        show_progress: Whether to show progress bar
        
    Returns:
        numpy array of shape (num_sentences, 768)
    """
    if model is None:
        model = get_model()
    
    if batch_size is None:
        batch_size = settings.BATCH_SIZE
    
    embeddings = []
    
    iterator = range(0, len(sentences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings")
    
    for i in iterator:
        batch = sentences[i:i+batch_size]
        
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def generate_single_embedding(
    text: str,
    model: SentenceTransformer = None
) -> np.ndarray:
    """Generate embedding for a single text"""
    if model is None:
        model = get_model()
    
    with torch.no_grad():
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    
    return embedding


def generate_all_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer = None,
    batch_size: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate all three types of embeddings for translation data
    
    Args:
        df: DataFrame with 'source' and 'target' columns
        model: SentenceTransformer model
        batch_size: Batch size for encoding
    
    Returns:
        tuple: (source_embeddings, target_embeddings, cross_lingual_embeddings)
    """
    if model is None:
        model = get_model()
    
    if batch_size is None:
        batch_size = settings.BATCH_SIZE
    
    # 1. Source embeddings
    logger.info("1/3 Generating source embeddings...")
    source_embeddings = generate_embeddings_batch(
        df['source'].tolist(),
        model,
        batch_size
    )
    
    # 2. Target embeddings
    logger.info("2/3 Generating target embeddings...")
    target_embeddings = generate_embeddings_batch(
        df['target'].tolist(),
        model,
        batch_size
    )
    
    # 3. Cross-lingual embeddings (average of source + target)
    logger.info("3/3 Generating cross-lingual embeddings...")
    cross_lingual_embeddings = (source_embeddings + target_embeddings) / 2
    cross_lingual_embeddings = normalize(cross_lingual_embeddings, axis=1)
    
    # Verify shapes
    assert source_embeddings.shape == (len(df), settings.EMBEDDING_DIM)
    assert target_embeddings.shape == (len(df), settings.EMBEDDING_DIM)
    assert cross_lingual_embeddings.shape == (len(df), settings.EMBEDDING_DIM)
    
    logger.info(f"Embeddings generated:")
    logger.info(f"  Source: {source_embeddings.shape}")
    logger.info(f"  Target: {target_embeddings.shape}")
    logger.info(f"  Cross-lingual: {cross_lingual_embeddings.shape}")
    
    return source_embeddings, target_embeddings, cross_lingual_embeddings


def compute_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """Compute cosine similarity between two embeddings"""
    # Embeddings are already normalized, so dot product = cosine similarity
    return float(np.dot(embedding1, embedding2))


def compute_batch_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray
) -> np.ndarray:
    """Compute similarity between query and all corpus embeddings"""
    # Matrix multiplication for batch similarity
    return np.dot(corpus_embeddings, query_embedding)


def verify_model() -> dict:
    """Verify model is working correctly"""
    model = get_model()
    
    # Test with sample sentences
    test_sentences = [
        "Hello world",
        "مرحبا بالعالم",  # Arabic: Hello world
        "Patients with severe symptoms require immediate care"
    ]
    
    embeddings = generate_embeddings_batch(test_sentences, model, show_progress=False)
    
    # Check shape
    assert embeddings.shape == (3, settings.EMBEDDING_DIM), f"Unexpected shape: {embeddings.shape}"
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), f"Embeddings not normalized: {norms}"
    
    # Check similarity (Hello world in English and Arabic should be similar)
    similarity_en_ar = compute_similarity(embeddings[0], embeddings[1])
    
    return {
        "status": "ok",
        "model": settings.MODEL_NAME,
        "embedding_dim": settings.EMBEDDING_DIM,
        "device": str(model.device),
        "test_similarity_en_ar": float(similarity_en_ar)
    }


if __name__ == "__main__":
    # Test embedding generation
    logging.basicConfig(level=logging.INFO)
    
    result = verify_model()
    print(f"\nModel verification:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test single embedding
    text = "Patients with severe symptoms require immediate care"
    embedding = generate_single_embedding(text)
    print(f"\nSingle embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
