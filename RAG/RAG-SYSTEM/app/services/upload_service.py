"""
Upload to Qdrant Module
Upload translation data with multi-vector embeddings to Qdrant
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.core.config import settings

logger = logging.getLogger(__name__)


def create_payload(row: pd.Series) -> dict:
    """Create rich metadata payload for each point"""
    return {
        "source": str(row['source']),
        "target": str(row['target']),
        "domain": str(row['domain']),
        "language_pair": str(row['language_pair']),
        "source_lang": str(row['source_lang']),
        "target_lang": str(row['target_lang']),
        "source_length": int(row['source_length']),
        "target_length": int(row['target_length']),
        "id": int(row['id'])
    }


def upload_to_qdrant(
    client: QdrantClient,
    df: pd.DataFrame,
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    cross_lingual_embeddings: np.ndarray,
    collection_name: str = None,
    batch_size: int = 100,
    show_progress: bool = True
) -> bool:
    """
    Upload translation data with multi-vector embeddings to Qdrant
    
    Args:
        client: Qdrant client
        df: DataFrame with translation data
        source_embeddings: Source sentence embeddings
        target_embeddings: Target sentence embeddings
        cross_lingual_embeddings: Cross-lingual embeddings
        collection_name: Target collection name
        batch_size: Upload batch size
        show_progress: Show progress bar
        
    Returns:
        True if upload successful
    """
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    logger.info(f"Uploading {len(df)} points to '{collection_name}'...")
    
    # Verify data consistency
    assert len(df) == len(source_embeddings), "DataFrame and source embeddings size mismatch"
    assert len(df) == len(target_embeddings), "DataFrame and target embeddings size mismatch"
    assert len(df) == len(cross_lingual_embeddings), "DataFrame and cross-lingual embeddings size mismatch"
    
    # Upload in batches
    points_uploaded = 0
    
    iterator = range(0, len(df), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Uploading to Qdrant")
    
    for i in iterator:
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end]
        
        points = []
        for j, (idx, row) in enumerate(batch_df.iterrows()):
            point_idx = i + j
            
            point = PointStruct(
                id=int(row['id']),
                vector={
                    "source_semantic": source_embeddings[point_idx].tolist(),
                    "target_semantic": target_embeddings[point_idx].tolist(),
                    "cross_lingual": cross_lingual_embeddings[point_idx].tolist()
                },
                payload=create_payload(row)
            )
            points.append(point)
        
        # Upload batch
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        
        points_uploaded += len(points)
    
    # Verify upload
    collection_info = client.get_collection(collection_name)
    
    if collection_info.points_count != len(df):
        logger.warning(f"Point count mismatch: expected {len(df)}, got {collection_info.points_count}")
    else:
        logger.info(f"Successfully uploaded {points_uploaded} points")
    
    return True


def upload_single_point(
    client: QdrantClient,
    point_id: int,
    source_text: str,
    target_text: str,
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    cross_lingual_embedding: np.ndarray,
    domain: str,
    language_pair: str,
    collection_name: str = None
) -> bool:
    """Upload a single translation pair to Qdrant"""
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    source_lang, target_lang = language_pair.split("-")
    
    point = PointStruct(
        id=point_id,
        vector={
            "source_semantic": source_embedding.tolist(),
            "target_semantic": target_embedding.tolist(),
            "cross_lingual": cross_lingual_embedding.tolist()
        },
        payload={
            "source": source_text,
            "target": target_text,
            "domain": domain,
            "language_pair": language_pair,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_length": len(source_text.split()),
            "target_length": len(target_text.split()),
            "id": point_id
        }
    )
    
    client.upsert(
        collection_name=collection_name,
        points=[point],
        wait=True
    )
    
    logger.debug(f"Uploaded point {point_id}")
    return True


def get_upload_progress(client: QdrantClient, collection_name: str = None) -> dict:
    """Get current upload progress/status"""
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    try:
        info = client.get_collection(collection_name)
        return {
            "collection": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if info.status else "unknown"
        }
    except Exception as e:
        logger.error(f"Failed to get upload progress: {e}")
        return {"error": str(e)}


def delete_points_by_domain(
    client: QdrantClient,
    domain: str,
    collection_name: str = None
) -> int:
    """Delete all points for a specific domain"""
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    result = client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="domain",
                    match=MatchValue(value=domain)
                )
            ]
        ),
        wait=True
    )
    
    logger.info(f"Deleted points for domain: {domain}")
    return result


if __name__ == "__main__":
    # Test upload functionality
    logging.basicConfig(level=logging.INFO)
    
    from app.services.setup_qdrant import get_qdrant_client
    
    client = get_qdrant_client()
    progress = get_upload_progress(client)
    print(f"Upload progress: {progress}")
