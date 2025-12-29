"""
Qdrant Setup Module
Create and configure Qdrant collection with multi-vector configuration
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PayloadSchemaType,
    HnswConfigDiff,
    OptimizersConfigDiff
)

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with proper configuration"""
    try:
        client = QdrantClient(**settings.qdrant_connection_params)
        logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


def create_collection(
    client: QdrantClient, 
    collection_name: str = None,
    recreate: bool = False
) -> bool:
    """
    Create Qdrant collection with multi-vector configuration
    
    Three vectors per point:
    1. source_semantic: Embedding of source sentence
    2. target_semantic: Embedding of target sentence  
    3. cross_lingual: Average of source+target (captures meaning)
    
    Args:
        client: Qdrant client instance
        collection_name: Name of collection to create
        recreate: If True, delete existing collection first
        
    Returns:
        True if collection created successfully
    """
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            return True
    
    # Create collection with 3 named vectors
    logger.info(f"Creating collection: {collection_name}")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Vector 1: Source sentence embedding
            "source_semantic": VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
                on_disk=False  # Keep in memory for speed
            ),
            # Vector 2: Target sentence embedding
            "target_semantic": VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
                on_disk=False
            ),
            # Vector 3: Cross-lingual embedding (average of source + target)
            "cross_lingual": VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
                on_disk=False
            )
        }
    )
    
    # Configure HNSW index for performance
    logger.info("Configuring HNSW index...")
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(
            m=settings.HNSW_M,  # Number of edges per node
            ef_construct=settings.HNSW_EF_CONSTRUCT  # Construction time accuracy
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=10000  # Start indexing after 10k points
        )
    )
    
    # Create payload indices for fast filtering
    create_payload_indices(client, collection_name)
    
    logger.info(f"Collection '{collection_name}' created successfully!")
    return True


def create_payload_indices(client: QdrantClient, collection_name: str):
    """Create indices for metadata filtering"""
    
    indices = [
        ("domain", PayloadSchemaType.KEYWORD),
        ("language_pair", PayloadSchemaType.KEYWORD),
        ("source_lang", PayloadSchemaType.KEYWORD),
        ("target_lang", PayloadSchemaType.KEYWORD),
        ("source_length", PayloadSchemaType.INTEGER),
        ("target_length", PayloadSchemaType.INTEGER),
    ]
    
    for field_name, field_type in indices:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            logger.info(f"Created index for: {field_name}")
        except Exception as e:
            # Index might already exist
            logger.debug(f"Index creation for {field_name}: {e}")


def get_collection_info(client: QdrantClient, collection_name: str = None) -> dict:
    """Get information about a collection"""
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": getattr(info, 'vectors_count', 0),
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if info.status else "unknown",
            "config": {
                "vector_size": settings.EMBEDDING_DIM,
                "distance": "cosine"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return None


def delete_collection(client: QdrantClient, collection_name: str = None) -> bool:
    """Delete a collection"""
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False


def verify_qdrant_connection(client: QdrantClient) -> bool:
    """Verify Qdrant is accessible"""
    try:
        collections = client.get_collections()
        logger.info(f"Qdrant connection verified. Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test Qdrant setup
    logging.basicConfig(level=logging.INFO)
    
    client = get_qdrant_client()
    
    if verify_qdrant_connection(client):
        print("✓ Qdrant connection successful")
        
        # Create collection
        create_collection(client, recreate=False)
        
        # Get info
        info = get_collection_info(client)
        if info:
            print(f"✓ Collection info: {info}")
    else:
        print("✗ Qdrant connection failed")
