"""
Setup Script for RAG System
Initialize the system with translation data
"""

import logging
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.data_loader import load_translation_data, get_domain_statistics
from app.services.setup_qdrant import get_qdrant_client, create_collection, verify_qdrant_connection
from app.services.embedding_service import get_model, generate_all_embeddings, verify_model
from app.services.upload_service import upload_to_qdrant, get_upload_progress

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_phase2_system(
    data_dir: str = None,
    collection_name: str = None,
    recreate: bool = False,
    batch_size: int = 32
) -> dict:
    """
    Complete setup of Phase 2 RAG system
    
    Steps:
    1. Verify Qdrant connection
    2. Load and clean translation data
    3. Create/verify Qdrant collection
    4. Generate embeddings
    5. Upload to Qdrant
    
    Args:
        data_dir: Directory containing translation CSVs
        collection_name: Qdrant collection name
        recreate: If True, delete and recreate collection
        batch_size: Batch size for embedding generation
        
    Returns:
        Setup status and statistics
    """
    logger.info("=" * 60)
    logger.info("Starting Phase 2 RAG System Setup")
    logger.info("=" * 60)
    
    results = {
        "status": "incomplete",
        "steps_completed": [],
        "errors": []
    }
    
    try:
        # Step 1: Verify Qdrant connection
        logger.info("\n[1/5] Verifying Qdrant connection...")
        client = get_qdrant_client()
        
        if not verify_qdrant_connection(client):
            results["errors"].append("Qdrant connection failed")
            logger.error("❌ Qdrant connection failed!")
            return results
        
        logger.info("✓ Qdrant connection successful")
        results["steps_completed"].append("qdrant_connection")
        
        # Step 2: Load translation data
        logger.info("\n[2/5] Loading translation data...")
        
        if data_dir:
            df = load_translation_data(data_dir)
        else:
            df = load_translation_data()
        
        if len(df) == 0:
            results["errors"].append("No translation data loaded")
            logger.error("❌ No translation data found!")
            return results
        
        logger.info(f"✓ Loaded {len(df)} translation pairs")
        results["steps_completed"].append("data_loading")
        results["translation_pairs"] = len(df)
        results["domain_stats"] = get_domain_statistics(df)
        
        # Step 3: Create/verify collection
        logger.info("\n[3/5] Setting up Qdrant collection...")
        
        if collection_name is None:
            collection_name = settings.QDRANT_COLLECTION
        
        success = create_collection(client, collection_name, recreate=recreate)
        
        if not success:
            results["errors"].append("Collection creation failed")
            logger.error("❌ Collection setup failed!")
            return results
        
        logger.info(f"✓ Collection '{collection_name}' ready")
        results["steps_completed"].append("collection_setup")
        results["collection_name"] = collection_name
        
        # Step 4: Generate embeddings
        logger.info("\n[4/5] Generating embeddings (this may take a while)...")
        
        model = get_model()
        model_info = verify_model()
        logger.info(f"Using model: {model_info.get('model', 'unknown')}")
        logger.info(f"Device: {model_info.get('device', 'unknown')}")
        
        source_emb, target_emb, cross_emb = generate_all_embeddings(
            df, model, batch_size
        )
        
        logger.info(f"✓ Generated embeddings: {source_emb.shape}")
        results["steps_completed"].append("embedding_generation")
        results["embedding_shape"] = list(source_emb.shape)
        
        # Step 5: Upload to Qdrant
        logger.info("\n[5/5] Uploading to Qdrant...")
        
        upload_to_qdrant(
            client=client,
            df=df,
            source_embeddings=source_emb,
            target_embeddings=target_emb,
            cross_lingual_embeddings=cross_emb,
            collection_name=collection_name,
            batch_size=100
        )
        
        progress = get_upload_progress(client, collection_name)
        logger.info(f"✓ Upload complete: {progress.get('points_count', 0)} points")
        results["steps_completed"].append("data_upload")
        results["upload_status"] = progress
        
        # Complete!
        results["status"] = "complete"
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Phase 2 RAG System Setup Complete!")
        logger.info("=" * 60)
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Total points: {progress.get('points_count', 0)}")
        logger.info(f"  Vectors per point: 3 (source, target, cross-lingual)")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        results["errors"].append(str(e))
        return results


def verify_setup(collection_name: str = None) -> dict:
    """
    Verify that the system is set up correctly
    
    Args:
        collection_name: Collection to verify
        
    Returns:
        Verification status
    """
    logger.info("Verifying RAG system setup...")
    
    results = {
        "qdrant": False,
        "collection": False,
        "model": False,
        "data": False
    }
    
    try:
        # Check Qdrant
        client = get_qdrant_client()
        results["qdrant"] = verify_qdrant_connection(client)
        
        # Check collection
        if collection_name is None:
            collection_name = settings.QDRANT_COLLECTION
        
        info = client.get_collection(collection_name)
        results["collection"] = info.points_count > 0
        results["collection_info"] = {
            "name": collection_name,
            "points": info.points_count,
            "vectors": getattr(info, 'vectors_count', 0)
        }
        
        # Check model
        model_info = verify_model()
        results["model"] = model_info.get("status") == "ok"
        results["model_info"] = model_info
        
        # Check data
        results["data"] = info.points_count > 0
        
        # Overall status
        results["status"] = all([
            results["qdrant"],
            results["collection"],
            results["model"],
            results["data"]
        ])
        
        return results
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        results["error"] = str(e)
        return results


def main():
    """Main entry point for setup script"""
    parser = argparse.ArgumentParser(
        description="Setup RAG System for Semantic Translation Memory"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing translation CSV files"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        help="Qdrant collection name"
    )
    
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify setup, don't create"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        results = verify_setup(args.collection)
        
        print("\n" + "=" * 40)
        print("RAG System Verification")
        print("=" * 40)
        
        for key, value in results.items():
            if isinstance(value, bool):
                status = "✓" if value else "✗"
                print(f"  {key}: {status}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
        
        print("=" * 40)
        
        overall = results.get("status", False)
        print(f"\nOverall: {'✓ Ready' if overall else '✗ Not Ready'}")
        
        return 0 if overall else 1
    
    else:
        results = setup_phase2_system(
            data_dir=args.data_dir,
            collection_name=args.collection,
            recreate=args.recreate,
            batch_size=args.batch_size
        )
        
        if results["status"] == "complete":
            print("\n✓ Setup completed successfully!")
            return 0
        else:
            print(f"\n✗ Setup failed: {results.get('errors', [])}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
