"""
Build Vector Database from Parallel Corpora

This script processes the UN parallel corpus and builds a vector database
for semantic search and retrieval.

Usage:
    python build_vector_db.py --config config.ini
    python build_vector_db.py --corpus_file ../ATA/economic_v1.csv --db_type chromadb
"""

import argparse
import os
import sys
import pandas as pd
from loguru import logger

from vector_db import VectorDBManager
from utils import load_config, setup_logging, ensure_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Build vector database from parallel corpus"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        help="Path to corpus CSV file (overrides config)"
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="en",
        help="Name of source language column"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="ar",
        help="Name of target language column"
    )
    parser.add_argument(
        "--domain_column",
        type=str,
        help="Name of domain/category column"
    )
    parser.add_argument(
        "--db_type",
        type=str,
        choices=["chromadb", "faiss"],
        help="Vector database type (overrides config)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="Embedding model name (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--clear_existing",
        action="store_true",
        help="Clear existing vector database"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Use only a sample of N rows (for testing)"
    )
    
    return parser.parse_args()


def load_corpus(
    corpus_file: str,
    source_column: str,
    target_column: str,
    domain_column: str = None,
    sample: int = None
) -> pd.DataFrame:
    """
    Load parallel corpus from CSV file
    
    Args:
        corpus_file: Path to CSV file
        source_column: Name of source language column
        target_column: Name of target language column
        domain_column: Name of domain column
        sample: Number of rows to sample (None for all)
        
    Returns:
        DataFrame with parallel corpus
    """
    logger.info(f"Loading corpus from: {corpus_file}")
    
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    # Try to detect encoding and read CSV
    try:
        df = pd.read_csv(corpus_file, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, trying latin-1")
        df = pd.read_csv(corpus_file, encoding='latin-1')
    
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' not found in CSV")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")
    
    # Check domain column
    if domain_column and domain_column not in df.columns:
        logger.warning(f"Domain column '{domain_column}' not found, will use 'general'")
        domain_column = None
    
    # Remove rows with missing translations
    df = df.dropna(subset=[source_column, target_column])
    logger.info(f"After removing NaN: {len(df)} rows")
    
    # Remove empty strings
    df = df[df[source_column].str.strip().str.len() > 0]
    df = df[df[target_column].str.strip().str.len() > 0]
    logger.info(f"After removing empty strings: {len(df)} rows")
    
    # Sample if requested
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
        logger.info(f"Sampled {sample} rows")
    
    # Show statistics
    logger.info(f"\n{'='*50}")
    logger.info("Corpus Statistics:")
    logger.info(f"{'='*50}")
    logger.info(f"Total entries: {len(df)}")
    logger.info(f"Source column: {source_column}")
    logger.info(f"Target column: {target_column}")
    
    if domain_column and domain_column in df.columns:
        logger.info(f"\nDomain distribution:")
        domain_counts = df[domain_column].value_counts()
        for domain, count in domain_counts.items():
            logger.info(f"  {domain}: {count}")
    
    # Show sample entries
    logger.info(f"\n{'='*50}")
    logger.info("Sample entries:")
    logger.info(f"{'='*50}")
    for i, row in df.head(3).iterrows():
        logger.info(f"\nEntry {i}:")
        logger.info(f"  EN: {row[source_column][:100]}...")
        logger.info(f"  AR: {row[target_column][:100]}...")
        if domain_column and domain_column in df.columns:
            logger.info(f"  Domain: {row[domain_column]}")
    
    return df


def main():
    """Main function"""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logs_dir = config.get("GENERAL", "logs_dir", fallback="./logs")
    setup_logging(logs_dir)
    
    logger.info("Starting vector database build process")
    logger.info(f"Configuration file: {args.config}")
    
    # Get parameters from config or args
    corpus_file = args.corpus_file or config.get("DATA", "corpus_file")
    source_column = args.source_column or config.get("DATA", "source_column", fallback="en")
    target_column = args.target_column or config.get("DATA", "target_column", fallback="ar")
    domain_column = args.domain_column or config.get("DATA", "domain_column", fallback=None)
    
    db_type = args.db_type or config.get("VECTOR_DB", "db_type", fallback="chromadb")
    embedding_model = args.embedding_model or config.get("EMBEDDINGS", "model_name")
    db_path = config.get("GENERAL", "vector_db_dir", fallback="./vector_db")
    collection_name = config.get("VECTOR_DB", "collection_name", fallback="translation_corpus")
    device = config.get("EMBEDDINGS", "device", fallback="cuda")
    
    # Load corpus
    df = load_corpus(
        corpus_file=corpus_file,
        source_column=source_column,
        target_column=target_column,
        domain_column=domain_column,
        sample=args.sample
    )
    
    # Initialize Vector DB Manager
    logger.info(f"\n{'='*50}")
    logger.info("Initializing Vector Database Manager")
    logger.info(f"{'='*50}")
    logger.info(f"DB Type: {db_type}")
    logger.info(f"Embedding Model: {embedding_model}")
    logger.info(f"DB Path: {db_path}")
    logger.info(f"Device: {device}")
    
    vector_db = VectorDBManager(
        db_type=db_type,
        embedding_model=embedding_model,
        db_path=db_path,
        collection_name=collection_name,
        device=device
    )
    
    # Build database
    logger.info(f"\n{'='*50}")
    logger.info("Building Vector Database")
    logger.info(f"{'='*50}")
    
    vector_db.build_from_dataframe(
        df=df,
        source_column=source_column,
        target_column=target_column,
        domain_column=domain_column,
        batch_size=args.batch_size,
        clear_existing=args.clear_existing
    )
    
    # Show statistics
    stats = vector_db.get_stats()
    logger.info(f"\n{'='*50}")
    logger.info("Vector Database Statistics")
    logger.info(f"{'='*50}")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info(f"\n{'='*50}")
    logger.info("✅ Vector database built successfully!")
    logger.info(f"{'='*50}")
    logger.info(f"Database saved to: {db_path}")
    
    # Test retrieval
    logger.info(f"\n{'='*50}")
    logger.info("Testing retrieval...")
    logger.info(f"{'='*50}")
    
    test_query = df[source_column].iloc[0]
    logger.info(f"Test query: {test_query[:100]}...")
    
    results = vector_db.retrieve(query=test_query, top_k=3)
    logger.info(f"Retrieved {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        logger.info(f"\nResult {i} (similarity: {result['similarity']:.4f}):")
        logger.info(f"  Source: {result['source'][:100]}...")
        logger.info(f"  Target: {result['target'][:100]}...")
        if 'domain' in result:
            logger.info(f"  Domain: {result['domain']}")
    
    logger.info(f"\n{'='*50}")
    logger.info("All done! 🎉")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
