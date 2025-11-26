"""
Vector Database Manager for RAG Translation Agent

This module handles the creation and management of vector databases for storing
and retrieving translation examples. It supports both ChromaDB and FAISS.

Key Features:
- Build vector database from parallel corpora
- Semantic search for similar translations
- Support for multiple vector DB backends
- Efficient batch processing
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import pickle

from sentence_transformers import SentenceTransformer
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    import faiss
    FAISS_AVAILABLE = True
    # Check if GPU is available for FAISS
    try:
        gpu_res = faiss.StandardGpuResources()
        FAISS_GPU_AVAILABLE = True
        logger.info("FAISS GPU support available")
    except Exception:
        FAISS_GPU_AVAILABLE = False
        logger.info("FAISS GPU not available, using CPU")
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-gpu")

from utils import ensure_dir, normalize_english_text, normalize_arabic_text


class VectorDBManager:
    """
    Manages vector database for storing and retrieving translation examples
    
    This class encapsulates the logic for:
    1. Creating embeddings from parallel corpora
    2. Storing embeddings in a vector database
    3. Performing semantic search to retrieve similar examples
    """
    
    def __init__(
        self,
        db_type: str = "chromadb",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        db_path: str = "./vector_db",
        collection_name: str = "translation_corpus",
        device: str = "cuda"
    ):
        """
        Initialize the Vector Database Manager
        
        Args:
            db_type: Type of vector DB ('chromadb' or 'faiss')
            embedding_model: Name of the sentence transformer model
            db_path: Path to store the vector database
            collection_name: Name of the collection/index
            device: Device for embedding model ('cuda', 'cpu', or 'mps')
        """
        self.db_type = db_type.lower()
        self.db_path = db_path
        self.collection_name = collection_name
        self.device = device
        self.use_gpu = device == "cuda" and FAISS_GPU_AVAILABLE
        
        ensure_dir(db_path)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize vector database
        self.db = None
        self.collection = None
        self.index = None
        self.metadata_store = []
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the vector database backend"""
        if self.db_type == "chromadb":
            if not CHROMADB_AVAILABLE:
                raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
            self._initialize_chromadb()
        elif self.db_type == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
            self._initialize_faiss()
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB"""
        logger.info("Initializing ChromaDB...")
        self.db = chromadb.PersistentClient(path=self.db_path)
        
        try:
            # Try to get existing collection
            self.collection = self.db.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection
            self.collection = self.db.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _initialize_faiss(self):
        """Initialize FAISS index with GPU support if available"""
        logger.info("Initializing FAISS...")
        
        index_path = os.path.join(self.db_path, f"{self.collection_name}.index")
        metadata_path = os.path.join(self.db_path, f"{self.collection_name}_metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            cpu_index = faiss.read_index(index_path)
            
            # Move to GPU if available and requested
            if self.use_gpu:
                logger.info("Moving FAISS index to GPU...")
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                logger.info("FAISS index loaded on GPU")
            else:
                self.index = cpu_index
                logger.info("FAISS index loaded on CPU")
            
            with open(metadata_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        else:
            # Create new index
            # Using IndexFlatIP (inner product) for cosine similarity
            # Normalize vectors before adding to use cosine similarity
            cpu_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Move to GPU if available and requested
            if self.use_gpu:
                logger.info("Creating FAISS index on GPU...")
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                logger.info("FAISS index created on GPU")
            else:
                self.index = cpu_index
                logger.info("FAISS index created on CPU")
            
            self.metadata_store = []
    
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        source_column: str = "en",
        target_column: str = "ar",
        domain_column: str = None,
        batch_size: int = 32,
        clear_existing: bool = False
    ):
        """
        Build vector database from a pandas DataFrame
        
        Args:
            df: DataFrame containing parallel corpus
            source_column: Name of the source language column
            target_column: Name of the target language column
            domain_column: Optional domain/category column
            batch_size: Batch size for embedding generation
            clear_existing: Whether to clear existing data
        """
        logger.info(f"Building vector database from DataFrame with {len(df)} entries")
        
        if clear_existing:
            self._clear_db()
        
        # Prepare data
        source_texts = df[source_column].fillna("").tolist()
        target_texts = df[target_column].fillna("").tolist()
        
        # Get domain if available
        if domain_column and domain_column in df.columns:
            domains = df[domain_column].fillna("general").tolist()
        else:
            domains = ["general"] * len(df)
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(source_texts), batch_size), desc="Embedding batches"):
            batch_texts = source_texts[i:i + batch_size]
            # Normalize texts before embedding
            batch_texts = [normalize_english_text(text) for text in batch_texts]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Store in vector database
        logger.info("Storing embeddings in vector database...")
        self._add_embeddings(source_texts, target_texts, domains, all_embeddings)
        
        logger.info(f"Successfully built vector database with {len(source_texts)} entries")
    
    def _add_embeddings(
        self,
        source_texts: List[str],
        target_texts: List[str],
        domains: List[str],
        embeddings: np.ndarray
    ):
        """Add embeddings to the vector database"""
        if self.db_type == "chromadb":
            self._add_to_chromadb(source_texts, target_texts, domains, embeddings)
        elif self.db_type == "faiss":
            self._add_to_faiss(source_texts, target_texts, domains, embeddings)
    
    def _add_to_chromadb(
        self,
        source_texts: List[str],
        target_texts: List[str],
        domains: List[str],
        embeddings: np.ndarray
    ):
        """Add embeddings to ChromaDB"""
        ids = [f"doc_{i}" for i in range(len(source_texts))]
        metadatas = [
            {
                "source": source,
                "target": target,
                "domain": domain
            }
            for source, target, domain in zip(source_texts, target_texts, domains)
        ]
        
        # ChromaDB expects list of lists for embeddings
        embeddings_list = embeddings.tolist()
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in tqdm(range(0, len(ids), batch_size), desc="Adding to ChromaDB"):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings_list[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
    
    def _add_to_faiss(
        self,
        source_texts: List[str],
        target_texts: List[str],
        domains: List[str],
        embeddings: np.ndarray
    ):
        """Add embeddings to FAISS"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        for source, target, domain in zip(source_texts, target_texts, domains):
            self.metadata_store.append({
                "source": source,
                "target": target,
                "domain": domain
            })
        
        # Save index and metadata
        self._save_faiss()
    
    def _save_faiss(self):
        """Save FAISS index and metadata to disk"""
        index_path = os.path.join(self.db_path, f"{self.collection_name}.index")
        metadata_path = os.path.join(self.db_path, f"{self.collection_name}_metadata.pkl")
        
        # Move index to CPU before saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
        
        logger.info(f"FAISS index saved to {index_path}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar translation examples for a query
        
        Args:
            query: Source text to find similar examples for
            top_k: Number of results to retrieve
            domain_filter: Optional domain to filter results
            similarity_threshold: Minimum similarity score (0-1 for cosine)
            
        Returns:
            List of dictionaries containing retrieved examples with metadata
        """
        # Generate query embedding
        query_normalized = normalize_english_text(query)
        query_embedding = self.embedding_model.encode(
            [query_normalized],
            convert_to_numpy=True
        )
        
        if self.db_type == "chromadb":
            return self._retrieve_chromadb(query_embedding, top_k, domain_filter, similarity_threshold)
        elif self.db_type == "faiss":
            return self._retrieve_faiss(query_embedding, top_k, domain_filter, similarity_threshold)
    
    def _retrieve_chromadb(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        domain_filter: Optional[str],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Retrieve from ChromaDB"""
        where_filter = {"domain": domain_filter} if domain_filter else None
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where_filter
        )
        
        retrieved = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for i, metadata in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i]
                # Convert distance to similarity (ChromaDB returns cosine distance)
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    retrieved.append({
                        'source': metadata['source'],
                        'target': metadata['target'],
                        'domain': metadata['domain'],
                        'similarity': similarity
                    })
        
        return retrieved
    
    def _retrieve_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        domain_filter: Optional[str],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Retrieve from FAISS"""
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        # Get more results if we need to filter by domain
        search_k = top_k * 10 if domain_filter else top_k
        similarities, indices = self.index.search(query_embedding, search_k)
        
        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata = self.metadata_store[idx]
            similarity = float(similarities[0][i])
            
            # Apply filters
            if domain_filter and metadata['domain'] != domain_filter:
                continue
            
            if similarity >= similarity_threshold:
                retrieved.append({
                    'source': metadata['source'],
                    'target': metadata['target'],
                    'domain': metadata['domain'],
                    'similarity': similarity
                })
            
            if len(retrieved) >= top_k:
                break
        
        return retrieved
    
    def _clear_db(self):
        """Clear existing database"""
        if self.db_type == "chromadb":
            try:
                self.db.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.db.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        elif self.db_type == "faiss":
            self.index.reset()
            self.metadata_store = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        if self.db_type == "chromadb":
            count = self.collection.count()
        elif self.db_type == "faiss":
            count = self.index.ntotal
        else:
            count = 0
        
        return {
            "db_type": self.db_type,
            "collection_name": self.collection_name,
            "total_entries": count,
            "embedding_model": self.embedding_model.__class__.__name__,
            "embedding_dim": self.embedding_dim
        }
