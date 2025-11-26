"""
Utility functions for the RAG Translation Agent

This module provides common utilities including:
- Configuration loading
- Logging setup
- Data preprocessing
- Text normalization
"""

import os
import yaml
import json
import configparser
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
import torch


def load_config(config_path: str = "config.ini") -> configparser.ConfigParser:
    """
    Load configuration from INI file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigParser object with loaded configuration
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def setup_logging(logs_dir: str = "./logs") -> None:
    """
    Setup logging configuration using loguru
    
    Args:
        logs_dir: Directory to store log files
    """
    os.makedirs(logs_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add custom handlers
    logger.add(
        os.path.join(logs_dir, "rag_agent_{time}.log"),
        rotation="500 MB",
        retention="10 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
    )


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """
    Get the best available device (cuda, mps, or cpu)
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by handling common variations
    
    Args:
        text: Input Arabic text
        
    Returns:
        Normalized Arabic text
    """
    if not text:
        return ""
    
    # Normalize Arabic characters
    replacements = {
        'أ': 'ا',
        'إ': 'ا',
        'آ': 'ا',
        'ة': 'ه',
        'ى': 'ي',
    }
    
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized.strip()


def normalize_english_text(text: str) -> str:
    """
    Normalize English text
    
    Args:
        text: Input English text
        
    Returns:
        Normalized English text
    """
    if not text:
        return ""
    
    # Basic normalization
    normalized = text.strip()
    normalized = ' '.join(normalized.split())  # Remove extra whitespace
    
    return normalized


def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Split long text into overlapping chunks
    
    Args:
        text: Input text to chunk
        max_length: Maximum length of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for delimiter in ['. ', '! ', '? ', '؟ ', '۔ ', '। ']:
                last_delim = text[start:end].rfind(delimiter)
                if last_delim != -1:
                    end = start + last_delim + len(delimiter)
                    break
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def format_context_examples(examples: List[Dict[str, str]], 
                           include_source: bool = True,
                           include_target: bool = True) -> str:
    """
    Format retrieved examples into context string
    
    Args:
        examples: List of retrieved examples with 'source' and 'target' keys
        include_source: Whether to include source text
        include_target: Whether to include target text
        
    Returns:
        Formatted context string
    """
    if not examples:
        return ""
    
    context_parts = []
    for i, example in enumerate(examples, 1):
        parts = []
        if include_source:
            parts.append(f"English: {example.get('source', '')}")
        if include_target:
            parts.append(f"Arabic: {example.get('target', '')}")
        
        context_parts.append(f"Example {i}:\n" + "\n".join(parts))
    
    return "\n\n".join(context_parts)


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")
