"""
Evaluation Script for RAG Translation Agent

This script evaluates the performance of the RAG translation system
using various metrics (BLEU, chrF, TER).

Usage:
    python evaluate.py --config config.ini
    python evaluate.py --test_file test.csv --output_file results.json
"""

import argparse
import os
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger
import sacrebleu

from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager
from utils import load_config, setup_logging, save_json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG Translation Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--with_context",
        action="store_true",
        help="Evaluate with RAG context"
    )
    parser.add_argument(
        "--without_context",
        action="store_true",
        help="Evaluate without RAG context"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Evaluate on a sample of N examples"
    )
    
    return parser.parse_args()


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        BLEU score
    """
    # sacrebleu expects references as List[List[str]]
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    return bleu.score


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """
    Compute chrF score (character-level F-score)
    Better for morphologically rich languages like Arabic
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        chrF score
    """
    refs = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(predictions, refs)
    return chrf.score


def evaluate_agent(
    agent: RAGTranslationAgent,
    test_df: pd.DataFrame,
    source_column: str = "en",
    target_column: str = "ar",
    domain_column: str = None,
    use_context: bool = True,
    sample: int = None
) -> Dict[str, Any]:
    """
    Evaluate RAG Translation Agent
    
    Args:
        agent: Initialized RAGTranslationAgent
        test_df: Test DataFrame
        source_column: Name of source column
        target_column: Name of target column
        domain_column: Name of domain column
        use_context: Whether to use RAG context
        sample: Number of examples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating on {len(test_df)} examples (use_context={use_context})")
    
    if sample and sample < len(test_df):
        test_df = test_df.sample(n=sample, random_state=42)
        logger.info(f"Sampled {sample} examples")
    
    # Get source texts and references
    source_texts = test_df[source_column].tolist()
    references = test_df[target_column].tolist()
    domains = test_df[domain_column].tolist() if domain_column and domain_column in test_df.columns else [None] * len(test_df)
    
    # Translate
    predictions = []
    all_results = []
    
    for source in tqdm(zip(source_texts), total=len(source_texts), desc="Translating"):
        source_text = source[0] if isinstance(source, tuple) else source
        result = agent.translate(
            source_text=source_text,
            use_context=use_context,
            return_context=True
        )
        predictions.append(result["translation"])
        all_results.append(result)
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    bleu_score = compute_bleu(predictions, references)
    chrf_score = compute_chrf(predictions, references)
    
    logger.info(f"BLEU: {bleu_score:.2f}")
    logger.info(f"chrF: {chrf_score:.2f}")
    
    # Prepare results
    results = {
        "use_context": use_context,
        "num_examples": len(test_df),
        "metrics": {
            "bleu": bleu_score,
            "chrf": chrf_score
        },
        "examples": []
    }
    
    # Add sample translations
    for i in range(min(10, len(source_texts))):
        results["examples"].append({
            "source": source_texts[i],
            "reference": references[i],
            "prediction": predictions[i],
            "used_context": all_results[i]["used_context"],
            "num_retrieved": all_results[i].get("num_retrieved", 0)
        })
    
    return results


def main():
    """Main function"""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logs_dir = config.get("GENERAL", "logs_dir", fallback="./logs")
    setup_logging(logs_dir)
    
    logger.info("Starting RAG Translation Agent evaluation")
    
    # Get parameters
    test_file = args.test_file or config.get("EVALUATION", "test_file", fallback="./data/test.csv")
    source_column = config.get("DATA", "source_column", fallback="en")
    target_column = config.get("DATA", "target_column", fallback="ar")
    domain_column = config.get("DATA", "domain_column", fallback=None)
    
    # Check if we should evaluate with/without context
    eval_with_context = args.with_context or (not args.without_context)
    eval_without_context = args.without_context
    
    # Load test data
    logger.info(f"Loading test data from: {test_file}")
    test_df = pd.read_csv(test_file)
    logger.info(f"Loaded {len(test_df)} test examples")
    
    # Initialize Vector DB
    logger.info("Initializing Vector Database Manager...")
    db_type = config.get("VECTOR_DB", "db_type", fallback="chromadb")
    embedding_model = config.get("EMBEDDINGS", "model_name")
    db_path = config.get("GENERAL", "vector_db_dir", fallback="./vector_db")
    collection_name = config.get("VECTOR_DB", "collection_name", fallback="translation_corpus")
    device = config.get("EMBEDDINGS", "device", fallback="cuda")
    
    vector_db = VectorDBManager(
        db_type=db_type,
        embedding_model=embedding_model,
        db_path=db_path,
        collection_name=collection_name,
        device=device
    )
    
    # Initialize RAG Agent with Local Llama
    logger.info("Initializing RAG Translation Agent (Local Llama)...")
    translation_model = config.get("TRANSLATION", "model_name")
    max_length = config.getint("TRANSLATION", "max_length", fallback=256)
    temperature = config.getfloat("TRANSLATION", "temperature", fallback=0.3)
    top_p = config.getfloat("TRANSLATION", "top_p", fallback=0.9)
    top_k_retrieval = config.getint("VECTOR_DB", "top_k", fallback=3)
    use_4bit = config.getboolean("TRANSLATION", "use_4bit", fallback=True)
    
    agent = RAGTranslationAgent(
        model_name=translation_model,
        vector_db_manager=vector_db,
        device=device,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k_retrieval=top_k_retrieval,
        use_4bit=use_4bit
    )
    
    # Evaluate
    all_results = {}
    
    if eval_with_context:
        logger.info("\n" + "="*50)
        logger.info("Evaluating WITH RAG context")
        logger.info("="*50)
        results_with = evaluate_agent(
            agent=agent,
            test_df=test_df,
            source_column=source_column,
            target_column=target_column,
            domain_column=domain_column,
            use_context=True,
            sample=args.sample
        )
        all_results["with_context"] = results_with
    
    if eval_without_context:
        logger.info("\n" + "="*50)
        logger.info("Evaluating WITHOUT RAG context")
        logger.info("="*50)
        results_without = evaluate_agent(
            agent=agent,
            test_df=test_df,
            source_column=source_column,
            target_column=target_column,
            domain_column=domain_column,
            use_context=False,
            sample=args.sample
        )
        all_results["without_context"] = results_without
    
    # Save results
    output_file = args.output_file or os.path.join(
        config.get("GENERAL", "output_dir", fallback="./outputs"),
        "evaluation_results.json"
    )
    save_json(all_results, output_file)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    if "with_context" in all_results:
        logger.info("\nWith RAG Context:")
        logger.info(f"  BLEU: {all_results['with_context']['metrics']['bleu']:.2f}")
        logger.info(f"  chrF: {all_results['with_context']['metrics']['chrf']:.2f}")
    
    if "without_context" in all_results:
        logger.info("\nWithout RAG Context:")
        logger.info(f"  BLEU: {all_results['without_context']['metrics']['bleu']:.2f}")
        logger.info(f"  chrF: {all_results['without_context']['metrics']['chrf']:.2f}")
    
    if "with_context" in all_results and "without_context" in all_results:
        bleu_diff = all_results['with_context']['metrics']['bleu'] - all_results['without_context']['metrics']['bleu']
        chrf_diff = all_results['with_context']['metrics']['chrf'] - all_results['without_context']['metrics']['chrf']
        logger.info("\nImprovement with RAG:")
        logger.info(f"  BLEU: {bleu_diff:+.2f}")
        logger.info(f"  chrF: {chrf_diff:+.2f}")
    
    logger.info("\n" + "="*50)
    logger.info("Evaluation complete! 🎉")
    logger.info("="*50)


if __name__ == "__main__":
    main()
