"""
RAG Comparison Evaluation Script

Compares translation quality with and without RAG context using metrics:
- BLEU score
- chrF score

Usage:
    python compare_rag.py --test_file test_data.csv --sample 100
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm
import json
from datetime import datetime

from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager
from utils import load_config, setup_logging, normalize_arabic_text

# Import metrics
from sacrebleu.metrics import BLEU, CHRF


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare RAG vs Non-RAG translation")
    
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test CSV file with 'en' and 'ar' columns"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to config file"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of context examples to retrieve"
    )
    
    return parser.parse_args()


def calculate_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU and chrF scores
    
    Args:
        references: List of reference translations
        hypotheses: List of predicted translations
        
    Returns:
        Dictionary with metric scores
    """
    # Initialize metrics
    bleu = BLEU()
    chrf = CHRF()
    
    # Calculate scores
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    
    return {
        "bleu": bleu_score.score,
        "chrf": chrf_score.score
    }


def evaluate_translations(
    agent: RAGTranslationAgent,
    test_data: pd.DataFrame,
    use_context: bool,
    desc: str
) -> Dict:
    """
    Evaluate translations with or without context
    
    Args:
        agent: RAGTranslationAgent instance
        test_data: Test dataframe with 'en' and 'ar' columns
        use_context: Whether to use RAG context
        desc: Description for progress bar
        
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {desc}")
    print(f"{'='*60}")
    
    translations = []
    errors = 0
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc=desc):
        source = row['en']
        
        try:
            result = agent.translate(
                source,
                use_context=use_context,
                return_context=use_context
            )
            translations.append({
                'source': source,
                'reference': row['ar'],
                'translation': result['translation'],
                'used_context': result.get('used_context', False),
                'num_retrieved': result.get('num_retrieved', 0) if use_context else 0
            })
        except Exception as e:
            print(f"\nError translating row {idx}: {e}")
            errors += 1
            translations.append({
                'source': source,
                'reference': row['ar'],
                'translation': '',
                'used_context': False,
                'num_retrieved': 0,
                'error': str(e)
            })
    
    # Extract translations and references
    hypotheses = [t['translation'] for t in translations if t['translation']]
    references = [t['reference'] for t in translations if t['translation']]
    
    # Calculate metrics
    if hypotheses and references:
        metrics = calculate_metrics(references, hypotheses)
    else:
        metrics = {"bleu": 0.0, "chrf": 0.0}
    
    return {
        'translations': translations,
        'metrics': metrics,
        'errors': errors,
        'total': len(test_data),
        'successful': len(hypotheses)
    }


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    print("🔬 RAG Translation Comparison Evaluation")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load test data
    print(f"\n📊 Loading test data from: {args.test_file}")
    test_df = pd.read_csv(args.test_file)
    
    # Validate columns
    if 'en' not in test_df.columns or 'ar' not in test_df.columns:
        raise ValueError("Test file must have 'en' and 'ar' columns")
    
    # Remove empty rows
    test_df = test_df.dropna(subset=['en', 'ar'])
    test_df = test_df[test_df['en'].str.strip() != '']
    test_df = test_df[test_df['ar'].str.strip() != '']
    
    # Sample if requested
    if args.sample and args.sample < len(test_df):
        test_df = test_df.sample(n=args.sample, random_state=42)
        print(f"   Sampled {args.sample} examples")
    
    print(f"   Total examples: {len(test_df)}")
    
    # Initialize Vector Database
    print("\n🗄️  Loading Vector Database...")
    vector_db = VectorDBManager(
        db_type=config.get("VECTOR_DB", "db_type", fallback="faiss"),
        embedding_model=config.get("EMBEDDINGS", "model_name"),
        db_path=config.get("GENERAL", "vector_db_dir", fallback="./vector_db"),
        collection_name=config.get("VECTOR_DB", "collection_name", fallback="translation_corpus"),
        device=config.get("EMBEDDINGS", "device", fallback="cuda")
    )
    
    stats = vector_db.get_stats()
    print(f"   ✓ Loaded {stats['total_entries']} translation examples")
    
    # Initialize Local Llama Agent
    print("\n🤖 Initializing Local Llama Translation Agent...")
    agent = RAGTranslationAgent(
        model_name=config.get("TRANSLATION", "model_name"),
        vector_db_manager=vector_db,
        device=config.get("TRANSLATION", "device", fallback="cuda"),
        max_length=config.getint("TRANSLATION", "max_length", fallback=256),
        temperature=config.getfloat("TRANSLATION", "temperature", fallback=0.3),
        top_p=config.getfloat("TRANSLATION", "top_p", fallback=0.9),
        top_k_retrieval=args.top_k,
        use_4bit=config.getboolean("TRANSLATION", "use_4bit", fallback=True)
    )
    print("   ✓ Local Llama agent ready")
    
    # Evaluate WITHOUT context (baseline)
    print("\n" + "="*60)
    print("1️⃣  BASELINE: Translation WITHOUT RAG Context")
    print("="*60)
    
    results_no_rag = evaluate_translations(
        agent=agent,
        test_data=test_df,
        use_context=False,
        desc="Baseline (No RAG)"
    )
    
    # Evaluate WITH context (RAG-enabled)
    print("\n" + "="*60)
    print("2️⃣  RAG-ENHANCED: Translation WITH Context")
    print("="*60)
    
    results_with_rag = evaluate_translations(
        agent=agent,
        test_data=test_df,
        use_context=True,
        desc="RAG-Enhanced"
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    print("\n🔵 WITHOUT RAG (Baseline):")
    print(f"   BLEU Score:  {results_no_rag['metrics']['bleu']:.2f}")
    print(f"   chrF Score:  {results_no_rag['metrics']['chrf']:.2f}")
    print(f"   Successful:  {results_no_rag['successful']}/{results_no_rag['total']}")
    print(f"   Errors:      {results_no_rag['errors']}")
    
    print("\n🟢 WITH RAG (Context-Aware):")
    print(f"   BLEU Score:  {results_with_rag['metrics']['bleu']:.2f}")
    print(f"   chrF Score:  {results_with_rag['metrics']['chrf']:.2f}")
    print(f"   Successful:  {results_with_rag['successful']}/{results_with_rag['total']}")
    print(f"   Errors:      {results_with_rag['errors']}")
    
    print("\n📈 IMPROVEMENT:")
    bleu_improvement = results_with_rag['metrics']['bleu'] - results_no_rag['metrics']['bleu']
    chrf_improvement = results_with_rag['metrics']['chrf'] - results_no_rag['metrics']['chrf']
    
    print(f"   BLEU: {bleu_improvement:+.2f} points ({bleu_improvement/results_no_rag['metrics']['bleu']*100:+.1f}%)")
    print(f"   chrF: {chrf_improvement:+.2f} points ({chrf_improvement/results_no_rag['metrics']['chrf']*100:+.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_summary = {
        "timestamp": timestamp,
        "test_file": args.test_file,
        "num_samples": len(test_df),
        "top_k": args.top_k,
        "baseline": {
            "metrics": results_no_rag['metrics'],
            "successful": results_no_rag['successful'],
            "errors": results_no_rag['errors']
        },
        "rag_enhanced": {
            "metrics": results_with_rag['metrics'],
            "successful": results_with_rag['successful'],
            "errors": results_with_rag['errors']
        },
        "improvement": {
            "bleu": bleu_improvement,
            "chrf": chrf_improvement,
            "bleu_percent": (bleu_improvement/results_no_rag['metrics']['bleu']*100) if results_no_rag['metrics']['bleu'] > 0 else 0,
            "chrf_percent": (chrf_improvement/results_no_rag['metrics']['chrf']*100) if results_no_rag['metrics']['chrf'] > 0 else 0
        }
    }
    
    summary_file = os.path.join(args.output_dir, f"comparison_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Save detailed translations
    detailed_results = []
    for i in range(len(results_no_rag['translations'])):
        detailed_results.append({
            'source': results_no_rag['translations'][i]['source'],
            'reference': results_no_rag['translations'][i]['reference'],
            'baseline_translation': results_no_rag['translations'][i]['translation'],
            'rag_translation': results_with_rag['translations'][i]['translation'],
            'num_retrieved_examples': results_with_rag['translations'][i]['num_retrieved']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = os.path.join(args.output_dir, f"detailed_translations_{timestamp}.csv")
    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    
    print(f"💾 Detailed translations saved to: {detailed_file}")
    
    # Save sample comparisons
    print("\n" + "="*60)
    print("📝 SAMPLE COMPARISONS (First 3 examples)")
    print("="*60)
    
    for i in range(min(3, len(detailed_results))):
        sample = detailed_results[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Source: {sample['source'][:100]}...")
        print(f"\nReference:  {sample['reference'][:100]}...")
        print(f"\nBaseline:   {sample['baseline_translation'][:100]}...")
        print(f"\nRAG (w/ {sample['num_retrieved_examples']} examples): {sample['rag_translation'][:100]}...")
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
