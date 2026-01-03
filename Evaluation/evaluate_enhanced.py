"""
Enhanced Translation Evaluation with Multiple Metrics
======================================================
This script includes BLEU, CHRF+, TER, and optional BERTScore for comprehensive evaluation.
"""

import pandas as pd
import requests
import json
from typing import List, Dict, Tuple
import time
from pathlib import Path
import logging
from datetime import datetime
import sys

# Import evaluation metrics
try:
    from sacrebleu.metrics import BLEU, CHRF, TER
except ImportError:
    print("Installing sacrebleu...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu"])
    from sacrebleu.metrics import BLEU, CHRF, TER

# Try to import BERTScore (optional)
BERTSCORE_AVAILABLE = False
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("\nBERTScore not available. Install with: pip install bert-score")
    print("   BERTScore metrics will be skipped.\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTranslationEvaluator:
    """Enhanced evaluator with multiple metrics."""
    
    def __init__(self, api_url: str = "http://localhost:5002/api/translate"):
        self.api_url = api_url
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.ter = TER()
        
    def translate_text(self, text: str, source_lang: str, target_lang: str = "ar", 
                      max_retries: int = 3, delay: float = 1.0) -> str:
        """Translate text using the API."""
        payload = {
            "text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "translation_service": "openrouter",
            "num_variants": 1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "translations" in result and len(result["translations"]) > 0:
                        translation = result["translations"][0]
                        if isinstance(translation, str):
                            return translation
                        elif isinstance(translation, dict) and "translated_text" in translation:
                            return translation["translated_text"]
                    elif "translation" in result:
                        translation = result["translation"]
                        if isinstance(translation, str):
                            return translation
                    return ""
                else:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return ""
                    
            except Exception as e:
                logger.error(f"Error during translation (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                return ""
        
        return ""
    
    def load_dataset(self, csv_path: str, source_col: str, target_col: str) -> Tuple[List[str], List[str]]:
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(
                csv_path, 
                quoting=1,
                on_bad_lines='skip',
                encoding='utf-8',
                engine='python'
            )
            
            if source_col not in df.columns or target_col not in df.columns:
                logger.warning(f"Expected columns '{source_col}' and '{target_col}' not found.")
                logger.info(f"Available columns: {df.columns.tolist()}")
                if len(df.columns) >= 2:
                    source_col = df.columns[0]
                    target_col = df.columns[1]
                    logger.info(f"Using columns: '{source_col}' and '{target_col}'")
            
            source_texts = df[source_col].astype(str).tolist()
            reference_texts = df[target_col].astype(str).tolist()
            
            valid_pairs = [(s, r) for s, r in zip(source_texts, reference_texts) 
                          if s and r and s != 'nan' and r != 'nan' and len(s.strip()) > 0 and len(r.strip()) > 0]
            
            if not valid_pairs:
                raise ValueError("No valid translation pairs found in dataset")
                
            source_texts, reference_texts = zip(*valid_pairs)
            logger.info(f"Loaded {len(source_texts)} translation pairs from {csv_path}")
            return list(source_texts), list(reference_texts)
            
        except Exception as e:
            logger.error(f"Error loading dataset from {csv_path}: {e}")
            raise
    
    def evaluate_translations(self, source_texts: List[str], reference_texts: List[str],
                            source_lang: str, sample_size: int = None, 
                            use_bertscore: bool = False) -> Dict:
        """Evaluate translations using multiple metrics."""
        
        if sample_size and sample_size < len(source_texts):
            logger.info(f"Sampling {sample_size} pairs from {len(source_texts)} total pairs")
            import random
            indices = random.sample(range(len(source_texts)), sample_size)
            source_texts = [source_texts[i] for i in indices]
            reference_texts = [reference_texts[i] for i in indices]
        
        hypotheses = []
        references = []
        failed_translations = 0
        
        logger.info(f"Starting evaluation of {len(source_texts)} translations...")
        
        for idx, (source_text, reference_text) in enumerate(zip(source_texts, reference_texts)):
            logger.info(f"Translating {idx + 1}/{len(source_texts)}: {source_text[:50]}...")
            
            hypothesis = self.translate_text(source_text, source_lang, "ar")
            
            if hypothesis:
                hypotheses.append(hypothesis)
                references.append([reference_text])
            else:
                failed_translations += 1
                logger.warning(f"  Failed to translate text {idx + 1}")
            
            if idx < len(source_texts) - 1:
                time.sleep(0.5)
        
        if not hypotheses:
            raise ValueError("All translations failed. Please check your API connection.")
        
        logger.info(f"Completed {len(hypotheses)} translations ({failed_translations} failed)")
        
        # Calculate BLEU score
        logger.info("Calculating BLEU score...")
        bleu_score = self.bleu.corpus_score(hypotheses, [[ref[0] for ref in references]])
        
        # Calculate CHRF+ score
        logger.info("Calculating CHRF+ score...")
        chrf_score = self.chrf.corpus_score(hypotheses, [[ref[0] for ref in references]])
        
        # Calculate TER score
        logger.info("Calculating TER score...")
        ter_score = self.ter.corpus_score(hypotheses, [[ref[0] for ref in references]])
        
        results = {
            "total_samples": len(source_texts),
            "successful_translations": len(hypotheses),
            "failed_translations": failed_translations,
            "metrics": {
                "bleu": float(bleu_score.score),
                "chrf": float(chrf_score.score),
                "ter": float(ter_score.score)
            },
            "source_language": source_lang,
            "target_language": "ar",
            "timestamp": datetime.now().isoformat(),
            "detailed_results": {
                "hypotheses": hypotheses,
                "references": [ref[0] for ref in references],
                "source_texts": source_texts[:len(hypotheses)]
            }
        }
        
        # Calculate BERTScore if available and requested
        if use_bertscore and BERTSCORE_AVAILABLE:
            logger.info("Calculating BERTScore (this may take a while)...")
            try:
                P, R, F1 = bert_score(
                    hypotheses,
                    [ref[0] for ref in references],
                    lang="ar",
                    model_type="bert-base-multilingual-cased",
                    verbose=False
                )
                results["metrics"]["bertscore"] = {
                    "precision": float(P.mean().item()),
                    "recall": float(R.mean().item()),
                    "f1": float(F1.mean().item())
                }
                logger.info(f"BERTScore F1: {float(F1.mean().item()):.2f}")
            except Exception as e:
                logger.error(f"BERTScore calculation failed: {e}")
                results["metrics"]["bertscore"] = None
        
        return results
    
    def print_results(self, results: Dict, language_pair: str):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS: {language_pair}")
        print("="*80)
        print(f"Total samples: {results['total_samples']}")
        print(f"Successful translations: {results['successful_translations']}")
        print(f"Failed translations: {results['failed_translations']}")
        print(f"\n📊 METRICS:")
        print(f"  BLEU Score:  {results['metrics']['bleu']:.2f}")
        print(f"  CHRF+ Score: {results['metrics']['chrf']:.2f}")
        print(f"  TER Score:   {results['metrics']['ter']:.2f} (lower is better)")
        
        if "bertscore" in results["metrics"] and results["metrics"]["bertscore"]:
            bs = results["metrics"]["bertscore"]
            print(f"\n  BERTScore:")
            print(f"    Precision: {bs['precision']:.4f}")
            print(f"    Recall:    {bs['recall']:.4f}")
            print(f"    F1:        {bs['f1']:.4f}")
        
        # Quality assessment
        print(f"\n🎯 QUALITY ASSESSMENT:")
        bleu = results['metrics']['bleu']
        chrf = results['metrics']['chrf']
        
        if bleu >= 30:
            quality = "🌟 GOOD - Acceptable for production use"
        elif bleu >= 20:
            quality = "✅ FAIR - Needs some post-editing"
        elif bleu >= 10:
            quality = "⚠️  BASIC - Requires significant post-editing"
        else:
            quality = "❌ POOR - Needs system improvement"
        
        print(f"  {quality}")
        print("="*80 + "\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        results_to_save = {k: v for k, v in results.items() if k != 'detailed_results'}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        detailed_output_path = output_path.replace('.json', '_detailed.json')
        with open(detailed_output_path, 'w', encoding='utf-8') as f:
            json.dump(results['detailed_results'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {detailed_output_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced evaluation with BLEU, CHRF+, TER, and optional BERTScore"
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--api-url", type=str, default="http://localhost:5002/api/translate")
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--french-only", action="store_true")
    parser.add_argument("--use-bertscore", action="store_true", 
                       help="Calculate BERTScore (requires bert-score package)")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "Data"
    results_dir = base_dir / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Check API
    api_base = args.api_url.rsplit('/', 1)[0]
    try:
        health_check = requests.get(f"{api_base}/health", timeout=5)
        if health_check.status_code != 200:
            logger.error("API is not responding correctly.")
            return
    except requests.exceptions.RequestException:
        logger.error(f"Cannot connect to API at {api_base}")
        return
    
    logger.info("API is running. Starting enhanced evaluation...")
    if args.use_bertscore and not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore requested but not available. Install: pip install bert-score")
    
    evaluator = EnhancedTranslationEvaluator(args.api_url)
    
    # Evaluate English to Arabic
    if not args.french_only:
        print("\n" + "🇬🇧 → 🇸🇦 EVALUATING ENGLISH TO ARABIC TRANSLATION" + "\n")
        try:
            en_sources, en_references = evaluator.load_dataset(
                str(data_dir / "english.csv"),
                source_col="english",
                target_col="arabic"
            )
            
            en_results = evaluator.evaluate_translations(
                en_sources, en_references, 
                source_lang="en",
                sample_size=args.sample_size,
                use_bertscore=args.use_bertscore
            )
            
            evaluator.print_results(en_results, "English → Arabic")
            evaluator.save_results(
                en_results, 
                str(results_dir / f"enhanced_english_to_arabic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            )
        except Exception as e:
            logger.error(f"Error evaluating English to Arabic: {e}")
    
    # Evaluate French to Arabic
    if not args.english_only:
        print("\n" + "🇫🇷 → 🇸🇦 EVALUATING FRENCH TO ARABIC TRANSLATION" + "\n")
        try:
            fr_sources, fr_references = evaluator.load_dataset(
                str(data_dir / "french.csv"),
                source_col="french",
                target_col="arabic"
            )
            
            fr_results = evaluator.evaluate_translations(
                fr_sources, fr_references,
                source_lang="fr",
                sample_size=args.sample_size,
                use_bertscore=args.use_bertscore
            )
            
            evaluator.print_results(fr_results, "French → Arabic")
            evaluator.save_results(
                fr_results,
                str(results_dir / f"enhanced_french_to_arabic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            )
        except Exception as e:
            logger.error(f"Error evaluating French to Arabic: {e}")
    
    print("\n✅ Enhanced evaluation complete!")


if __name__ == "__main__":
    main()
