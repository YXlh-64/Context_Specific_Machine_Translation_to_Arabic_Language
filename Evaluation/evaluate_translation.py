"""
Translation System Evaluation Script
=====================================
This script evaluates the translation system using BLEU and CHRF+ metrics.
It supports both English-to-Arabic and French-to-Arabic translation evaluation.
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
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    print("Error: sacrebleu package not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sacrebleu"])
    from sacrebleu.metrics import BLEU, CHRF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """Evaluator for translation system using BLEU and CHRF+ metrics."""
    
    def __init__(self, api_url: str = "http://localhost:5002/api/translate"):
        """
        Initialize the evaluator.
        
        Args:
            api_url: URL of the translation API endpoint
        """
        self.api_url = api_url
        self.bleu = BLEU()
        self.chrf = CHRF()
        
    def translate_text(self, text: str, source_lang: str, target_lang: str = "ar", 
                      max_retries: int = 3, delay: float = 1.0) -> str:
        """
        Translate text using the API.
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, fr)
            target_lang: Target language code (ar)
            max_retries: Maximum number of retries on failure
            delay: Delay between retries in seconds
            
        Returns:
            Translated text
        """
        payload = {
            "text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "translation_service": "openrouter",
            "num_variants": 1  # We only need one translation for evaluation
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
                    # Get the first translation variant
                    # Handle different response formats
                    if "translations" in result and len(result["translations"]) > 0:
                        translation = result["translations"][0]
                        # If translation is a string, return it directly
                        if isinstance(translation, str):
                            return translation
                        # If translation is a dict with translated_text
                        elif isinstance(translation, dict) and "translated_text" in translation:
                            return translation["translated_text"]
                        else:
                            logger.error(f"Unexpected translation format: {translation}")
                            return ""
                    elif "translation" in result:
                        # Direct translation field
                        translation = result["translation"]
                        if isinstance(translation, str):
                            return translation
                        else:
                            logger.error(f"Translation is not a string: {translation}")
                            return ""
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return ""
                else:
                    logger.warning(f"API returned status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return ""
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                return ""
            except Exception as e:
                logger.error(f"Error during translation (attempt {attempt + 1}/{max_retries}): {e}")
                # Log the full response for debugging
                try:
                    if 'response' in locals():
                        logger.debug(f"Response status: {response.status_code}")
                        logger.debug(f"Response body: {response.text[:500]}")
                except:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                return ""
        
        return ""
    
    def load_dataset(self, csv_path: str, source_col: str, target_col: str) -> Tuple[List[str], List[str]]:
        """
        Load dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            source_col: Name of source language column
            target_col: Name of target language column (reference translations)
            
        Returns:
            Tuple of (source_texts, reference_translations)
        """
        try:
            # Use quoting to handle commas in fields
            df = pd.read_csv(
                csv_path, 
                quoting=1,  # QUOTE_ALL - handle fields with commas
                on_bad_lines='skip',  # Skip malformed lines
                encoding='utf-8',
                engine='python'  # More flexible parser
            )
            
            # Check if required columns exist
            if source_col not in df.columns or target_col not in df.columns:
                logger.warning(f"Expected columns '{source_col}' and '{target_col}' not found.")
                logger.info(f"Available columns: {df.columns.tolist()}")
                # Try to use first two columns if column names don't match
                if len(df.columns) >= 2:
                    source_col = df.columns[0]
                    target_col = df.columns[1]
                    logger.info(f"Using columns: '{source_col}' and '{target_col}'")
                else:
                    raise ValueError("CSV must have at least 2 columns")
            
            source_texts = df[source_col].astype(str).tolist()
            reference_texts = df[target_col].astype(str).tolist()
            
            # Filter out empty or NaN values
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
                            source_lang: str, sample_size: int = None) -> Dict:
        """
        Evaluate translations using BLEU and CHRF+ metrics.
        
        Args:
            source_texts: List of source texts to translate
            reference_texts: List of reference translations
            source_lang: Source language code
            sample_size: Optional sample size for evaluation (None = use all)
            
        Returns:
            Dictionary containing evaluation results
        """
        # Sample if requested
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
                references.append([reference_text])  # BLEU expects list of references
                logger.info(f"  Hypothesis: {hypothesis[:100]}")
                logger.info(f"  Reference: {reference_text[:100]}")
            else:
                failed_translations += 1
                logger.warning(f"  Failed to translate text {idx + 1}")
            
            # Add delay to avoid overwhelming the API
            if idx < len(source_texts) - 1:
                time.sleep(0.5)
        
        if not hypotheses:
            raise ValueError("All translations failed. Please check your API connection.")
        
        logger.info(f"Completed {len(hypotheses)} translations ({failed_translations} failed)")
        
        # Calculate BLEU score
        logger.info("Calculating BLEU score...")
        # BLEU expects: hypotheses = list of strings, references = list of lists of strings
        # We have one reference per hypothesis
        bleu_score = self.bleu.corpus_score(hypotheses, [[ref[0] for ref in references]])
        
        # Calculate CHRF+ score
        logger.info("Calculating CHRF+ score...")
        # CHRF expects the same format
        chrf_score = self.chrf.corpus_score(hypotheses, [[ref[0] for ref in references]])
        
        results = {
            "total_samples": len(source_texts),
            "successful_translations": len(hypotheses),
            "failed_translations": failed_translations,
            "bleu_score": float(bleu_score.score),
            "chrf_score": float(chrf_score.score),
            "source_language": source_lang,
            "target_language": "ar",
            "timestamp": datetime.now().isoformat(),
            "detailed_results": {
                "hypotheses": hypotheses,
                "references": [ref[0] for ref in references],
                "source_texts": source_texts[:len(hypotheses)]
            }
        }
        
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
        print(f"  BLEU Score:  {results['bleu_score']:.2f}")
        print(f"  CHRF+ Score: {results['chrf_score']:.2f}")
        print("="*80 + "\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        # Remove detailed results from saved file to reduce size
        results_to_save = {k: v for k, v in results.items() if k != 'detailed_results'}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # Save detailed results separately
        detailed_output_path = output_path.replace('.json', '_detailed.json')
        with open(detailed_output_path, 'w', encoding='utf-8') as f:
            json.dump(results['detailed_results'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {detailed_output_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate translation system using BLEU and CHRF+ metrics"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=500,
        help="Number of samples to evaluate from each dataset (default: all)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:5002/api/translate",
        help="URL of the translation API"
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Evaluate only English to Arabic"
    )
    parser.add_argument(
        "--french-only",
        action="store_true",
        help="Evaluate only French to Arabic"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "Data"
    results_dir = base_dir / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Check if API is running
    api_url = args.api_url
    api_base = api_url.rsplit('/', 1)[0]  # Get base URL for health check
    try:
        health_check = requests.get(f"{api_base}/health", timeout=5)
        if health_check.status_code != 200:
            logger.error("API is not responding correctly. Please start the backend server.")
            return
    except requests.exceptions.RequestException:
        logger.error(f"Cannot connect to API at {api_base}")
        logger.info("Please start the backend server: cd app && ./run_backend.sh")
        return
    
    logger.info("API is running. Starting evaluation...")
    if args.sample_size:
        logger.info(f"Evaluating {args.sample_size} samples from each dataset")
    else:
        logger.info("Evaluating all samples in datasets")
    
    # Initialize evaluator
    evaluator = TranslationEvaluator(api_url)
    
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
                en_sources, 
                en_references, 
                source_lang="en",
                sample_size=args.sample_size
            )
            
            evaluator.print_results(en_results, "English → Arabic")
            evaluator.save_results(
                en_results, 
                str(results_dir / f"english_to_arabic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            )
        except Exception as e:
            logger.error(f"Error evaluating English to Arabic: {e}")
            import traceback
            traceback.print_exc()
    
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
                fr_sources, 
                fr_references, 
                source_lang="fr",
                sample_size=args.sample_size
            )
            
            evaluator.print_results(fr_results, "French → Arabic")
            evaluator.save_results(
                fr_results, 
                str(results_dir / f"french_to_arabic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            )
        except Exception as e:
            logger.error(f"Error evaluating French to Arabic: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Evaluation complete! Check the Results directory for detailed outputs.")


if __name__ == "__main__":
    main()
