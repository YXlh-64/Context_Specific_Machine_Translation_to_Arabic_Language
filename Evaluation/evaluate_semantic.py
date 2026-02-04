"""
Semantic-Based Translation Evaluation
======================================
This script evaluates translations using advanced semantic similarity metrics:
- COMET: Neural metric trained on human judgments
- BERTScore: Contextual embedding-based similarity
- Sentence Embeddings: Cosine similarity using multilingual sentence transformers
- Semantic Similarity Score: Combined semantic metrics

These metrics better capture meaning preservation than static n-gram based metrics.
"""

import pandas as pd
import requests
import json
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path
import logging
from datetime import datetime
import sys
import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track available metrics
AVAILABLE_METRICS = {
    'sacrebleu': False,
    'bertscore': False,
    'comet': False,
    'sentence_transformers': False
}

# Try importing basic metrics
try:
    from sacrebleu.metrics import BLEU, CHRF
    AVAILABLE_METRICS['sacrebleu'] = True
except ImportError:
    logger.warning("sacrebleu not available. Install with: pip install sacrebleu")

# Try importing BERTScore
try:
    from bert_score import score as bert_score
    AVAILABLE_METRICS['bertscore'] = True
except ImportError:
    logger.warning("BERTScore not available. Install with: pip install bert-score")

# Try importing COMET
try:
    from comet import download_model, load_from_checkpoint
    AVAILABLE_METRICS['comet'] = True
except ImportError:
    logger.warning("COMET not available. Install with: pip install unbabel-comet")

# Try importing SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AVAILABLE_METRICS['sentence_transformers'] = True
except ImportError:
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")


@dataclass
class SemanticMetrics:
    """Container for semantic evaluation metrics."""
    
    # Semantic metrics
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    
    comet_score: Optional[float] = None
    comet_qe_score: Optional[float] = None  # Quality estimation without reference
    
    embedding_similarity: Optional[float] = None
    
    # Combined semantic score
    semantic_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class SemanticTranslationEvaluator:
    """Advanced evaluator focusing on semantic similarity metrics."""
    
    def __init__(self, api_url: str = "http://localhost:5002/api/translate"):
        """
        Initialize the semantic evaluator.
        
        Args:
            api_url: URL of the translation API endpoint
        """
        self.api_url = api_url
        
        # Initialize available metrics
        if AVAILABLE_METRICS['sacrebleu']:
            self.bleu = BLEU()
            self.chrf = CHRF()
        
        # Load COMET models (lazy loading)
        self.comet_model = None
        self.comet_qe_model = None
        
        # Load sentence transformer model (lazy loading)
        self.sentence_model = None
        
    def _load_comet_model(self):
        """Lazy load COMET model."""
        if self.comet_model is None and AVAILABLE_METRICS['comet']:
            logger.info("Loading COMET model (this may take a while on first run)...")
            try:
                model_path = download_model("Unbabel/wmt22-comet-da")
                self.comet_model = load_from_checkpoint(model_path)
                logger.info("COMET model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load COMET model: {e}")
                AVAILABLE_METRICS['comet'] = False
    
    def _load_comet_qe_model(self):
        """Lazy load COMET-QE model (reference-free)."""
        if self.comet_qe_model is None and AVAILABLE_METRICS['comet']:
            logger.info("Loading COMET-QE model...")
            try:
                model_path = download_model("Unbabel/wmt22-cometkiwi-da")
                self.comet_qe_model = load_from_checkpoint(model_path)
                logger.info("COMET-QE model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load COMET-QE model: {e}")
    
    def _load_sentence_model(self):
        """Lazy load sentence transformer model."""
        if self.sentence_model is None and AVAILABLE_METRICS['sentence_transformers']:
            logger.info("Loading multilingual sentence transformer model...")
            try:
                # Use multilingual model that works well with Arabic
                self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                AVAILABLE_METRICS['sentence_transformers'] = False
    
    def translate_text(self, text: str, source_lang: str, target_lang: str = "ar", 
                      max_retries: int = 3, delay: float = 1.0) -> str:
        """
        Translate text using the API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            max_retries: Maximum number of retries
            delay: Delay between retries
            
        Returns:
            Translated text or empty string on failure
        """
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
                logger.error(f"Translation error (attempt {attempt + 1}/{max_retries}): {e}")
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
    
    def calculate_bertscore(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore metrics."""
        if not AVAILABLE_METRICS['bertscore']:
            return {}
        
        try:
            logger.info("Calculating BERTScore...")
            P, R, F1 = bert_score(
                hypotheses,
                references,
                lang="ar",
                model_type="bert-base-multilingual-cased",
                verbose=False,
                batch_size=8
            )
            
            return {
                'precision': float(P.mean().item()),
                'recall': float(R.mean().item()),
                'f1': float(F1.mean().item())
            }
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return {}
    
    def calculate_comet_score(self, sources: List[str], hypotheses: List[str], 
                             references: List[str]) -> Optional[float]:
        """Calculate COMET score (with reference)."""
        if not AVAILABLE_METRICS['comet']:
            return None
        
        self._load_comet_model()
        if self.comet_model is None:
            return None
        
        try:
            logger.info("Calculating COMET score...")
            data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(sources, hypotheses, references)
            ]
            
            scores = self.comet_model.predict(data, batch_size=8, gpus=0)
            return float(np.mean(scores['scores']))
        except Exception as e:
            logger.error(f"COMET calculation failed: {e}")
            return None
    
    def calculate_comet_qe_score(self, sources: List[str], hypotheses: List[str]) -> Optional[float]:
        """Calculate COMET-QE score (reference-free quality estimation)."""
        if not AVAILABLE_METRICS['comet']:
            return None
        
        self._load_comet_qe_model()
        if self.comet_qe_model is None:
            return None
        
        try:
            logger.info("Calculating COMET-QE score (reference-free)...")
            data = [
                {"src": src, "mt": hyp}
                for src, hyp in zip(sources, hypotheses)
            ]
            
            scores = self.comet_qe_model.predict(data, batch_size=8, gpus=0)
            return float(np.mean(scores['scores']))
        except Exception as e:
            logger.error(f"COMET-QE calculation failed: {e}")
            return None
    
    def calculate_embedding_similarity(self, hypotheses: List[str], references: List[str]) -> Optional[float]:
        """Calculate cosine similarity using sentence embeddings."""
        if not AVAILABLE_METRICS['sentence_transformers']:
            return None
        
        self._load_sentence_model()
        if self.sentence_model is None:
            return None
        
        try:
            logger.info("Calculating embedding similarity...")
            hyp_embeddings = self.sentence_model.encode(hypotheses, show_progress_bar=False)
            ref_embeddings = self.sentence_model.encode(references, show_progress_bar=False)
            
            # Calculate cosine similarity for each pair
            similarities = []
            for hyp_emb, ref_emb in zip(hyp_embeddings, ref_embeddings):
                sim = cosine_similarity([hyp_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return float(np.mean(similarities))
        except Exception as e:
            logger.error(f"Embedding similarity calculation failed: {e}")
            return None
    
    def calculate_semantic_score(self, metrics: SemanticMetrics) -> float:
        """
        Calculate combined semantic score from available metrics.
        
        This combines BERTScore, COMET, and embedding similarity with appropriate weights.
        """
        scores = []
        weights = []
        
        # COMET score (highest weight - trained on human judgments)
        if metrics.comet_score is not None:
            scores.append(metrics.comet_score * 100)  # Scale to 0-100
            weights.append(0.4)
        
        # BERTScore F1 (contextual embeddings)
        if metrics.bertscore_f1 is not None:
            scores.append(metrics.bertscore_f1 * 100)  # Scale to 0-100
            weights.append(0.35)
        
        # Embedding similarity (sentence-level semantic similarity)
        if metrics.embedding_similarity is not None:
            scores.append(metrics.embedding_similarity * 100)  # Scale to 0-100
            weights.append(0.25)
        
        if not scores:
            return None
        
        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    def evaluate_translations(self, source_texts: List[str], reference_texts: List[str],
                            source_lang: str, sample_size: int = None,
                            use_traditional_metrics: bool = True,
                            use_bertscore: bool = True,
                            use_comet: bool = True,
                            use_comet_qe: bool = False,
                            use_embeddings: bool = True) -> Dict:
        """
        Evaluate translations using semantic metrics.
        
        Args:
            source_texts: Source texts to translate
            reference_texts: Reference translations
            source_lang: Source language code
            sample_size: Optional sample size
            use_traditional_metrics: Calculate BLEU/CHRF
            use_bertscore: Calculate BERTScore
            use_comet: Calculate COMET score
            use_comet_qe: Calculate COMET-QE (reference-free)
            use_embeddings: Calculate embedding similarity
            
        Returns:
            Dictionary with evaluation results
        """
        # Sample if requested
        if sample_size and sample_size < len(source_texts):
            logger.info(f"Sampling {sample_size} pairs from {len(source_texts)} total pairs")
            import random
            indices = random.sample(range(len(source_texts)), sample_size)
            source_texts = [source_texts[i] for i in indices]
            reference_texts = [reference_texts[i] for i in indices]
        
        # Translate all texts
        hypotheses = []
        valid_sources = []
        valid_references = []
        failed_translations = 0
        
        logger.info(f"Starting translation of {len(source_texts)} texts...")
        
        for idx, (source_text, reference_text) in enumerate(zip(source_texts, reference_texts)):
            logger.info(f"Translating {idx + 1}/{len(source_texts)}: {source_text[:50]}...")
            
            hypothesis = self.translate_text(source_text, source_lang, "ar")
            
            if hypothesis:
                hypotheses.append(hypothesis)
                valid_sources.append(source_text)
                valid_references.append(reference_text)
            else:
                failed_translations += 1
                logger.warning(f"  Failed to translate text {idx + 1}")
            
            if idx < len(source_texts) - 1:
                time.sleep(0.5)
        
        if not hypotheses:
            raise ValueError("All translations failed. Please check your API connection.")
        
        logger.info(f"Completed {len(hypotheses)} translations ({failed_translations} failed)")
        
        # Initialize metrics container
        metrics = SemanticMetrics()
        
        # Calculate traditional metrics
        if use_traditional_metrics and AVAILABLE_METRICS['sacrebleu']:
            logger.info("Calculating traditional metrics (BLEU, CHRF)...")
            bleu_score = self.bleu.corpus_score(hypotheses, [valid_references])
            chrf_score = self.chrf.corpus_score(hypotheses, [valid_references])
            metrics.bleu = float(bleu_score.score)
            metrics.chrf = float(chrf_score.score)
        
        # Calculate BERTScore
        if use_bertscore:
            bertscore_results = self.calculate_bertscore(hypotheses, valid_references)
            if bertscore_results:
                metrics.bertscore_precision = bertscore_results['precision']
                metrics.bertscore_recall = bertscore_results['recall']
                metrics.bertscore_f1 = bertscore_results['f1']
        
        # Calculate COMET score
        if use_comet:
            metrics.comet_score = self.calculate_comet_score(valid_sources, hypotheses, valid_references)
        
        # Calculate COMET-QE score
        if use_comet_qe:
            metrics.comet_qe_score = self.calculate_comet_qe_score(valid_sources, hypotheses)
        
        # Calculate embedding similarity
        if use_embeddings:
            metrics.embedding_similarity = self.calculate_embedding_similarity(hypotheses, valid_references)
        
        # Calculate combined semantic score
        metrics.semantic_score = self.calculate_semantic_score(metrics)
        
        results = {
            "total_samples": len(source_texts),
            "successful_translations": len(hypotheses),
            "failed_translations": failed_translations,
            "metrics": metrics.to_dict(),
            "source_language": source_lang,
            "target_language": "ar",
            "timestamp": datetime.now().isoformat(),
            "detailed_results": {
                "hypotheses": hypotheses,
                "references": valid_references,
                "source_texts": valid_sources
            }
        }
        
        return results
    
    def print_results(self, results: Dict, language_pair: str):
        """Print evaluation results with quality assessment."""
        print("\n" + "="*80)
        print(f"SEMANTIC EVALUATION RESULTS: {language_pair}")
        print("="*80)
        print(f"Total samples: {results['total_samples']}")
        print(f"Successful translations: {results['successful_translations']}")
        print(f"Failed translations: {results['failed_translations']}")
        
        metrics = results['metrics']
        
        # Traditional metrics
        if 'bleu' in metrics or 'chrf' in metrics:
            print(f"\n📊 TRADITIONAL METRICS:")
            if 'bleu' in metrics:
                print(f"  BLEU Score:  {metrics['bleu']:.2f}")
            if 'chrf' in metrics:
                print(f"  CHRF+ Score: {metrics['chrf']:.2f}")
        
        # Semantic metrics
        print(f"\n🧠 SEMANTIC METRICS:")
        
        if 'bertscore_f1' in metrics:
            print(f"  BERTScore:")
            print(f"    Precision: {metrics['bertscore_precision']:.4f}")
            print(f"    Recall:    {metrics['bertscore_recall']:.4f}")
            print(f"    F1:        {metrics['bertscore_f1']:.4f}")
        
        if 'comet_score' in metrics:
            print(f"  COMET Score:     {metrics['comet_score']:.4f}")
        
        if 'comet_qe_score' in metrics:
            print(f"  COMET-QE Score:  {metrics['comet_qe_score']:.4f}")
        
        if 'embedding_similarity' in metrics:
            print(f"  Embedding Similarity: {metrics['embedding_similarity']:.4f}")
        
        if 'semantic_score' in metrics:
            print(f"\n🎯 COMBINED SEMANTIC SCORE: {metrics['semantic_score']:.2f}/100")
            
            # Quality assessment based on semantic score
            score = metrics['semantic_score']
            if score >= 80:
                quality = "🌟 EXCELLENT - High semantic similarity"
            elif score >= 70:
                quality = "✅ GOOD - Acceptable semantic quality"
            elif score >= 60:
                quality = "⚠️  FAIR - Some semantic differences"
            else:
                quality = "❌ NEEDS IMPROVEMENT - Significant semantic gaps"
            
            print(f"  Quality: {quality}")
        
        print("="*80 + "\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON files."""
        results_to_save = {k: v for k, v in results.items() if k != 'detailed_results'}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        detailed_output_path = output_path.replace('.json', '_detailed.json')
        with open(detailed_output_path, 'w', encoding='utf-8') as f:
            json.dump(results['detailed_results'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {detailed_output_path}")
    
    def save_results_visualization(self, results: Dict, output_path: str, language_pair: str):
        """Save evaluation results as a PNG visualization."""
        try:
            metrics = results['metrics']
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f'Translation Evaluation Results: {language_pair}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Create grid for subplots
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, 
                                 left=0.08, right=0.95, top=0.93, bottom=0.08)
            
            # 1. Main Metrics Bar Chart (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            metric_names = []
            metric_values = []
            colors = []
            
            if 'semantic_score' in metrics:
                metric_names.append('Semantic\nScore')
                metric_values.append(metrics['semantic_score'])
                colors.append('#2ecc71')
            
            if 'bleu' in metrics:
                metric_names.append('BLEU')
                metric_values.append(metrics['bleu'])
                colors.append('#3498db')
            
            if 'chrf' in metrics:
                metric_names.append('CHRF+')
                metric_values.append(metrics['chrf'])
                colors.append('#9b59b6')
            
            bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
            ax1.set_title('Main Quality Metrics', fontsize=12, fontweight='bold', pad=10)
            ax1.set_ylim(0, 100)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 2. Semantic Metrics Breakdown (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            semantic_names = []
            semantic_values = []
            semantic_colors = []
            
            if 'comet_score' in metrics:
                semantic_names.append('COMET')
                semantic_values.append(metrics['comet_score'] * 100)
                semantic_colors.append('#e74c3c')
            
            if 'bertscore_f1' in metrics:
                semantic_names.append('BERTScore\nF1')
                semantic_values.append(metrics['bertscore_f1'] * 100)
                semantic_colors.append('#f39c12')
            
            if 'embedding_similarity' in metrics:
                semantic_names.append('Embedding\nSimilarity')
                semantic_values.append(metrics['embedding_similarity'] * 100)
                semantic_colors.append('#1abc9c')
            
            if semantic_values:
                bars2 = ax2.bar(semantic_names, semantic_values, color=semantic_colors, 
                              alpha=0.8, edgecolor='black')
                ax2.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
                ax2.set_title('Semantic Metrics Breakdown', fontsize=12, fontweight='bold', pad=10)
                ax2.set_ylim(0, 100)
                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                
                for bar, value in zip(bars2, semantic_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No semantic metrics available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_xticks([])
                ax2.set_yticks([])
            
            # 3. BERTScore Components (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            if 'bertscore_f1' in metrics:
                bert_components = ['Precision', 'Recall', 'F1']
                bert_values = [
                    metrics.get('bertscore_precision', 0) * 100,
                    metrics.get('bertscore_recall', 0) * 100,
                    metrics.get('bertscore_f1', 0) * 100
                ]
                bert_colors = ['#3498db', '#e67e22', '#2ecc71']
                
                bars3 = ax3.bar(bert_components, bert_values, color=bert_colors, 
                              alpha=0.8, edgecolor='black')
                ax3.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
                ax3.set_title('BERTScore Components', fontsize=12, fontweight='bold', pad=10)
                ax3.set_ylim(0, 100)
                ax3.grid(axis='y', alpha=0.3, linestyle='--')
                
                for bar, value in zip(bars3, bert_values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'BERTScore not calculated', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_xticks([])
                ax3.set_yticks([])
            
            # 4. Quality Assessment Gauge (middle right)
            ax4 = fig.add_subplot(gs[1, 1])
            if 'semantic_score' in metrics:
                score = metrics['semantic_score']
                
                # Create gauge
                theta = np.linspace(0, np.pi, 100)
                r = np.ones_like(theta)
                
                # Color segments
                colors_gauge = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
                boundaries = [0, 60, 70, 80, 100]
                
                for i in range(len(colors_gauge)):
                    theta_seg = np.linspace(boundaries[i] * np.pi / 100, 
                                           boundaries[i+1] * np.pi / 100, 50)
                    ax4.fill_between(theta_seg, 0, 1, color=colors_gauge[i], alpha=0.3)
                
                # Needle
                needle_angle = score * np.pi / 100
                ax4.plot([needle_angle, needle_angle], [0, 0.8], 'k-', linewidth=3)
                ax4.plot(needle_angle, 0.8, 'ko', markersize=10)
                
                # Labels
                ax4.text(0, -0.2, '0', ha='center', fontsize=10)
                ax4.text(np.pi, -0.2, '100', ha='center', fontsize=10)
                ax4.text(np.pi/2, 1.15, f'{score:.1f}', ha='center', 
                        fontsize=20, fontweight='bold')
                
                # Quality label
                if score >= 80:
                    quality_text = 'EXCELLENT'
                    quality_color = '#2ecc71'
                elif score >= 70:
                    quality_text = 'GOOD'
                    quality_color = '#f1c40f'
                elif score >= 60:
                    quality_text = 'FAIR'
                    quality_color = '#f39c12'
                else:
                    quality_text = 'NEEDS IMPROVEMENT'
                    quality_color = '#e74c3c'
                
                ax4.text(np.pi/2, -0.5, quality_text, ha='center', 
                        fontsize=14, fontweight='bold', color=quality_color)
                
                ax4.set_xlim(0, np.pi)
                ax4.set_ylim(-0.6, 1.2)
                ax4.axis('off')
                ax4.set_title('Quality Assessment (Semantic Score)', 
                            fontsize=12, fontweight='bold', pad=20)
            else:
                ax4.text(0.5, 0.5, 'Semantic score not available', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.axis('off')
            
            # 5. Summary Statistics (bottom span)
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            
            # Create summary table
            summary_data = [
                ['Total Samples', str(results['total_samples'])],
                ['Successful', str(results['successful_translations'])],
                ['Failed', str(results['failed_translations'])],
                ['Source Lang', results['source_language'].upper()],
                ['Target Lang', results['target_language'].upper()],
            ]
            
            # Add metrics to summary
            metric_display = []
            if 'semantic_score' in metrics:
                metric_display.append(['Semantic Score', f"{metrics['semantic_score']:.2f}/100"])
            if 'comet_score' in metrics:
                metric_display.append(['COMET', f"{metrics['comet_score']:.4f}"])
            if 'bertscore_f1' in metrics:
                metric_display.append(['BERTScore F1', f"{metrics['bertscore_f1']:.4f}"])
            if 'embedding_similarity' in metrics:
                metric_display.append(['Embedding Sim.', f"{metrics['embedding_similarity']:.4f}"])
            if 'bleu' in metrics:
                metric_display.append(['BLEU', f"{metrics['bleu']:.2f}"])
            if 'chrf' in metrics:
                metric_display.append(['CHRF+', f"{metrics['chrf']:.2f}"])
            
            # Split into two columns
            col1_data = summary_data
            col2_data = metric_display
            
            # Create two tables side by side
            y_start = 0.85
            x_col1 = 0.15
            x_col2 = 0.55
            
            # Column 1: General Info
            ax5.text(x_col1, y_start + 0.08, 'General Information', 
                    fontsize=12, fontweight='bold', transform=ax5.transAxes)
            for i, (label, value) in enumerate(col1_data):
                y_pos = y_start - i * 0.12
                ax5.text(x_col1, y_pos, f'{label}:', 
                        fontsize=10, fontweight='bold', transform=ax5.transAxes)
                ax5.text(x_col1 + 0.15, y_pos, value, 
                        fontsize=10, transform=ax5.transAxes)
            
            # Column 2: Metrics
            ax5.text(x_col2, y_start + 0.08, 'Evaluation Metrics', 
                    fontsize=12, fontweight='bold', transform=ax5.transAxes)
            for i, (label, value) in enumerate(col2_data):
                y_pos = y_start - i * 0.12
                ax5.text(x_col2, y_pos, f'{label}:', 
                        fontsize=10, fontweight='bold', transform=ax5.transAxes)
                ax5.text(x_col2 + 0.18, y_pos, value, 
                        fontsize=10, transform=ax5.transAxes)
            
            # Add timestamp
            timestamp = datetime.fromisoformat(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            ax5.text(0.5, 0.02, f'Generated: {timestamp}', 
                    ha='center', fontsize=9, style='italic', 
                    transform=ax5.transAxes, alpha=0.7)
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Semantic-based translation evaluation using COMET, BERTScore, and embeddings"
    )
    parser.add_argument("--sample-size", type=int, default=50,
                       help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--api-url", type=str, default="http://localhost:5002/api/translate",
                       help="Translation API URL")
    parser.add_argument("--english-only", action="store_true",
                       help="Evaluate only English to Arabic")
    parser.add_argument("--french-only", action="store_true",
                       help="Evaluate only French to Arabic")
    parser.add_argument("--no-bertscore", action="store_true",
                       help="Skip BERTScore calculation")
    parser.add_argument("--no-comet", action="store_true",
                       help="Skip COMET calculation")
    parser.add_argument("--use-comet-qe", action="store_true",
                       help="Calculate COMET-QE (reference-free)")
    parser.add_argument("--no-embeddings", action="store_true",
                       help="Skip embedding similarity calculation")
    parser.add_argument("--no-traditional", action="store_true",
                       help="Skip traditional metrics (BLEU, CHRF)")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "Data"
    results_dir = base_dir / "Results"
    outputs_dir = base_dir / "outputs"
    results_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Check API connection
    api_base = args.api_url.rsplit('/', 1)[0]
    try:
        health_check = requests.get(f"{api_base}/health", timeout=5)
        if health_check.status_code != 200:
            logger.error("API is not responding correctly.")
            return
    except requests.exceptions.RequestException:
        logger.error(f"Cannot connect to API at {api_base}")
        logger.info("Please start the backend server: cd app && ./run_backend.sh")
        return
    
    # Check available metrics
    available_count = sum(AVAILABLE_METRICS.values())
    logger.info(f"Available metric libraries: {available_count}/4")
    for metric, available in AVAILABLE_METRICS.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {metric}")
    
    if available_count == 0:
        logger.error("No evaluation metrics available. Please install required packages.")
        return
    
    logger.info("\nStarting semantic evaluation...")
    
    # Initialize evaluator
    evaluator = SemanticTranslationEvaluator(args.api_url)
    
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
                use_traditional_metrics=not args.no_traditional,
                use_bertscore=not args.no_bertscore,
                use_comet=not args.no_comet,
                use_comet_qe=args.use_comet_qe,
                use_embeddings=not args.no_embeddings
            )
            
            evaluator.print_results(en_results, "English → Arabic")
            
            # Save JSON results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evaluator.save_results(
                en_results,
                str(results_dir / f"semantic_english_to_arabic_{timestamp}.json")
            )
            
            # Save PNG visualization
            evaluator.save_results_visualization(
                en_results,
                str(outputs_dir / f"semantic_english_to_arabic_{timestamp}.png"),
                "English → Arabic"
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
                fr_sources, fr_references,
                source_lang="fr",
                sample_size=args.sample_size,
                use_traditional_metrics=not args.no_traditional,
                use_bertscore=not args.no_bertscore,
                use_comet=not args.no_comet,
                use_comet_qe=args.use_comet_qe,
                use_embeddings=not args.no_embeddings
            )
            
            evaluator.print_results(fr_results, "French → Arabic")
            
            # Save JSON results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evaluator.save_results(
                fr_results,
                str(results_dir / f"semantic_french_to_arabic_{timestamp}.json")
            )
            
            # Save PNG visualization
            evaluator.save_results_visualization(
                fr_results,
                str(outputs_dir / f"semantic_french_to_arabic_{timestamp}.png"),
                "French → Arabic"
            )
        except Exception as e:
            logger.error(f"Error evaluating French to Arabic: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Semantic evaluation complete!")
    print(f"\n📊 Results saved to:")
    print(f"   - JSON: {results_dir}/")
    print(f"   - Visualizations: {outputs_dir}/")
    print("\n💡 TIP: For more accurate results, install all metric libraries:")
    print("   pip install unbabel-comet sentence-transformers bert-score matplotlib")


if __name__ == "__main__":
    main()
