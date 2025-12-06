#!/usr/bin/env python3
"""
Translation Evaluation System
Evaluates machine translation quality using chrF++, COMET, and BLEU metrics
with a weighted combined score optimized for Arabic translations.
"""

import sacrebleu
from comet import download_model, load_from_checkpoint
import warnings
warnings.filterwarnings('ignore')


class TranslationEvaluator:
    """
    Comprehensive translation evaluation using multiple metrics.
    
    Metrics:
    - chrF++: Character-level metric (excellent for Arabic morphology)
    - COMET: Neural semantic metric (best for semantic fidelity)
    - BLEU: Classical n-gram metric (standard baseline)
    
    Combined Score Formula:
    Score = 0.5 * (COMET+1)/2 + 0.3 * chrF++/100 + 0.2 * BLEU/100
    """
    
    def __init__(self, comet_model="Unbabel/wmt22-comet-da"):
        """
        Initialize the evaluator with specified COMET model.
        
        Args:
            comet_model: COMET model identifier (default: wmt22-comet-da)
        """
        print("🔧 Initializing Translation Evaluator...")
        print(f"   Loading COMET model: {comet_model}")
        
        # Try to load COMET model
        try:
            self.comet_model = load_from_checkpoint(comet_model)
            print("✅ Evaluator initialized successfully!\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not load COMET model '{comet_model}': {e}")
            print("   COMET evaluation will be skipped. Other metrics will still work.\n")
            self.comet_model = None
    
    def compute_chrf(self, hypothesis, reference):
        """
        Compute chrF++ score.
        
        Args:
            hypothesis: Translation output
            reference: Ground truth translation
            
        Returns:
            float: chrF++ score (0-100)
        """
        chrf = sacrebleu.sentence_chrf(
            hypothesis, 
            [reference],
            word_order=2  # chrF++ uses word order component
        )
        return chrf.score
    
    def compute_bleu(self, hypothesis, reference):
        """
        Compute sentence-level BLEU score.
        
        Args:
            hypothesis: Translation output
            reference: Ground truth translation
            
        Returns:
            float: BLEU score (0-100)
        """
        bleu = sacrebleu.sentence_bleu(
            hypothesis,
            [reference]
        )
        return bleu.score
    
    def compute_comet(self, source, hypothesis, reference):
        """
        Compute COMET score.
        
        Args:
            source: Source sentence (English)
            hypothesis: Translation output
            reference: Ground truth translation
            
        Returns:
            float: COMET score (typically -1 to 1), or None if model not available
        """
        if self.comet_model is None:
            return None
            
        data = [{
            "src": source,
            "mt": hypothesis,
            "ref": reference
        }]
        
        # COMET returns an object with .scores attribute
        output = self.comet_model.predict(data, batch_size=1, gpus=0)
        return output.scores[0]
    
    def compute_combined_score(self, comet_score, chrf_score, bleu_score):
        """
        Compute weighted combined score.
        
        Formula: 0.5 * (COMET+1)/2 + 0.3 * chrF++/100 + 0.2 * BLEU/100
        If COMET is unavailable: 0.6 * chrF++/100 + 0.4 * BLEU/100
        
        Args:
            comet_score: COMET score (-1 to 1) or None if unavailable
            chrf_score: chrF++ score (0-100)
            bleu_score: BLEU score (0-100)
            
        Returns:
            float: Combined score (0-1), multiply by 100 for percentage
        """
        # Normalize chrF++ and BLEU from [0, 100] to [0, 1]
        normalized_chrf = chrf_score / 100
        normalized_bleu = bleu_score / 100
        
        if comet_score is None:
            # Fallback weights when COMET is not available
            combined = (0.6 * normalized_chrf + 0.4 * normalized_bleu)
        else:
            # Normalize COMET from [-1, 1] to [0, 1]
            normalized_comet = (comet_score + 1) / 2
            
            # Weighted combination
            combined = (0.5 * normalized_comet + 
                       0.3 * normalized_chrf + 
                       0.2 * normalized_bleu)
        
        return combined
    
    def evaluate(self, source, hypothesis, reference, translation_name="Translation"):
        """
        Complete evaluation of a translation.
        
        Args:
            source: Source sentence (English)
            hypothesis: Translation output (Arabic)
            reference: Ground truth translation (Arabic)
            translation_name: Name/identifier for this translation
            
        Returns:
            dict: Dictionary containing all metrics and scores
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {translation_name}")
        print(f"{'='*70}")
        
        # Compute individual metrics
        print("⏳ Computing metrics...")
        chrf_score = self.compute_chrf(hypothesis, reference)
        bleu_score = self.compute_bleu(hypothesis, reference)
        comet_score = self.compute_comet(source, hypothesis, reference)
        
        # Compute combined score
        combined_score = self.compute_combined_score(comet_score, chrf_score, bleu_score)
        combined_percentage = combined_score * 100
        
        # Print results
        print(f"\n📊 Individual Metrics:")
        print(f"   • chrF++:  {chrf_score:6.2f} / 100  (weight: {0.6 if comet_score is None else 0.3})")
        print(f"   • BLEU:    {bleu_score:6.2f} / 100  (weight: {0.4 if comet_score is None else 0.2})")
        if comet_score is not None:
            print(f"   • COMET:   {comet_score:7.4f}       (weight: 0.5, range: -1 to 1)")
        else:
            print(f"   • COMET:   N/A                (model not loaded)")
        
        print(f"\n🎯 Combined Score:")
        print(f"   • Raw:        {combined_score:.4f}")
        print(f"   • Percentage: {combined_percentage:.2f}%")
        
        # Quality interpretation
        quality = self._interpret_quality(combined_percentage)
        print(f"   • Quality:    {quality}")
        
        # Create results dictionary
        results = {
            'translation_name': translation_name,
            'source': source,
            'hypothesis': hypothesis,
            'reference': reference,
            'chrf_score': chrf_score,
            'bleu_score': bleu_score,
            'comet_score': comet_score,
            'combined_score': combined_score,
            'combined_percentage': combined_percentage,
            'quality_level': quality
        }
        
        return results
    
    def _interpret_quality(self, percentage):
        """
        Interpret quality based on combined score percentage.
        
        Args:
            percentage: Combined score as percentage (0-100)
            
        Returns:
            str: Quality interpretation
        """
        if percentage >= 90:
            return "🌟 Excellent"
        elif percentage >= 80:
            return "✅ Very Good"
        elif percentage >= 70:
            return "👍 Good"
        elif percentage >= 60:
            return "⚠️  Acceptable"
        elif percentage >= 50:
            return "⚠️  Poor"
        else:
            return "❌ Very Poor"
    
    def evaluate_batch(self, translations):
        """
        Evaluate multiple translations and compare them.
        
        Args:
            translations: List of tuples (name, source, hypothesis, reference)
            
        Returns:
            list: List of result dictionaries
        """
        results = []
        
        for name, source, hypothesis, reference in translations:
            result = self.evaluate(source, hypothesis, reference, name)
            results.append(result)
        
        # Print comparison summary
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results):
        """
        Print a comparison table of all evaluated translations.
        
        Args:
            results: List of result dictionaries
        """
        print(f"\n\n{'='*70}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        # Sort by combined score
        sorted_results = sorted(results, key=lambda x: x['combined_percentage'], reverse=True)
        
        # Print header
        print(f"\n{'Rank':<6}{'Translation':<25}{'chrF++':<10}{'BLEU':<10}{'COMET':<10}{'Combined':<12}{'Quality'}")
        print(f"{'-'*6}{'-'*25}{'-'*10}{'-'*10}{'-'*10}{'-'*12}{'-'*15}")
        
        # Print each result
        for i, result in enumerate(sorted_results, 1):
            comet_display = f"{result['comet_score']:>6.3f}" if result['comet_score'] is not None else "   N/A"
            print(f"{i:<6}"
                  f"{result['translation_name']:<25}"
                  f"{result['chrf_score']:>6.2f}    "
                  f"{result['bleu_score']:>6.2f}    "
                  f"{comet_display}    "
                  f"{result['combined_percentage']:>6.2f}%     "
                  f"{result['quality_level']}")
        
        # Print best translation
        best = sorted_results[0]
        print(f"\n🏆 Best Translation: {best['translation_name']}")
        print(f"   Combined Score: {best['combined_percentage']:.2f}%")
        print(f"{'='*70}\n")


def test_evaluator():
    """
    Test the evaluator with three sample translations.
    """
    print("="*70)
    print("TRANSLATION EVALUATION TEST")
    print("="*70)
    
    # Initialize evaluator
    evaluator = TranslationEvaluator()
    
    # Test data: (name, source, hypothesis, reference)
    # Using a sentence from the economic domain
    source = "Quantitative easing program must be followed by balance sheet normalization in inflation targeting."
    
    reference = "يجب أن يتبع برنامج التيسير الكمي تطبيع الميزانية العمومية في استهداف التضخم."
    
    # Three different translation hypotheses with varying quality
    translations = [
        (
            "Translation 1: Perfect Match",
            source,
            "يجب أن يتبع برنامج التيسير الكمي تطبيع الميزانية العمومية في استهداف التضخم.",
            reference
        ),
        (
            "Translation 2: Good Quality",
            source,
            "يجب أن يتبع برنامج التيسير الكمي بتطبيع الميزانية في استهداف التضخم.",
            reference
        ),
        (
            "Translation 3: Moderate Quality",
            source,
            "برنامج التخفيف الكمي يجب أن يتبعه تطبيع الميزان في استهداف التضخم.",
            reference
        )
    ]
    
    # Evaluate all translations
    results = evaluator.evaluate_batch(translations)
    
    # Print detailed information for the source and reference
    print("\n" + "="*70)
    print("SOURCE & REFERENCE")
    print("="*70)
    print(f"\n📝 Source (English):")
    print(f"   {source}")
    print(f"\n✅ Reference (Arabic):")
    print(f"   {reference}")
    print(f"\n{'='*70}\n")
    
    return results


def demo_with_different_domains():
    """
    Demonstrate evaluation with examples from different domains.
    """
    print("\n\n" + "="*70)
    print("MULTI-DOMAIN EVALUATION DEMO")
    print("="*70)
    
    evaluator = TranslationEvaluator()
    
    # Examples from different domains
    examples = [
        # Economic domain
        (
            "Economic - High Quality",
            "Basel III capital requirements must include stress testing exercise for too-big-to-fail resolution.",
            "يجب أن تتضمن متطلبات رأس المال وفق بازل 3 تمرين اختبار الضغط لحل مشكلة المؤسسات الأكبر من أن تفشل.",
            "يجب أن تتضمن متطلبات رأس المال وفق بازل 3 تمرين اختبار الضغط لحل مشكلة المؤسسات الأكبر من أن تفشل."
        ),
        # Technology domain
        (
            "Technology - Medium Quality",
            "Container orchestration in a Kubernetes cluster handles pod management, service discovery, and rolling updates seamlessly.",
            "تنسيق الحاويات في مجموعة كوبرنيتس يتعامل مع إدارة البودات واكتشاف الخدمات والتحديثات المتدحرجة.",
            "يتعامل تنسيق الحاويات في مجموعة كوبرنيتس مع إدارة البودات واكتشاف الخدمة والتحديثات المتدحرجة بسلاسة."
        ),
        # Medical domain
        (
            "Medical - Good Quality",
            "Patients with acute myocardial infarction must undergo immediate percutaneous coronary intervention.",
            "يجب أن يخضع المرضى المصابون باحتشاء القلب الحاد للتدخل التاجي الفوري عن طريق الجلد.",
            "يجب أن يخضع المرضى المصابون باحتشاء عضلة القلب الحاد للتدخل التاجي عن طريق الجلد الفوري."
        )
    ]
    
    results = evaluator.evaluate_batch(examples)
    
    return results


if __name__ == "__main__":
    # Run basic test
    print("\n🧪 Running Basic Evaluation Test...\n")
    test_results = test_evaluator()
    
    # Run multi-domain demo
    print("\n🧪 Running Multi-Domain Evaluation Demo...\n")
    demo_results = demo_with_different_domains()
    
    print("\n✅ All tests completed successfully!")
    print("="*70)
