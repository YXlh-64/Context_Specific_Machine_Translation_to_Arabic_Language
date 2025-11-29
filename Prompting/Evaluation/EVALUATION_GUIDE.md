# Translation Evaluation System Guide

## Overview
This evaluation system provides comprehensive quality assessment for Arabic translations using multiple metrics with a weighted scoring approach.

## Features

### Metrics Implemented
1. **chrF++ (Character-level F-score++)** 
   - Weight: 0.6 (60%) when COMET unavailable, 0.3 (30%) with COMET
   - Range: 0-100
   - Best for: Morphologically rich languages like Arabic
   - Considers character n-grams and word order

2. **BLEU (Bilingual Evaluation Understudy)**
   - Weight: 0.4 (40%) when COMET unavailable, 0.2 (20%) with COMET
   - Range: 0-100
   - Best for: Word-level precision
   - Industry standard baseline metric

3. **COMET (Crosslingual Optimized Metric for Evaluation of Translation)**
   - Weight: 0.5 (50%) when available
   - Range: -1 to 1
   - Best for: Semantic similarity and fluency
   - Neural model trained on human judgments
   - *Note: Currently optional due to network requirements*

### Combined Scoring Formula

**With COMET:**
```
Score = 0.5 × ((COMET+1)/2) + 0.3 × (chrF++/100) + 0.2 × (BLEU/100)
```

**Without COMET (Fallback):**
```
Score = 0.6 × (chrF++/100) + 0.4 × (BLEU/100)
```

### Quality Interpretation
- **90-100%**: 🌟 Excellent
- **80-89%**: ✨ Very Good
- **70-79%**: 👍 Good
- **60-69%**: ⚠️ Acceptable
- **50-59%**: ⚡ Poor
- **0-49%**: ❌ Very Poor

## Usage

### Basic Usage
```python
from translation_evaluator import TranslationEvaluator

# Initialize evaluator
evaluator = TranslationEvaluator()

# Evaluate single translation
result = evaluator.evaluate(
    source="Quantitative easing program must be followed by balance sheet normalization.",
    hypothesis="يجب أن يتبع برنامج التيسير الكمي تطبيع الميزانية العمومية.",
    reference="يجب أن يتبع برنامج التيسير الكمي تطبيع الميزانية العمومية.",
    translation_name="Economic Translation"
)

print(f"Combined Score: {result['combined_percentage']:.2f}%")
```

### Batch Evaluation
```python
# Evaluate multiple translations
translations = [
    {
        'name': 'Translation 1',
        'source': 'Source text...',
        'hypothesis': 'Translated text...',
        'reference': 'Ground truth...'
    },
    # ... more translations
]

results = evaluator.evaluate_batch(translations)
# Automatically prints comparison summary and ranks translations
```

### Individual Metrics
```python
# Compute individual metrics
chrf_score = evaluator.compute_chrf(hypothesis, reference)
bleu_score = evaluator.compute_bleu(hypothesis, reference)
comet_score = evaluator.compute_comet(source, hypothesis, reference)

# Compute custom combined score
combined = evaluator.compute_combined_score(comet_score, chrf_score, bleu_score)
```

## Output Format

### Single Evaluation
```
======================================================================
Evaluating: Economic Translation
======================================================================
⏳ Computing metrics...

📊 Individual Metrics:
   • chrF++:  100.00 / 100  (weight: 0.6)
   • BLEU:    100.00 / 100  (weight: 0.4)
   • COMET:   N/A                (model not loaded)

🎯 Combined Score:
   • Raw:        1.0000
   • Percentage: 100.00%
   • Quality:    🌟 Excellent
```

### Batch Comparison
```
======================================================================
📊 COMPARISON SUMMARY
======================================================================

Rank  Translation              chrF++    BLEU      COMET     Combined    Quality
----------------------------------------------------------------------------------------
1     Translation 1            100.00    100.00       N/A    100.00%     🌟 Excellent
2     Translation 2             81.72     59.74       N/A     72.93%     👍 Good
3     Translation 3             60.54     25.67       N/A     46.59%     ❌ Very Poor

🏆 Best Translation: Translation 1
   Combined Score: 100.00%
```

## Result Dictionary Structure
```python
{
    'translation_name': str,
    'source': str,                    # English source text
    'hypothesis': str,                # Arabic translation
    'reference': str,                 # Arabic ground truth
    'chrf_score': float,              # 0-100
    'bleu_score': float,              # 0-100
    'comet_score': float or None,     # -1 to 1 or None
    'combined_score': float,          # 0-1
    'combined_percentage': float,     # 0-100
    'quality_level': str              # Quality interpretation
}
```

## Requirements
```
sacrebleu>=2.3.0
unbabel-comet>=2.0.0  # Optional but recommended
torch>=2.0.0          # Required for COMET
numpy>=1.24.0
tabulate>=0.9.0
```

Install dependencies:
```bash
pip install -r evaluator_requirements.txt
```

## Notes

### COMET Model
- The COMET model requires internet connection for first-time download
- Model files are cached locally after first download
- If COMET is unavailable, the system automatically uses fallback weights
- Current model: `Unbabel/wmt22-comet-da` (optimized for quality estimation)

### Arabic-Specific Considerations
- **chrF++** is weighted higher (60% vs 40% BLEU without COMET) because:
  - It handles Arabic morphology better
  - Less sensitive to tokenization differences
  - Better correlation with human judgments for Arabic
  
- **BLEU** limitations for Arabic:
  - Sensitive to agglutination and morphological variations
  - May underestimate quality of valid morphological variants
  - Still useful as baseline metric

### Performance
- chrF++ and BLEU: Very fast (< 1ms per sentence)
- COMET: Slower (requires neural network inference)
  - First evaluation: ~2-3 seconds (model loading)
  - Subsequent evaluations: ~100-200ms per sentence
  - GPU acceleration supported (set `gpus=1` in code)

## Examples

### Test File
Run the test file to see the evaluator in action:
```bash
python3 translation_evaluator.py
```

This runs two test scenarios:
1. Basic evaluation with three translation quality levels
2. Multi-domain evaluation (economic, technology, medical)

## Advanced Usage

### Custom Weights
Modify the weights in `compute_combined_score()`:
```python
# Example: Give more weight to BLEU
combined = (0.4 * normalized_comet + 
           0.3 * normalized_chrf + 
           0.3 * normalized_bleu)
```

### Quality Thresholds
Modify quality interpretation in `_interpret_quality()`:
```python
if percentage >= 95:
    return "🌟 Excellent"
elif percentage >= 85:
    return "✨ Very Good"
# ... etc
```

### GPU Acceleration
Enable GPU for COMET (if available):
```python
# In compute_comet() method, change:
output = self.comet_model.predict(data, batch_size=1, gpus=1)
```

## Troubleshooting

### COMET Model Not Loading
- **Issue**: Network errors or model not found
- **Solution**: System automatically falls back to chrF++ and BLEU
- **Alternative**: Download model manually or use offline mode

### Low Scores Despite Good Translation
- Check reference translation quality
- Verify correct language direction (EN→AR)
- Consider morphological variants in Arabic
- Use multiple references if available

### Memory Issues with Large Batches
- Process translations in smaller batches
- Disable COMET for large-scale evaluation
- Use GPU if available

## Citation

If using this evaluation system in research, please cite the underlying metrics:

**chrF++:**
```
Popović, M. (2017). chrF++: words helping character n-grams. 
WMT 2017.
```

**BLEU:**
```
Papineni, K., et al. (2002). BLEU: a method for automatic 
evaluation of machine translation. ACL 2002.
```

**COMET:**
```
Rei, R., et al. (2020). COMET: A Neural Framework for MT 
Evaluation. EMNLP 2020.
```

## Support
For issues or questions, please refer to the project documentation or contact the development team.
