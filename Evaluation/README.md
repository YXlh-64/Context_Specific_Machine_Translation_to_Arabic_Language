# Translation System Evaluation

This directory contains scripts and data for evaluating the translation system using both **traditional** and **semantic-based metrics**. The semantic metrics provide more accurate quality assessment by measuring meaning preservation rather than just surface-level text similarity.

## 📁 Directory Structure

```
Evaluation/
├── Data/
│   ├── english.csv    # English to Arabic test data
│   └── french.csv     # French to Arabic test data
├── Results/           # Evaluation results (auto-generated)
├── evaluate_translation.py    # Basic evaluation (BLEU, CHRF)
├── evaluate_enhanced.py       # Enhanced with TER and BERTScore
├── evaluate_semantic.py       # Advanced semantic metrics (NEW!)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Which Script Should You Use?

### Quick Comparison

| Script | Metrics | Speed | Best For |
|--------|---------|-------|----------|
| `evaluate_translation.py` | BLEU, CHRF | Fast ⚡ | Quick baseline check |
| `evaluate_enhanced.py` | BLEU, CHRF, TER, BERTScore | Medium 🚶 | Comprehensive evaluation |
| `evaluate_semantic.py` | All semantic metrics | Slow 🐌 | Production quality assessment |

### Recommended: `evaluate_semantic.py` (NEW!)
Use this for the most accurate quality assessment. It includes:
- **COMET**: Neural metric trained on human judgments (state-of-the-art)
- **BERTScore**: Contextual embedding similarity
- **Sentence Embeddings**: Multilingual semantic similarity
- **Combined Semantic Score**: Weighted average of all semantic metrics
- Traditional metrics (BLEU, CHRF) for comparison

## 🚀 Quick Start

### 1. Install Dependencies

#### Full Installation (Recommended for Production)
Install all semantic metrics for the most accurate evaluation:

```bash
cd Evaluation
pip install -r requirements.txt
```

This includes: BLEU, CHRF, TER, BERTScore, COMET, and Sentence Transformers.

**Note**: First-time COMET usage will download model files (~1-2GB).

#### Minimal Installation (For Quick Testing)
If you just want basic metrics:

```bash
pip install pandas requests sacrebleu
```

### 2. Start the Translation API

Make sure your translation API backend is running:

```bash
cd ../app
./run_backend.sh
```

The API should be accessible at `http://localhost:5002`

### 3. Run the Evaluation

#### Option A: Semantic Evaluation (Recommended)
```bash
python evaluate_semantic.py --sample-size 50
```

#### Option B: Enhanced Evaluation
```bash
python evaluate_enhanced.py --sample-size 50 --use-bertscore
```

#### Option C: Basic Evaluation
```bash
python evaluate_translation.py --sample-size 50
```

## 📊 Evaluation Metrics Explained

### 🔵 Traditional Metrics (Static N-gram Based)

#### BLEU (Bilingual Evaluation Understudy)
- **What it measures**: N-gram precision between hypothesis and reference
- **Range**: 0-100 (higher is better)
- **Pros**: Fast, widely used, industry standard
- **Cons**: 
  - Doesn't capture semantic similarity
  - Penalizes valid paraphrases
  - Weak correlation with human judgments for morphologically rich languages like Arabic
- **When to use**: Quick baseline, comparison with other systems

#### CHRF+ (Character n-gram F-score)
- **What it measures**: Character-level overlap
- **Range**: 0-100 (higher is better)
- **Pros**: More robust to morphological variations than BLEU
- **Cons**: Still surface-level, doesn't understand meaning
- **When to use**: Arabic evaluation (better than BLEU for Arabic)

#### TER (Translation Edit Rate)
- **What it measures**: Number of edits needed to transform hypothesis into reference
- **Range**: 0-100 (lower is better)
- **Pros**: Intuitive (counts edits)
- **Cons**: Doesn't consider semantic equivalence
- **When to use**: Post-editing effort estimation

### 🟢 Semantic Metrics (Meaning-Based) ⭐ **Recommended**

#### COMET (Crosslingual Optimized Metric for Evaluation of Translation)
- **What it measures**: Translation quality using neural networks trained on human judgments
- **Range**: Typically 0-1 (higher is better)
- **Pros**: 
  - ✅ State-of-the-art accuracy
  - ✅ High correlation with human judgments (0.87+ Kendall's τ)
  - ✅ Understands semantic similarity
  - ✅ Works well for Arabic and multilingual contexts
- **Cons**: Slower than traditional metrics, requires GPU for large datasets
- **When to use**: **Production quality assessment, research papers, final evaluation**
- **Model**: Uses pre-trained XLM-RoBERTa

#### BERTScore
- **What it measures**: Contextual embedding similarity using BERT
- **Range**: 0-1 (higher is better)
- **Precision**: How much of the hypothesis is covered by the reference
- **Recall**: How much of the reference is covered by the hypothesis
- **F1**: Harmonic mean of precision and recall
- **Pros**:
  - ✅ Captures semantic similarity
  - ✅ Handles paraphrases well
  - ✅ No need for exact word matches
  - ✅ Language-specific models available
- **Cons**: Slower than BLEU, requires neural models
- **When to use**: **Semantic quality assessment, paraphrase handling**
- **Model**: Uses bert-base-multilingual-cased for Arabic

#### Sentence Embeddings Similarity
- **What it measures**: Cosine similarity between sentence embeddings
- **Range**: 0-1 (higher is better)
- **Pros**:
  - ✅ Captures sentence-level semantic meaning
  - ✅ Good for multilingual evaluation
  - ✅ Fast inference with pre-computed embeddings
- **Cons**: May miss fine-grained differences
- **When to use**: **Overall meaning preservation check**
- **Model**: Uses paraphrase-multilingual-mpnet-base-v2

#### Combined Semantic Score
- **What it measures**: Weighted average of COMET, BERTScore, and Embedding Similarity
- **Range**: 0-100 (higher is better)
- **Weights**: 
  - COMET: 40% (highest - trained on human judgments)
  - BERTScore F1: 35% (contextual embeddings)
  - Embedding Similarity: 25% (sentence-level semantics)
- **When to use**: **Single comprehensive quality indicator**

### 📈 Quality Benchmarks

#### Traditional Metrics (BLEU)
- **30+**: Good - Acceptable for production
- **20-30**: Fair - Needs post-editing
- **10-20**: Basic - Significant editing required
- **<10**: Poor - System needs improvement

#### Semantic Score (Combined)
- **80+**: Excellent - High semantic fidelity
- **70-80**: Good - Acceptable semantic quality
- **60-70**: Fair - Some semantic differences
- **<60**: Needs improvement - Significant semantic gaps

### 🎯 Why Semantic Metrics Are Better for Arabic

1. **Morphological Richness**: Arabic has complex morphology; semantic metrics handle variations better
2. **Word Order Flexibility**: Arabic allows flexible word order; semantic metrics capture equivalent meanings
3. **Paraphrasing**: Multiple valid translations exist; semantic metrics recognize them all
4. **Human Correlation**: Semantic metrics (especially COMET) correlate much better with human judgments

## 🔧 Configuration & Usage Examples

### Semantic Evaluation Script (`evaluate_semantic.py`)

#### Basic Usage
```bash
# Evaluate both English and French with all metrics
python evaluate_semantic.py --sample-size 50

# English only
python evaluate_semantic.py --sample-size 50 --english-only

# French only
python evaluate_semantic.py --sample-size 50 --french-only
```

#### Advanced Options
```bash
# Skip specific metrics to save time
python evaluate_semantic.py --sample-size 50 --no-comet  # Skip COMET (fastest)
python evaluate_semantic.py --sample-size 50 --no-bertscore  # Skip BERTScore

# Use reference-free quality estimation
python evaluate_semantic.py --sample-size 50 --use-comet-qe

# Minimal evaluation (fastest)
python evaluate_semantic.py --sample-size 50 --no-comet --no-bertscore --no-embeddings
```

#### Full Evaluation (All Metrics)
```bash
# This will take longer but gives comprehensive results
python evaluate_semantic.py \
  --sample-size 100 \
  --use-comet-qe
```

### Enhanced Evaluation Script (`evaluate_enhanced.py`)

```bash
# With BERTScore
python evaluate_enhanced.py --sample-size 50 --use-bertscore

# English only
python evaluate_enhanced.py --sample-size 50 --english-only --use-bertscore
```

### Basic Evaluation Script (`evaluate_translation.py`)

```bash
# Default: all samples
python evaluate_translation.py

# With sample size
python evaluate_translation.py --sample-size 100 --english-only
```

### Custom API URL

If your API runs on a different host/port:

```bash
python evaluate_semantic.py \
  --api-url http://192.168.1.100:8000/api/translate \
  --sample-size 50
```

### Evaluation Sample Size

- **10-20 samples**: Quick test, approximate results
- **50 samples**: Good balance of speed and accuracy (recommended for development)
- **100+ samples**: More reliable results (recommended for reporting)
- **All samples** (no `--sample-size`): Most accurate (use for final evaluation)

## 📈 Output

All scripts generate outputs in the `Results/` directory:

### 1. Summary JSON (`*_{timestamp}.json`)
Contains overall metrics and metadata:

```json
{
  "total_samples": 50,
  "successful_translations": 48,
  "failed_translations": 2,
  "metrics": {
    "bleu": 28.5,
    "chrf": 52.3,
    "bertscore_f1": 0.8234,
    "comet_score": 0.7891,
    "embedding_similarity": 0.8456,
    "semantic_score": 81.27
  },
  "source_language": "en",
  "target_language": "ar",
  "timestamp": "2026-02-04T10:30:00"
}
```

### 2. Detailed JSON (`*_{timestamp}_detailed.json`)
Contains all translations for error analysis:

```json
{
  "source_texts": ["Hello world", "..."],
  "hypotheses": ["مرحبا بالعالم", "..."],
  "references": ["مرحبا بالعالم", "..."]
}
```

### 3. Console Output
Real-time progress and formatted results:

```
================================================================================
SEMANTIC EVALUATION RESULTS: English → Arabic
================================================================================
Total samples: 50
Successful translations: 48
Failed translations: 2

📊 TRADITIONAL METRICS:
  BLEU Score:  28.50
  CHRF+ Score: 52.30

🧠 SEMANTIC METRICS:
  BERTScore:
    Precision: 0.8421
    Recall:    0.8051
    F1:        0.8234
  COMET Score:     0.7891
  Embedding Similarity: 0.8456

🎯 COMBINED SEMANTIC SCORE: 81.27/100
  Quality: ✅ GOOD - Acceptable semantic quality
================================================================================
```

## 🎯 Interpreting Results

### Understanding Score Differences

**Example Result**:
- BLEU: 25.0 (fair)
- Semantic Score: 78.0 (good)

**Interpretation**: The translation preserves meaning well but uses different words/phrases than the reference. This is actually good - it shows your system produces natural, semantically equivalent translations.

### When Traditional Metrics Are Misleading

❌ **BLEU can be low even for good translations when:**
- System uses valid synonyms
- System uses different word order
- System produces more fluent paraphrases

✅ **Semantic metrics solve this by:**
- Recognizing equivalent meanings
- Understanding context
- Correlating with human judgments

### Red Flags to Watch For

⚠️ **High BLEU + Low Semantic Score**: System might be overfitting to reference style
⚠️ **Low BLEU + Low Semantic Score**: System has fundamental quality issues
✅ **Moderate BLEU + High Semantic Score**: Ideal - natural, meaning-preserving translations

### Example Output

```
================================================================================
EVALUATION RESULTS: English → Arabic
================================================================================
Total samples: 662
Successful translations: 660
Failed translations: 2

📊 METRICS:
  BLEU Score:  45.23
  CHRF+ Score: 67.89
================================================================================
```

## 📝 Data Format

The CSV files should have the following format:

**english.csv:**
```csv
english,arabic
"Source text in English","Reference translation in Arabic"
...
```

**french.csv:**
```csv
french,arabic
"Source text in French","Reference translation in Arabic"
...
```

## 🐛 Troubleshooting

### "Cannot connect to API"
- Make sure the backend server is running on port 5002
- Check with: `curl http://localhost:5002/api/health`
- Verify firewall settings if running on different hosts

### "All translations failed"
- Verify your API key is configured in `app/.env`
- Check the backend logs for errors
- Test a single translation manually
- Try with smaller sample size first

### "BERTScore/COMET not available"
Install missing dependencies:
```bash
# For BERTScore
pip install bert-score torch transformers

# For COMET (state-of-the-art metric)
pip install unbabel-comet

# For Sentence Transformers
pip install sentence-transformers
```

### "Memory error" or "CUDA out of memory"
For large evaluations with semantic metrics:
```bash
# Reduce batch size by evaluating fewer samples
python evaluate_semantic.py --sample-size 20

# Or disable heavy metrics temporarily
python evaluate_semantic.py --no-comet --sample-size 50
```

### "Too slow" / Performance issues
- Reduce the `sample_size` parameter
- Check your API rate limits
- Increase the `delay` parameter between requests
- Use `--no-comet` flag for faster evaluation
- Consider using GPU for BERTScore/COMET (10x speedup)
- Disable metrics you don't need with `--no-bertscore`, `--no-embeddings`

### First-time COMET is very slow
- COMET downloads model files (~1-2GB) on first use
- Subsequent runs will be much faster
- Models are cached in `~/.cache/huggingface/`
- Consider downloading models in advance for production

## 🔍 Advanced Usage

### Programmatic Usage

Use the evaluators directly in your Python code:

```python
from evaluate_semantic import SemanticTranslationEvaluator

# Initialize
evaluator = SemanticTranslationEvaluator(
    api_url="http://localhost:5002/api/translate"
)

# Load data
sources = ["Hello world", "How are you?"]
references = ["مرحبا بالعالم", "كيف حالك؟"]

# Evaluate with semantic metrics
results = evaluator.evaluate_translations(
    sources, 
    references, 
    source_lang="en",
    use_comet=True,
    use_bertscore=True,
    use_embeddings=True
)

# Access results
print(f"Semantic Score: {results['metrics']['semantic_score']:.2f}")
print(f"COMET: {results['metrics']['comet_score']:.4f}")
print(f"BERTScore F1: {results['metrics']['bertscore_f1']:.4f}")
```

### Comparing Multiple Systems

```python
import json
from pathlib import Path

def compare_systems():
    results_dir = Path("Results")
    
    systems = {
        "system_a": "semantic_english_to_arabic_20260204_100000.json",
        "system_b": "semantic_english_to_arabic_20260204_110000.json"
    }
    
    comparison = {}
    for name, filename in systems.items():
        with open(results_dir / filename) as f:
            data = json.load(f)
            comparison[name] = data['metrics']
    
    # Print comparison
    for metric in ['semantic_score', 'comet_score', 'bertscore_f1']:
        print(f"\n{metric}:")
        for system, metrics in comparison.items():
            if metric in metrics:
                print(f"  {system}: {metrics[metric]:.4f}")
```

### Batch Processing for Large Datasets

For very large datasets (1000+ samples):

```python
def evaluate_large_dataset(csv_path, batch_size=100):
    from evaluate_semantic import SemanticTranslationEvaluator
    import numpy as np
    
    evaluator = SemanticTranslationEvaluator()
    sources, references = evaluator.load_dataset(csv_path, "source", "target")
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i:i+batch_size]
        batch_refs = references[i:i+batch_size]
        
        results = evaluator.evaluate_translations(
            batch_sources, 
            batch_refs,
            source_lang="en"
        )
        all_results.append(results)
        
        # Save intermediate results
        evaluator.save_results(
            results, 
            f"Results/batch_{i//batch_size}.json"
        )
    
    return all_results
```

## 📊 Benchmarking & Reporting Guide

### Recommended Evaluation Protocol

1. **Development Phase** (frequent testing):
   ```bash
   python evaluate_semantic.py --sample-size 20 --no-comet
   ```

2. **Pre-release Testing** (comprehensive check):
   ```bash
   python evaluate_semantic.py --sample-size 100
   ```

3. **Final Evaluation** (publication/production):
   ```bash
   python evaluate_semantic.py  # All samples, all metrics
   ```

### Reporting Evaluation Results

When reporting results in papers or documentation, include:

1. **Primary metrics**: Semantic Score, COMET, BERTScore F1
2. **Sample size** and dataset description
3. **Traditional metrics** (BLEU, CHRF) for comparison with prior work
4. **Model versions**: Which COMET/BERTScore models were used
5. **Statistical significance** (if comparing systems)

**Example report**:
```
Evaluation Results on 500 English→Arabic sentence pairs:

Semantic Metrics (Primary):
- Combined Semantic Score: 81.27/100
- COMET (wmt22-comet-da): 0.7891
- BERTScore F1 (mBERT): 0.8234
- Embedding Similarity (mpnet): 0.8456

Traditional Metrics (Reference):
- BLEU: 28.50
- CHRF+: 52.30
- TER: 42.10

Quality: GOOD - Acceptable semantic quality
```

## 📚 References & Further Reading

### Academic Papers

- **COMET**: Rei et al. (2020) - "COMET: A Neural Framework for MT Evaluation"
  - [Paper](https://arxiv.org/abs/2009.09025)
  - State-of-the-art learned metric with 0.87+ correlation with human judgments
  
- **BERTScore**: Zhang et al. (2020) - "BERTScore: Evaluating Text Generation with BERT"
  - [Paper](https://arxiv.org/abs/1904.09675)
  - Contextual embedding-based similarity metric

- **BLEU**: Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
  - [Paper](https://aclanthology.org/P02-1040/)
  - Industry standard baseline

- **CHRF+**: Popović (2017) - "chrF++: words helping character n-grams"
  - [Paper](https://aclanthology.org/W17-4770/)
  - Better for morphologically rich languages

- **sacrebleu**: Post (2018) - "A Call for Clarity in Reporting BLEU Scores"
  - [Paper](https://arxiv.org/abs/1804.08771)
  - Standardized BLEU implementation

### Useful Resources

- **WMT Metrics Task**: Annual evaluation of MT evaluation metrics
  - [Website](http://www.statmt.org/wmt23/metrics-task.html)
  
- **COMET Documentation**: Official guide and models
  - [GitHub](https://github.com/Unbabel/COMET)
  
- **BERTScore Documentation**: Implementation details
  - [GitHub](https://github.com/Tiiiger/bert_score)

### Why These Metrics Matter for Arabic

1. **Human Correlation**: Semantic metrics correlate 0.80+ with human judgments (vs 0.40-0.60 for BLEU)
2. **Morphology Handling**: Arabic's complex morphology is better captured by embeddings
3. **Synonym Recognition**: Semantic metrics recognize valid paraphrases
4. **Research Standard**: Top MT conferences (WMT, EMNLP) now require semantic metrics

## 🤝 Contributing

To add new evaluation metrics or improve the evaluation pipeline:

1. **Add new metrics**: Implement in `evaluate_semantic.py` or create new script
2. **Update documentation**: Add metric description to this README
3. **Test thoroughly**: Validate with sample data before full evaluation
4. **Benchmark**: Compare with existing metrics to ensure reliability

### Suggested Improvements

- [ ] Add BLEURT metric (another learned metric)
- [ ] Implement confidence intervals for scores
- [ ] Add visualization of results (plots, charts)
- [ ] Support for multi-reference evaluation
- [ ] Integration with MLflow for experiment tracking
- [ ] Automated A/B testing framework

## 🎓 Best Practices

### DO ✅
- **Use semantic metrics** (COMET, BERTScore) as primary quality indicators
- **Report multiple metrics** for comprehensive assessment
- **Include traditional metrics** (BLEU) for comparison with prior work
- **Evaluate on representative samples** (50+ for dev, 500+ for publication)
- **Document your evaluation setup** (sample size, models used, etc.)
- **Consider human evaluation** for critical applications

### DON'T ❌
- **Don't rely solely on BLEU** - it misses valid paraphrases
- **Don't optimize for BLEU alone** - leads to unnatural translations
- **Don't skip semantic evaluation** - it's more accurate than traditional metrics
- **Don't cherry-pick metrics** - report all results honestly
- **Don't ignore failed translations** - they indicate system issues

## 📄 License

This evaluation framework is part of the Context-Specific Machine Translation project.

---

## 💡 Quick Reference Card

```bash
# Fast baseline check (development)
python evaluate_semantic.py --sample-size 20 --no-comet

# Comprehensive evaluation (pre-release)
python evaluate_semantic.py --sample-size 100

# Full evaluation (production/publication)
python evaluate_semantic.py

# English only with all metrics
python evaluate_semantic.py --english-only --sample-size 50

# Traditional metrics only (fastest)
python evaluate_translation.py --sample-size 50
```

**Key Metrics to Report**:
1. **Semantic Score** (0-100): Primary quality indicator
2. **COMET** (0-1): Human judgment correlation
3. **BERTScore F1** (0-1): Contextual similarity
4. **BLEU** (0-100): Industry baseline

**Installation**:
```bash
# Full installation
pip install -r requirements.txt

# Minimal (traditional metrics)
pip install pandas requests sacrebleu
```

---

**Questions?** Check the troubleshooting section or open an issue in the project repository.
