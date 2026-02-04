# Evaluation Scripts Comparison

This document helps you choose the right evaluation script for your needs.

## Quick Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│                  Which Script Should I Use?                 │
└─────────────────────────────────────────────────────────────┘

Need quick baseline?          → evaluate_translation.py
  ├─ BLEU + CHRF only
  ├─ Fastest execution
  └─ Good for rapid iteration

Need balanced evaluation?     → evaluate_enhanced.py --use-bertscore
  ├─ BLEU + CHRF + TER + BERTScore
  ├─ Medium speed
  └─ Good for comprehensive testing

Need production quality?      → evaluate_semantic.py
  ├─ All semantic metrics (COMET, BERTScore, Embeddings)
  ├─ Slower but most accurate
  └─ Best correlation with human judgments
```

## Feature Matrix

| Feature | evaluate_translation.py | evaluate_enhanced.py | evaluate_semantic.py |
|---------|------------------------|---------------------|---------------------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡ Slow |
| **Accuracy** | ⭐⭐ Basic | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **BLEU** | ✅ | ✅ | ✅ |
| **CHRF+** | ✅ | ✅ | ✅ |
| **TER** | ❌ | ✅ | ❌ |
| **BERTScore** | ❌ | ✅ (optional) | ✅ |
| **COMET** | ❌ | ❌ | ✅ |
| **Embeddings** | ❌ | ❌ | ✅ |
| **Semantic Score** | ❌ | ❌ | ✅ |
| **Setup Complexity** | Easy | Easy | Moderate |
| **Dependencies** | Minimal | Medium | Many |
| **Best For** | Development | Testing | Production |

## Detailed Comparison

### evaluate_translation.py (Basic)

**Metrics**: BLEU, CHRF+

**Pros**:
- Very fast execution
- Minimal dependencies
- Easy to set up
- Good for quick iterations

**Cons**:
- Only surface-level metrics
- Poor correlation with human judgments
- Misses semantic equivalence

**Use When**:
- Rapid development cycles
- Quick sanity checks
- Comparing with older systems (that only report BLEU)
- Limited computational resources

**Example**:
```bash
python evaluate_translation.py --sample-size 50
```

---

### evaluate_enhanced.py (Intermediate)

**Metrics**: BLEU, CHRF+, TER, BERTScore (optional)

**Pros**:
- More comprehensive than basic
- BERTScore adds semantic understanding
- TER provides edit distance insights
- Reasonable speed

**Cons**:
- Still missing state-of-the-art metrics (COMET)
- BERTScore is optional
- Not as accurate as full semantic evaluation

**Use When**:
- Pre-release testing
- You need both traditional and some semantic metrics
- Balance between speed and accuracy is important
- You want TER for post-editing estimates

**Example**:
```bash
python evaluate_enhanced.py --sample-size 50 --use-bertscore
```

---

### evaluate_semantic.py (Advanced) ⭐ **RECOMMENDED**

**Metrics**: BLEU, CHRF+, BERTScore, COMET, Sentence Embeddings, Combined Semantic Score

**Pros**:
- State-of-the-art accuracy
- High correlation with human judgments (0.87+ for COMET)
- Captures semantic similarity
- Comprehensive quality assessment
- Best for Arabic evaluation
- Combined semantic score provides single quality metric

**Cons**:
- Slower execution (especially COMET)
- More dependencies to install
- Downloads large model files (~1-2GB on first use)
- May need GPU for large datasets

**Use When**:
- Final production evaluation
- Research papers / publications
- Critical quality assessment needed
- You want most accurate results
- Evaluating Arabic translation (highly recommended)

**Example**:
```bash
# Full evaluation
python evaluate_semantic.py --sample-size 50

# Faster (skip COMET)
python evaluate_semantic.py --sample-size 50 --no-comet
```

## Performance Benchmarks

Based on evaluating 50 English→Arabic samples on a typical laptop:

| Script | Execution Time | Peak Memory | Disk Space |
|--------|---------------|-------------|------------|
| evaluate_translation.py | ~30 seconds | ~200 MB | Minimal |
| evaluate_enhanced.py | ~45 seconds | ~500 MB | ~500 MB (models) |
| evaluate_semantic.py | ~2-3 minutes | ~2 GB | ~2 GB (models) |

*Note: Times include translation API calls. First-time execution of semantic script is slower due to model downloads.*

## Migration Path

### From Basic to Semantic

If you're currently using `evaluate_translation.py`, here's how to upgrade:

```bash
# Step 1: Install additional dependencies
pip install bert-score unbabel-comet sentence-transformers

# Step 2: Run enhanced evaluation first (faster, intermediate)
python evaluate_enhanced.py --sample-size 20 --use-bertscore

# Step 3: Once comfortable, move to full semantic
python evaluate_semantic.py --sample-size 20

# Step 4: For production, use all samples
python evaluate_semantic.py
```

## Recommendations by Use Case

### 🏃 Development (Daily Use)
```bash
python evaluate_translation.py --sample-size 20
```
- Quick feedback loop
- Enough to catch major regressions

### 🧪 Testing (Weekly/Pre-Release)
```bash
python evaluate_enhanced.py --sample-size 50 --use-bertscore
```
- More comprehensive
- Catches semantic issues

### 🎯 Production (Release/Publication)
```bash
python evaluate_semantic.py --sample-size 100
# Or for final evaluation:
python evaluate_semantic.py  # All samples
```
- Most accurate
- Required for research papers
- Best quality assurance

### 📊 Research Papers
```bash
python evaluate_semantic.py  # No sample limit
```
- Report all metrics
- Include COMET, BERTScore, and traditional metrics
- State-of-the-art evaluation

## Cost-Benefit Analysis

### evaluate_translation.py
- **Time Cost**: Very Low (30s for 50 samples)
- **Accuracy**: Low-Medium (BLEU correlation ~0.40)
- **Best ROI**: Early development

### evaluate_enhanced.py
- **Time Cost**: Low-Medium (45s for 50 samples)
- **Accuracy**: Medium-High (BERTScore correlation ~0.70)
- **Best ROI**: Pre-release testing

### evaluate_semantic.py
- **Time Cost**: Medium-High (2-3 min for 50 samples)
- **Accuracy**: Very High (COMET correlation ~0.87)
- **Best ROI**: Final evaluation, production releases

## Frequently Asked Questions

### Q: Can I use multiple scripts?
**A**: Yes! Use different scripts at different stages:
- Development: `evaluate_translation.py`
- Testing: `evaluate_enhanced.py`
- Production: `evaluate_semantic.py`

### Q: Do I need GPU?
**A**: No, all scripts work on CPU. GPU speeds up semantic metrics significantly but isn't required.

### Q: What if I don't have time for semantic evaluation?
**A**: At minimum, use `evaluate_enhanced.py --use-bertscore`. It's much better than BLEU alone and still reasonably fast.

### Q: Which metric should I optimize for?
**A**: 
1. **Primary**: Combined Semantic Score (from `evaluate_semantic.py`)
2. **Secondary**: COMET score
3. **Reference only**: BLEU (for comparison with older systems)

### Q: Can I skip traditional metrics?
**A**: While semantic metrics are more accurate, keep traditional metrics for:
- Comparison with prior work
- Benchmarking against published results
- Industry standard reporting

## Summary

| Priority | Script | Command |
|----------|--------|---------|
| **1st Choice** | evaluate_semantic.py | `python evaluate_semantic.py --sample-size 50` |
| **2nd Choice** | evaluate_enhanced.py | `python evaluate_enhanced.py --sample-size 50 --use-bertscore` |
| **3rd Choice** | evaluate_translation.py | `python evaluate_translation.py --sample-size 50` |

**Bottom Line**: For Arabic translation evaluation, semantic metrics (especially COMET) are significantly more accurate than BLEU. If you can only run one evaluation, use `evaluate_semantic.py`.
