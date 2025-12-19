# Generation Methods Configuration Guide

## Quick Start

### Switch Between Data Sources
In Cell 7 (Load Training Data):
```python
USE_SAMPLES = False  # Change to True for samples, False for full data
```

### Generation Methods Reference

All 4 methods are automatically applied to each source sentence.

## Method Details

### 1. Temperature Sampling (High Diversity)
**Best For:** Exploring diverse translation options, creating varied candidates

**Configuration:**
- `temperature`: 1.2 (higher = more random)
- `top_p`: 0.95 (nucleus cutoff)
- `top_k`: 50 (conservative top-k limit)
- `do_sample`: True

**Output Pattern:**
```jsonl
{"source": "...", "method": "temperature_sampling", "candidates": [...]}
```

**Characteristics:**
- Most diverse outputs
- May include lower quality translations
- Good for bootstrapping diverse training data

---

### 2. Top-K Sampling (Conservative)
**Best For:** Balanced diversity and quality, controlled sampling

**Configuration:**
- `temperature`: 0.7 (lower = more conservative)
- `top_k`: 30 (strict top-k cutoff)
- `top_p`: 0.9 (nucleus cutoff)
- `do_sample`: True

**Output Pattern:**
```jsonl
{"source": "...", "method": "top_k_sampling", "candidates": [...]}
```

**Characteristics:**
- Balanced between diversity and consistency
- Avoids very low probability tokens
- Practical for most use cases

---

### 3. Nucleus Sampling (Balanced)
**Best For:** Standard text generation, recommended method

**Configuration:**
- `temperature`: 0.9 (moderate)
- `top_p`: 0.95 (nucleus probability)
- `top_k`: 0 (disabled)
- `do_sample`: True

**Output Pattern:**
```jsonl
{"source": "...", "method": "nucleus_sampling", "candidates": [...]}
```

**Characteristics:**
- Industry-standard approach (used in ChatGPT, etc.)
- Good quality-diversity balance
- Well-established in research

---

### 4. Greedy Decoding (Deterministic)
**Best For:** Consistent outputs, baseline comparisons

**Configuration:**
- `do_sample`: False (deterministic)
- No temperature, top_k, or top_p applied

**Output Pattern:**
```jsonl
{"source": "...", "method": "greedy_decoding", "candidates": [...]}
```

**Characteristics:**
- Always produces the same output for same input
- Highest confidence translations (but may be narrow)
- Good for baselines and consistency testing

---

## Output File Formats

### Per-Method Files
**File:** `generated_candidates_<method_name>.jsonl`

```jsonl
{
  "source": "The question of financial assistance...",
  "source_lang": "en",
  "method": "temperature_sampling",
  "candidates": ["المسألة المتعلقة بتقديم المساعدة المالية...", ...]
}
```

### Combined File
**File:** `generated_candidates_all_methods.jsonl`

```jsonl
{
  "source": "The question of financial assistance...",
  "source_lang": "en",
  "candidates_by_method": {
    "temperature_sampling": ["translation1", "translation2", ...],
    "top_k_sampling": ["translation3", ...],
    "nucleus_sampling": ["translation4", ...],
    "greedy_decoding": ["translation5", ...]
  }
}
```

### Preference Pairs File
**File:** `synthetic_preferences.jsonl`

```jsonl
{
  "source": "The question of financial assistance...",
  "source_lang": "en",
  "chosen": "أفضل ترجمة",
  "rejected": "ترجمة أقل جودة",
  "chosen_score": 0.8234,
  "rejected_score": 0.6102,
  "margin": 0.2132,
  "chosen_method": "nucleus_sampling",
  "rejected_method": "temperature_sampling"
}
```

---

## Statistics Output Format

**File:** `synthetic_data_stats.json`

```json
{
  "total_pairs": 50000,
  "en_pairs": 25000,
  "fr_pairs": 25000,
  "avg_margin": 0.2145,
  "avg_chosen_score": 0.8102,
  "avg_rejected_score": 0.5957,
  "num_sources": 100000,
  "language_breakdown": {
    "english": {
      "pairs": 25000,
      "avg_margin": 0.2134,
      "avg_chosen_score": 0.8145
    },
    "french": {
      "pairs": 25000,
      "avg_margin": 0.2156,
      "avg_chosen_score": 0.8059
    }
  },
  "method_breakdown": {
    "temperature_sampling": {
      "pairs_as_chosen": 10000,
      "pairs_as_rejected": 10000,
      "avg_comet": 0.7834,
      "total_candidates": 100000
    },
    "top_k_sampling": {
      "pairs_as_chosen": 12500,
      "pairs_as_rejected": 7500,
      "avg_comet": 0.8012,
      "total_candidates": 100000
    },
    "nucleus_sampling": {
      "pairs_as_chosen": 15000,
      "pairs_as_rejected": 5000,
      "avg_comet": 0.8234,
      "total_candidates": 100000
    },
    "greedy_decoding": {
      "pairs_as_chosen": 12500,
      "pairs_as_rejected": 27500,
      "avg_comet": 0.7965,
      "total_candidates": 100000
    }
  },
  "data_source": "full"
}
```

---

## Performance Analysis

### Interpreting Method Metrics

**COMET Score:**
- Higher = better quality
- Range: typically 0.0-1.0
- Used by machine translation researchers

**Win Rate (pairs_as_chosen ratio):**
- Reflects how often method was selected as preferred candidate
- Calculated: `chosen / (chosen + rejected)`
- Higher = method produces higher quality translations according to COMET

### Example Analysis

```
Temperature Sampling: 50% win rate (10K chosen, 10K rejected)
├─ Interpretation: Balanced performance, comparable to average
├─ COMET: 0.7834 (below average quality)
└─ Use Case: Exploratory data, high diversity

Top-K Sampling: 62.5% win rate (12.5K chosen, 7.5K rejected)
├─ Interpretation: Good performer, above average
├─ COMET: 0.8012 (average quality)
└─ Use Case: Balanced approach, practical deployments

Nucleus Sampling: 75% win rate (15K chosen, 5K rejected)
├─ Interpretation: Best performer, highest quality
├─ COMET: 0.8234 (highest quality)
└─ Use Case: Production use, recommended

Greedy Decoding: 31.25% win rate (12.5K chosen, 27.5K rejected)
├─ Interpretation: Poor performer, conservative/narrow
├─ COMET: 0.7965 (below average quality)
└─ Use Case: Baseline/consistency checks only
```

---

## Customization

### Modify Method Parameters

To adjust method parameters, edit the generation functions in Cell 10:

```python
def generate_with_temperature(sources, langs, num_candidates=1):
    """Edit these values:"""
    outputs = model.generate(
        ...,
        temperature=1.2,  # ← Change this (0.0 = greedy, > 1.0 = more random)
        top_p=0.95,       # ← Change this (nucleus probability cutoff)
        top_k=50,         # ← Change this (keep top K tokens, 0 = disabled)
        ...
    )
```

### Add/Remove Methods

To add a new method:
1. Create a new function `generate_with_<method_name>()`
2. Add to `GENERATION_METHODS` dictionary
3. Update main loop to process new method

---

## FAQ

**Q: Why 4 candidates per source instead of other numbers?**
A: 4 provides good diversity while maintaining computational efficiency. This matches common practice in preference learning (2-4 candidates). Adjust `NUM_CANDIDATES` if needed.

**Q: Which method should I use?**
A: Based on statistics, nucleus_sampling typically performs best. Use multiple methods to capture diverse perspectives in training data.

**Q: Can I compare methods fairly?**
A: Yes! Each method processes the same sources and gets scored by the same COMET model. Method-specific COMET scores are directly comparable.

**Q: How much slower is 4 methods vs 1 method?**
A: Approximately 4× slower for generation, but COMET scoring is still batched across all methods.

**Q: Do all 4 methods work with samples and full data?**
A: Yes, USE_SAMPLES parameter is independent of method selection.

---

## Advanced Usage

### Filter by Method Quality

```python
import json

with open('data/synthetic_preferences.jsonl') as f:
    pairs = [json.loads(line) for line in f]

# Get only nucleus sampling chosen candidates
nucleus_chosen = [p for p in pairs if p['chosen_method'] == 'nucleus_sampling']
print(f"Nucleus sampling won {len(nucleus_chosen)}/{len(pairs)} times")
```

### Language-Specific Method Performance

```python
import json

with open('data/synthetic_data_stats.json') as f:
    stats = json.load(f)

# Check if methods perform differently for EN vs FR
en_stats = stats['language_breakdown']['english']
fr_stats = stats['language_breakdown']['french']

print(f"EN avg score: {en_stats['avg_chosen_score']:.4f}")
print(f"FR avg score: {fr_stats['avg_chosen_score']:.4f}")
```

### Audit Method Consistency

```python
import json

with open('data/generated_candidates_greedy_decoding.jsonl') as f:
    candidates = [json.loads(line) for line in f]

# Check for duplicate outputs (greedy should have many duplicates)
translations = [c['candidates'][0] for c in candidates[:1000]]
unique = len(set(translations))
print(f"Greedy diversity: {unique}/1000 unique (expected low)")
```
