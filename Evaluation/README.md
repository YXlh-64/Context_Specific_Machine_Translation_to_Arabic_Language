# Translation System Evaluation

This directory contains scripts and data for evaluating the translation system using BLEU and CHRF+ metrics.

## 📁 Directory Structure

```
Evaluation/
├── Data/
│   ├── english.csv    # English to Arabic test data
│   └── french.csv     # French to Arabic test data
├── Results/           # Evaluation results (auto-generated)
├── evaluate_translation.py  # Main evaluation script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Evaluation
pip install -r requirements.txt
```

### 2. Start the Translation API

Make sure your translation API backend is running:

```bash
cd ../app
./run_backend.sh
```

The API should be accessible at `http://localhost:5002`

### 3. Run the Evaluation

```bash
python evaluate_translation.py
```

## 📊 Metrics

The evaluation script computes two standard machine translation metrics:

### BLEU (Bilingual Evaluation Understudy)
- Measures n-gram precision between hypothesis and reference translations
- Range: 0-100 (higher is better)
- Widely used industry standard

### CHRF+ (Character n-gram F-score)
- Character-level metric that's more robust to morphological variations
- Range: 0-100 (higher is better)
- Particularly useful for Arabic which has rich morphology

## 🔧 Configuration

### Evaluation Sample Size

By default, the script evaluates all samples in the datasets. To test with a smaller sample:

Edit `evaluate_translation.py` and change the `sample_size` parameter:

```python
en_results = evaluator.evaluate_translations(
    en_sources, 
    en_references, 
    source_lang="en",
    sample_size=20  # Evaluate only 20 samples for testing
)
```

### API URL

If your API is running on a different host/port, update the `api_url` parameter:

```python
evaluator = TranslationEvaluator(api_url="http://your-host:port/api/translate")
```

## 📈 Output

The script generates several outputs in the `Results/` directory:

1. **Summary JSON** (`{language_pair}_{timestamp}.json`):
   - Overall metrics (BLEU, CHRF+)
   - Sample counts
   - Metadata

2. **Detailed JSON** (`{language_pair}_{timestamp}_detailed.json`):
   - Source texts
   - Hypothesis translations
   - Reference translations
   - Useful for error analysis

3. **Console Output**:
   - Real-time progress
   - Final metrics summary

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

### "All translations failed"
- Verify your API key is configured in `app/.env`
- Check the backend logs for errors
- Test a single translation manually

### "Memory error" or "Too slow"
- Reduce the `sample_size` parameter
- Check your API rate limits
- Increase the `delay` parameter between requests

## 🔍 Advanced Usage

### Custom Evaluation

You can use the `TranslationEvaluator` class programmatically:

```python
from evaluate_translation import TranslationEvaluator

evaluator = TranslationEvaluator(api_url="http://localhost:5002/api/translate")

# Load your data
sources = ["Hello world", "How are you?"]
references = ["مرحبا بالعالم", "كيف حالك؟"]

# Evaluate
results = evaluator.evaluate_translations(sources, references, source_lang="en")
print(f"BLEU: {results['bleu_score']:.2f}")
print(f"CHRF+: {results['chrf_score']:.2f}")
```

### Batch Processing

For very large datasets, consider:
1. Splitting the data into chunks
2. Running evaluations in parallel
3. Combining results afterwards

## 📚 References

- **BLEU**: Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
- **CHRF+**: Popović (2017) - "chrF++: words helping character n-grams"
- **sacrebleu**: Post (2018) - "A Call for Clarity in Reporting BLEU Scores"

## 🤝 Contributing

To add new evaluation metrics or improve the evaluation pipeline:

1. Add the metric implementation to `evaluate_translation.py`
2. Update this README with usage instructions
3. Test with sample data before running full evaluation

## 📄 License

This evaluation framework is part of the Context-Specific Machine Translation project.
