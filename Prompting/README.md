# Translation Orchestrator - Complete Guide

A Human-in-the-Loop (HITL) multi-agent translation system that generates **3 different Arabic translation variants** with official evaluation metrics (chrF++, BLEU, COMET).

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key in .env
GROQ_API_KEY=your-key-here

# 3. Run the demo
python demo.py
```

The demo will:
- Generate 3 translation variants using different strategies
- Evaluate them with chrF++, BLEU, and COMET metrics
- Compare against ground truth translations
- Create an HTML report (`demo_{domain}.html`)

## What This System Does

### Three Translation Strategies

The system generates **3 distinct translations** using different prompt strategies and temperature settings:

| Agent | Temperature | Strategy | Best For |
|-------|-------------|----------|----------|
| **Context-Aware** | 0.8 (creative) | Natural, fluent Arabic with idiomatic expressions | End-user facing content |
| **Terminology-Optimized** | 0.4 (balanced) | Strong glossary enforcement with professional precision | Technical documentation |
| **Conservative** | 0.1 (literal) | Literal translation preserving source structure | Legal/clinical precision |

### Evaluation Metrics (Official)

Each translation is evaluated against ground truth using:

- **chrF++** (60% weight): Character-level F-score, excellent for morphologically-rich languages like Arabic
- **BLEU** (40% weight): Word-level precision metric
- **COMET** (50% weight): Neural semantic similarity (requires model download)

**Quality Levels:**
- 90-100%: Excellent ⭐⭐⭐⭐⭐
- 80-89%: Very Good ⭐⭐⭐⭐
- 70-79%: Good ⭐⭐⭐
- 60-69%: Acceptable ⭐⭐
- 50-59%: Poor ⚠️
- <50%: Very Poor ❌

### Iterative Refinement Loop

The system can automatically improve translations through an iterative refinement loop (configurable in `demo.py`):

```
1. Plan → 2. Generate & Validate → 3. Critique & Evaluate
                    ↑                           ↓
                    └───── Loop if needed ──────┘
```

**Loop exits when:**
- Quality targets met (default: 7.5/10 quality, 85% glossary compliance)
- No improvement detected
- Max iterations reached (default: 3)

## Configuration

### Basic Settings (demo.py)

```python
# Domain and task
DOMAIN = "medical"  # or "technology", "education", "economic"
TASK_NUMBER = 1     # Which task from CSV (1-20 for medical)

# Iterative refinement
ENABLE_ITERATIVE_REFINEMENT = True  # Set False for single-pass
MAX_ITERATIONS = 3
MIN_QUALITY_THRESHOLD = 7.5
MIN_GLOSSARY_COMPLIANCE = 85.0
```

### Model Configuration

**Default Model:** Groq LLaMA 3.3 (`llama-3.3-70b-versatile`) - FREE

**Supported Models:**
- Groq: `llama-3.3-70b-versatile` (recommended, free)
- OpenAI: `gpt-4-turbo`, `gpt-4o`
- Anthropic: `claude-3-5-sonnet-20240620`

**To change model:** Edit `OrchestratorConfig` in `demo.py`:
```python
config = OrchestratorConfig(default_model="gpt-4-turbo")
```

### Temperature Settings (src/models.py)

```python
context_aware_temperature = 0.8    # More creative, fluent
terminology_temperature = 0.4      # Balanced
conservative_temperature = 0.1     # Literal, deterministic
```

## Setup Details

### 1. API Keys

Create a `.env` file in the project root:

```env
# Required (choose one or more)
GROQ_API_KEY=your-groq-key-here          # Get from: https://console.groq.com
OPENAI_API_KEY=your-openai-key-here      # Get from: https://platform.openai.com
ANTHROPIC_API_KEY=your-anthropic-key     # Get from: https://console.anthropic.com

# Model selection
DEFAULT_MODEL=llama-3.3-70b-versatile
```

**Getting a Groq API Key (FREE):**
1. Go to https://console.groq.com
2. Sign up with email/Google/GitHub
3. Navigate to "API Keys"
4. Click "Create API Key"
5. Copy and paste into `.env`

### 2. COMET Model (Optional)

COMET provides the most accurate semantic evaluation but requires downloading a 1-2 GB model.

**Option 1: Automatic download (first run)**
```powershell
python demo.py  # COMET downloads automatically
```

**Option 2: Pre-download**
```powershell
python download_comet.py
```

**Option 3: Skip COMET**
If you don't download COMET, the system still works with chrF++ and BLEU metrics. COMET scores will show as "N/A".

### 3. Data Files

The system uses official evaluation data from the `Evaluation/` folder:

```
Evaluation/
├── translation_evaluator.py              # Evaluation logic
└── Augmented Queries Samples/
    ├── medical_examples.csv              # Input queries
    ├── medical_translation_template.csv  # Ground truth
    ├── technology_examples.csv
    ├── technology_translation_template.csv
    ├── education_examples.csv
    ├── education_translation_template.csv
    ├── economic_examples.csv
    └── economic_translation_template.csv
```

## Project Structure

```
Orchestrator/
├── demo.py                    # Main demo script - START HERE
├── download_comet.py          # COMET model downloader
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
├── README.md                  # This file
│
├── src/                       # Core system modules
│   ├── orchestrator.py        # Main orchestration logic
│   ├── llm_client.py          # LLM provider abstraction
│   ├── prompt_builder.py      # Prompt strategies per agent
│   ├── models.py              # Data models and config
│   ├── parser.py              # CSV parsing logic
│   ├── validator.py           # Translation validation
│   └── critique.py            # Quality scoring and critique
│
├── Evaluation/                # Official evaluation system
│   ├── translation_evaluator.py
│   └── Augmented Queries Samples/
│
└── tests/                     # Unit tests
    ├── test_orchestrator.py
    ├── test_parser.py
    └── test_prompts.py
```

## How It Works

### 1. Input Processing

The system reads from CSV files containing:
- **Source text** (English)
- **Domain** (medical, technology, education, economic)
- **Glossary terms** (term pairs: English ↔ Arabic)
- **Similar examples** (fuzzy matches from RAG system)
- **Ground truth** (reference translation for evaluation)

### 2. Prompt Construction

Three different prompts are built for each agent:

**Context-Aware Prompt:**
```
Strategy: Natural, fluent Arabic
Glossary: Reference only (not mandatory)
Examples: Formatted to emphasize fluency
Priority: Readability > Literalness
```

**Terminology-Optimized Prompt:**
```
Strategy: Glossary enforcement
Glossary: MANDATORY terms
Examples: Show consistent term usage
Priority: Accuracy = Readability
```

**Conservative Prompt:**
```
Strategy: Literal fidelity
Glossary: Strict accuracy reference
Examples: Word-for-word alignment
Priority: Fidelity > Naturalness
```

### 3. Parallel Generation

All 3 translations are generated in parallel for speed:
```python
async with LLMClient():
    translations = await generate_all_three()
```

### 4. Validation & Critique

Each translation is:
- Validated (structure, length, glossary)
- Scored (quality 0-10, glossary compliance 0-100%)
- Critiqued (strengths, weaknesses, recommendations)

### 5. Evaluation (Official Metrics)

Translations are compared against ground truth:
```python
chrF++ = character_level_fscore()
BLEU = word_level_precision()
COMET = neural_semantic_similarity()
Combined = weighted_average()
```

### 6. HTML Report Generation

A comprehensive HTML report is created with:
- Source text and ground truth
- All 3 translation variants
- Evaluation metrics per variant
- Quality level classification
- Recommended translation
- Iteration metadata (if refinement enabled)

## Advanced Usage

### Running Tests

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_orchestrator.py

# Verbose output
pytest -v
```

### Customizing the Orchestrator

**Change quality thresholds:**
```python
config = OrchestratorConfig(
    min_quality_threshold=8.0,           # Stricter quality
    min_glossary_compliance=90.0,        # Higher compliance
    max_iterations=5                     # More attempts
)
```

**Disable parallel generation:**
```python
config = OrchestratorConfig(
    enable_parallel_generation=False     # Sequential instead
)
```

**Adjust temperature spread:**
Edit `src/models.py`:
```python
context_aware_temperature = 1.0     # More creative
terminology_temperature = 0.5       # Balanced
conservative_temperature = 0.0      # Fully deterministic
```

### Integration with Your RAG System

The orchestrator expects RAG output in this format:

```python
from src.models import RAGOutput, GlossaryMatch, FuzzyMatch

rag_output = RAGOutput(
    source_text="Your English text",
    domain="medical",
    glossary_matches=[
        GlossaryMatch(
            source_term="fever",
            target_term="حمى"
        )
    ],
    fuzzy_matches=[
        FuzzyMatch(
            source_text="Similar example in English",
            target_text="Translation in Arabic",
            similarity_score=0.85
        )
    ]
)

result = await orchestrator.orchestrate(rag_output)
```

See `src/parser.py` for CSV parsing examples.

## Troubleshooting

### COMET Shows "N/A"

**Cause:** COMET model not downloaded

**Fix:**
```powershell
python download_comet.py
```

Or skip COMET (chrF++ and BLEU still work).

### "Groq client not initialized"

**Cause:** API key not loaded

**Fix:** Ensure `.env` contains `GROQ_API_KEY` and `src/llm_client.py` calls `load_dotenv()` (already included).

### Translations Are Identical

**Cause:** Temperature spread too narrow

**Fix:** Increase temperature differences in `src/models.py`:
```python
context_aware_temperature = 0.9     # Was 0.8
terminology_temperature = 0.4       # Keep same
conservative_temperature = 0.1      # Keep same
```

### CSV Parsing Warnings

**Cause:** Malformed rows in example CSVs

**Fix:** These are warnings only and don't block execution. To clean:
1. Open the CSV file
2. Remove empty rows or fix formatting
3. Re-save with UTF-8 encoding

### Iterations Always 0

**Cause:** Quality scores exceed thresholds immediately

**Expected:** If translations meet quality targets on first attempt, no refinement is needed. To force iterations:
```python
MIN_QUALITY_THRESHOLD = 9.5    # Very strict
MIN_GLOSSARY_COMPLIANCE = 95.0
```

## Performance Tips

1. **Use Groq for speed:** LLaMA 3.3 on Groq is 5-10x faster than OpenAI/Anthropic
2. **Enable parallel generation:** Default is on, but verify in config
3. **Reduce max_iterations:** If quality is consistently high, set to 1
4. **Skip COMET:** If you only need chrF++ and BLEU, don't download COMET

## Dependencies

Core requirements (from `requirements.txt`):
```
openai>=1.3.0
anthropic>=0.7.0
groq>=0.4.0
pydantic>=2.5.0
python-dotenv>=1.0.0
sacrebleu>=2.3.0         # chrF++, BLEU
unbabel-comet>=2.0.0     # COMET (optional)
```

## Architecture Details

### Three-Agent Design

Each agent has:
- **Unique prompt strategy** (see `src/prompt_builder.py`)
- **Different temperature** (see `src/models.py`)
- **Specialized focus** (fluency vs. terminology vs. fidelity)

This creates meaningful variation without requiring multiple model calls with completely different instructions.

### Iterative Refinement

The refinement loop (see `src/orchestrator.py`):
1. **Plan:** Analyze input, set targets
2. **Generate:** Create translations with feedback
3. **Evaluate:** Score and check thresholds
4. **Decide:** Continue if below target, exit if met or no improvement

### Evaluation Pipeline

Two-layer evaluation:
1. **Internal heuristics** (fast, 0-10 scale) - for HITL interface
2. **Official metrics** (slower, research-grade) - for final assessment

## Citation

If you use this system in research or production:

```
Translation Orchestrator with Iterative Refinement
Multi-agent prompt engineering for Arabic translation
Using LLaMA 3.3 (Groq), chrF++, BLEU, COMET evaluation
November 2025
```

## Support

For issues, questions, or contributions:
1. Check this README
2. Review code comments in `src/` modules
3. Run tests: `pytest -v`
4. Check `.env` configuration

## License

[Add your license here]

## Changelog

### v2.0 (November 2025)
- ✅ Added iterative refinement loop
- ✅ Upgraded to LLaMA 3.3 (from deprecated 3.1)
- ✅ Increased temperature spread for better differentiation
- ✅ Added iteration metadata to HTML reports
- ✅ Fixed .env loading in llm_client

### v1.0 (Initial Release)
- ✅ Three-agent translation system
- ✅ Official evaluation metrics (chrF++, BLEU, COMET)
- ✅ HTML report generation
- ✅ Multi-provider support (Groq/OpenAI/Anthropic)
- ✅ Ground truth comparison
