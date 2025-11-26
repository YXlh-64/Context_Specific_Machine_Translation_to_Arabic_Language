# RAG Translation Agent - Context-Specific Machine Translation to Arabic<<<<<<< HEAD

# RAG Translation Agent - Context-Specific Machine Translation to Arabic

A Retrieval-Augmented Generation (RAG) system for English-to-Arabic translation using **local Mistral-7B-Instruct** with context-aware translation. Optimized for 3.6GB VRAM using 4-bit quantization.

A Retrieval-Augmented Generation (RAG) system for English-to-Arabic translation using **local Mistral-7B-Instruct** with context-aware translation.

---

## 🎯 Project Overview

## 🎯 Overview## 🎯 What is This?

This project implements a **Retrieval-Augmented Generation (RAG)** system for context-aware English-to-Arabic translation, specifically optimized for economic and financial domain texts. The system combines semantic search with neural machine translation to deliver domain-specific, consistent, and accurate translations.

This project implements a **RAG-based translation system** that combines:



- **Vector Database (FAISS)** - Stores translation examples from economic/financial domain

- **Sentence Embeddings** - Semantic similarity search to find relevant contextThis system combines:### What Makes This "Agentic"?

- **Local Mistral-7B-Instruct** - 4-bit quantized LLM for high-quality translation

- **RAG Pipeline** - Retrieves similar examples to improve translation accuracy- **Vector Database** (FAISS) with 1.8M+ economic translation pairs



### Key Features- **Sentence Embeddings** for semantic similarity searchThe **agentic** nature of this system lies in its autonomous decision-making and multi-step reasoning process:



✅ **100% Local** - No API keys, runs entirely on your machine  - **Google Gemini API** for context-aware translation

✅ **Memory Efficient** - 4-bit quantization fits in 3.6GB VRAM  

✅ **Context-Aware** - Uses similar translations to improve quality  - **RAG Pipeline** that retrieves similar examples to improve translation quality1. **Autonomous Context Retrieval**: The agent independently determines which translation examples are most relevant by:

✅ **Evaluation Ready** - Built-in BLEU and chrF metrics  

✅ **Domain-Specific** - Optimized for economic/financial texts     - Semantically analyzing the input text



---## 🚀 Quick Start   - Querying the vector database for similar examples



## 🚀 Quick Start   - Ranking and selecting the most contextually appropriate examples



### 1. Installation### 1. Installation



```bash2. **Adaptive Translation Strategy**: Based on retrieved context, the agent:

cd ATA/RAG_Retrieval

```bash   - Dynamically adjusts its translation approach

# Install dependencies

pip install -r requirements.txt# Install dependencies   - Incorporates domain-specific terminology

```

pip install -r requirements.txt   - Maintains consistency with similar past translations

**What gets installed:**

- PyTorch with CUDA support```

- Transformers + BitsAndBytes (4-bit quantization)

- Sentence transformers (embeddings)3. **Multi-Step Pipeline Orchestration**: The agent autonomously coordinates:

- FAISS (vector database)

- sacrebleu (evaluation metrics)### 2. Get Gemini API Key   - Text normalization and preprocessing



### 2. Build Vector Database   - Semantic retrieval operations



```bash1. Visit: https://makersuite.google.com/app/apikey   - Context formatting and prompt construction

# Index 10,000 translation examples (recommended for testing)

python build_vector_db.py --sample 100002. Create a free API key   - Translation generation with appropriate parameters



# Or use all available data:3. Set environment variable:   - Post-processing and result validation

# python build_vector_db.py

```



**Expected time:** 2-5 minutes  ```bash4. **Self-Optimization**: The system can:

**Output:** Vector database in `./vector_db/`

export GOOGLE_API_KEY="your_api_key_here"   - Filter results based on domain metadata

### 3. Test Translation

```   - Adjust retrieval parameters (top-k, similarity thresholds)

```bash

python test_translation.py   - Select between different translation models

```

### 3. Build Vector Database (First Time Only)   - Apply quantization strategies based on available resources

**What it does:**

- Loads vector database with translation examples

- Downloads & quantizes Mistral-7B (first run only, ~13GB)

- Translates test sentences WITH and WITHOUT RAG context```bashThis goes beyond a simple translation API by acting as an intelligent agent that makes contextual decisions at each step to optimize translation quality.

- Shows retrieved examples and improvements

python build_vector_db.py --config config.ini --sample 10000

**First run:** 60-90 seconds (model download)  

**Subsequent runs:** 10-20 seconds```---



### 4. Launch Web Interface



```bashThis indexes translation examples from `economic v1.csv` for retrieval.## 📊 Data Source

streamlit run app.py

```



Open browser at `http://localhost:8501` for interactive translation!### 4. Test Translation### Economic v1.csv - UN Parallel Corpus



---



## 📊 Model Configuration```bashThe system is trained and retrieves context from `economic v1.csv`, a parallel corpus containing:



### Current Model: **Mistral-7B-Instruct-v0.2**python test_gemini.py



Configured in `config.ini`:```- **Source**: English text from UN documents (economic and financial domain)



```ini- **Target**: Professional Arabic translations

[TRANSLATION]

model_name = mistralai/Mistral-7B-Instruct-v0.2### 5. Run Evaluation- **Size**: Contains thousands of parallel sentence pairs

device = cuda

use_4bit = true          # Critical for 3.6GB VRAM- **Domain**: Economic, financial, legal, and political terminology

max_length = 256

temperature = 0.3        # Low = deterministic, high = creative```bash- **Structure**:

top_p = 0.9

```# Extract test samples  - `id`: Unique identifier for each translation pair



### Alternative Modelspython extract_test_samples.py --num_samples 100  - `en`: English source text



You can change the model by editing `config.ini`:  - `ar`: Arabic target translation



```ini# Compare RAG vs non-RAG

# Option 1: Llama-2 (requires HuggingFace token)

model_name = meta-llama/Llama-2-7b-chat-hfpython compare_rag.py \**Example entries**:



# Option 2: Zephyr (good alternative)    --api_key $GOOGLE_API_KEY \```csv

model_name = HuggingFaceH4/zephyr-7b-beta

    --test_file test_samples.csv \id,en,ar

# Option 3: Your fine-tuned model

model_name = ../LLMs/QWEN/outputs/your_model    --sample 1001,"The question of financial assistance to defendants unable to fund their defence...","وهناك حاجة أيضا الى النظر في مسألة تقديم المساعدة المالية..."

```

```9,"It is estimated that the cost of maintaining UNFICYP...","تشير التقديرات الى أن تكاليف اﻹبقاء على قوة..."

**For Llama-2:** Run `huggingface-cli login` first

```

---

## 📊 How It Works

## 🧪 Evaluation & Comparison

This high-quality parallel corpus serves as:

### Create Test Dataset

### RAG Translation Pipeline1. **Training data** for building the vector database

```bash

python extract_test_samples.py --num_samples 100 --output test_data.csv2. **Retrieval source** for finding similar translation contexts

```

```3. **Quality reference** for domain-specific terminology

### Compare RAG vs Non-RAG

Input Text

```bash

python compare_rag.py \    ↓---

    --test_file test_data.csv \

    --sample 50 \1. Normalize & Preprocess

    --output_dir ./evaluation_results \

    --top_k 3    ↓## 🏗️ System Architecture

```

2. Generate Embedding (Sentence-BERT)

**What this does:**

1. Translates all samples **without RAG** (baseline)    ↓```

2. Translates same samples **with RAG** (context-aware)

3. Calculates BLEU and chrF scores3. Search Vector DB (FAISS)┌──────────────────────────────────────────────────────────────────┐

4. Shows improvement metrics

5. Saves detailed results    ↓│                         User Interface Layer                      │



**Expected Output:**4. Retrieve Top-K Similar Examples│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐ │



```    ↓│  │  Streamlit UI  │  │   Python API   │  │  CLI Scripts      │ │

📊 RESULTS COMPARISON

============================================================5. Build Context Prompt with Examples│  │   (app.py)     │  │ (rag_agent.py) │  │ (example.py)      │ │



🔵 WITHOUT RAG (Baseline):    ↓│  └────────────────┘  └────────────────┘  └───────────────────┘ │

   BLEU Score:  28.45

   chrF Score:  52.306. Send to Gemini API└──────────────────────────────────────────────────────────────────┘

   Successful:  50/50

    ↓                              ↓

🟢 WITH RAG (Context-Aware):

   BLEU Score:  34.787. Get Context-Aware Translation┌──────────────────────────────────────────────────────────────────┐

   chrF Score:  58.92

   Successful:  50/50    ↓│                     RAG Agent (Orchestrator)                      │



📈 IMPROVEMENT:Output Translation│                         rag_agent.py                              │

   BLEU: +6.33 points (+22.3%)

   chrF: +6.62 points (+12.7%)```│  ┌──────────────────────────────────────────────────────────┐  │

```

│  │  1. Text Normalization                                    │  │

**Results saved to:**

- `evaluation_results/comparison_summary_*.json` - Metrics### Example│  │  2. Context Retrieval (semantic search)                   │  │

- `evaluation_results/detailed_translations_*.csv` - All translations

│  │  3. Prompt Construction (with retrieved examples)         │  │

---

**Without RAG (Baseline)**:│  │  4. Translation Generation                                │  │

## 📁 Project Structure

```│  │  5. Result Post-processing                                │  │

```

RAG_Retrieval/Input: "The economic growth rate increased significantly."│  └──────────────────────────────────────────────────────────┘  │

├── rag_agent.py              # Core translation agent (Mistral-7B)

├── vector_db.py              # Vector database manager (FAISS)Output: [Generic translation]└──────────────────────────────────────────────────────────────────┘

├── utils.py                  # Utility functions

├── config.ini                # Configuration file```              ↓                                ↓

├── requirements.txt          # Dependencies

│┌────────────────────────────┐    ┌──────────────────────────────┐

├── compare_rag.py            # ⭐ RAG evaluation script

├── evaluate.py               # General evaluation**With RAG (Context-Aware)**:│   Retrieval Component      │    │   Translation Component      │

├── test_translation.py       # Quick functionality test

├── quick_test.py             # Minimal test```│     vector_db.py           │    │      Transformers            │

├── example.py                # Usage examples

│Input: "The economic growth rate increased significantly."│                            │    │                              │

├── app.py                    # Streamlit web interface

├── build_vector_db.py        # Build FAISS index│  ┌──────────────────────┐ │    │  ┌────────────────────────┐ │

├── extract_test_samples.py   # Create test datasets

│Retrieved Examples:│  │ Embedding Model      │ │    │  │ Translation Models     │ │

├── economic v1.csv           # Training corpus (UN data)

└── vector_db/                # FAISS index (created after build)1. "Economic indicators show positive trends" → "تظهر المؤشرات الاقتصادية اتجاهات إيجابية"│  │ • Sentence-BERT      │ │    │  │ • MarianMT (default)   │ │

```

2. "The growth rate exceeds expectations" → "معدل النمو يتجاوز التوقعات"│  │ • Multilingual       │ │    │  │ • NLLB                 │ │

---

3. "GDP growth accelerated" → "تسارع نمو الناتج المحلي الإجمالي"│  │ • 768D vectors       │ │    │  │ • QWEN (fine-tuned)    │ │

## 🎓 How RAG Works

│  └──────────────────────┘ │    │  └────────────────────────┘ │

### Without RAG (Baseline)

```Output: [Domain-specific, consistent translation using economic terminology]│                            │    │                              │

English Text → Mistral-7B → Arabic Translation

``````│  ┌──────────────────────┐ │    │  ┌────────────────────────┐ │



Simple, but lacks domain knowledge and consistency.│  │ Vector Database      │ │    │  │ Generation Config      │ │



### With RAG (Context-Aware)## 📁 File Structure│  │ • ChromaDB           │ │    │  │ • Beam search          │ │

```

English Text → Vector DB Search → Similar Examples│  │ • FAISS (GPU-accel.) │ │    │  │ • Temperature tuning   │ │

              ↓

    [Example 1, Example 2, Example 3]```│  │ • Cosine similarity  │ │    │  │ • Top-p sampling       │ │

              ↓

       Mistral-7B (with context) → Arabic TranslationRAG_Agent/│  └──────────────────────┘ │    │  └────────────────────────┘ │

```

├── rag_agent.py              # Main translation agent (Gemini API)└────────────────────────────┘    └──────────────────────────────┘

The model sees similar translations and adapts accordingly!

├── vector_db.py              # FAISS vector database manager              ↓

### Example

├── utils.py                  # Utilities (normalization, formatting)┌──────────────────────────────────────────────────────────────────┐

**Input:** "The economic growth rate increased significantly."

├── config.ini                # Configuration settings│                      Data Storage Layer                           │

**Retrieved Context:**

1. "Economic growth accelerated..." → "تسارع النمو الاقتصادي..."││  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │

2. "The GDP growth rate rose..." → "ارتفع معدل نمو الناتج المحلي..."

3. "Significant increase in growth..." → "زيادة كبيرة في النمو..."├── build_vector_db.py        # Build/update vector database│  │  Vector Index   │  │  Metadata Store  │  │  Economic Data │ │



**Result:** More accurate terminology and natural phrasing!├── test_gemini.py            # Quick test script│  │  (embeddings)   │  │   (text pairs)   │  │ (economic.csv) │ │



---├── compare_rag.py            # Evaluation with metrics (BLEU, chrF)│  └─────────────────┘  └──────────────────┘  └────────────────┘ │



## 📈 Performance Expectations├── extract_test_samples.py   # Extract test data└──────────────────────────────────────────────────────────────────┘



### VRAM Usage│```

- **Embeddings model:** ~0.5 GB

- **Mistral-7B (4-bit):** ~3.5-4 GB├── app.py                    # Streamlit web UI (optional)

- **Total:** ~4-4.5 GB

- **Recommended:** 6GB+ VRAM├── evaluate.py               # Full evaluation suite---



### Translation Speed├── requirements.txt          # Python dependencies

- **First translation:** 30-60 seconds (model loading)

- **With RAG:** 2-5 seconds/sentence│## 🔄 RAG Process Explained

- **Without RAG:** 1-3 seconds/sentence

├── economic v1.csv           # Translation corpus (1.8M pairs)

### Quality Improvements

- **BLEU:** +15-30% improvement└── vector_db/                # FAISS index files### Step-by-Step Translation Pipeline

- **chrF:** +10-20% improvement

- **Consistency:** Significantly better    └── translation_corpus.index

- **Domain accuracy:** Much improved

```#### 1. **Input Processing** (`utils.py`)

---

```

## ⚙️ Configuration Guide

## 🔧 ConfigurationUser Input: "The GDP growth rate increased by 3.5% in Q2 2024"

Edit `config.ini` to customize:

                    ↓

### Vector Database

```iniEdit `config.ini`:        [Text Normalization]

[VECTOR_DB]

db_type = faiss              # GPU-accelerated                    ↓

top_k = 3                    # Number of examples to retrieve

similarity_threshold = 0.5   # Minimum similarity (0-1)```iniNormalized: "The GDP growth rate increased by 3.5% in Q2 2024"

```

[GEMINI]```

### Embeddings

```inimodel_name = gemini-pro

[EMBEDDINGS]

model_name = sentence-transformers/paraphrase-multilingual-mpnet-base-v2temperature = 0.3#### 2. **Semantic Embedding** (`vector_db.py`)

device = cuda

batch_size = 16top_k_retrieval = 3```

```

max_output_tokens = 512Normalized Text → Sentence-BERT Encoder → 768D Vector

### Translation Model

```ini[0.023, -0.145, 0.892, ..., 0.334]

[TRANSLATION]

model_name = mistralai/Mistral-7B-Instruct-v0.2[VECTOR_DB]```

use_4bit = true              # CRITICAL for low VRAM

max_length = 256             # Output length limitdb_type = faiss

temperature = 0.3            # 0.1 = deterministic, 1.0 = creative

top_p = 0.9                  # Nucleus samplingcollection_name = translation_corpus#### 3. **Context Retrieval** (`vector_db.py`)

```

The vector database searches for similar translation examples:

---

[EMBEDDINGS]```

## 💡 Usage Examples

model_name = sentence-transformers/paraphrase-multilingual-mpnet-base-v2Query Vector → Vector Database (FAISS/ChromaDB)

### Basic Translation

device = cuda                    ↓

```python

from rag_agent import RAGTranslationAgent```    [Cosine Similarity Search]

from vector_db import VectorDBManager

from utils import load_config                    ↓



# Load configuration## 💻 UsageTop-5 Similar Examples (from economic v1.csv):

config = load_config("config.ini")

1. "The economic growth rate increased..." → "ارتفع معدل النمو الاقتصادي..."

# Initialize vector database

vector_db = VectorDBManager(### Python API2. "GDP estimates for the quarter..." → "تقديرات الناتج المحلي الإجمالي للربع..."

    db_type="faiss",

    embedding_model=config.get("EMBEDDINGS", "model_name"),3. "Growth statistics show..." → "تظهر إحصاءات النمو..."

    db_path="./vector_db",

    collection_name="translation_corpus",```python4. "Financial indicators improved..." → "تحسنت المؤشرات المالية..."

    device="cuda"

)from rag_agent import RAGTranslationAgent5. "The rate of increase..." → "معدل الزيادة..."



# Initialize translation agentfrom vector_db import VectorDBManager```

agent = RAGTranslationAgent(

    model_name="mistralai/Mistral-7B-Instruct-v0.2",from utils import load_config

    vector_db_manager=vector_db,

    device="cuda",#### 4. **Prompt Construction** (`rag_agent.py`)

    use_4bit=True,

    temperature=0.3,# Load config```

    top_k_retrieval=3

)config = load_config("config.ini")System: You are a professional translator specializing in economic texts.



# Translate with RAG

result = agent.translate(

    "The economic growth rate increased significantly.",# Initialize vector databaseContext (Similar Examples):

    use_context=True,

    return_context=Truevector_db = VectorDBManager(- EN: "The economic growth rate increased..."

)

    db_type="faiss",  AR: "ارتفع معدل النمو الاقتصادي..."

print(f"Translation: {result['translation']}")

print(f"Used {result['num_retrieved']} examples")    embedding_model=config.get("EMBEDDINGS", "model_name"),- EN: "GDP estimates for the quarter..."



# Show retrieved examples    db_path="./vector_db",  AR: "تقديرات الناتج المحلي الإجمالي للربع..."

for i, ex in enumerate(result['retrieved_examples'], 1):

    print(f"\nExample {i} (similarity: {ex['similarity']:.3f}):")    device="cuda"[... more examples ...]

    print(f"  EN: {ex['en']}")

    print(f"  AR: {ex['ar']}"))

```

Now translate:

### Translation Without RAG

# Initialize translation agentEN: "The GDP growth rate increased by 3.5% in Q2 2024"

```python

# Baseline translation (no context)agent = RAGTranslationAgent(AR: 

result = agent.translate(

    "The economic growth rate increased significantly.",    api_key="YOUR_GEMINI_API_KEY",  # or use env var```

    use_context=False

)    vector_db_manager=vector_db,



print(f"Translation: {result['translation']}")    model_name="gemini-pro",#### 5. **Translation Generation** (`rag_agent.py`)

```

    temperature=0.3,```

### Batch Translation

    top_k_retrieval=3Prompt → Translation Model (MarianMT/NLLB/QWEN)

```python

sentences = [)                    ↓

    "The economic growth rate increased.",

    "Inflation remains a concern.",        [Beam Search Decoding]

    "The central bank announced new policies."

]# Translate with RAG        [Context-Aware Generation]



results = agent.batch_translate(result = agent.translate(                    ↓

    sentences,

    use_context=True    "The economic growth rate increased significantly.",Arabic Translation: "ارتفع معدل نمو الناتج المحلي الإجمالي بنسبة 3.5% في الربع الثاني من عام 2024"

)

    use_context=True,```

for i, result in enumerate(results):

    print(f"{i+1}. {result['translation']}")    return_context=True

```

)#### 6. **Post-Processing** (`utils.py`)

---

```

## 📊 Understanding Metrics

print(f"Translation: {result['translation']}")Raw Output → [Normalization] → Final Translation

### BLEU Score (0-100)

- Measures n-gram overlap with reference translationprint(f"Used {result['num_retrieved']} context examples")```

- Industry standard for machine translation

- **20-30:** Understandable```

- **30-40:** Good quality

- **40-50:** Very good### Why RAG Improves Translation

- **50+:** Excellent

### Command Line

**RAG typically adds +5 to +15 points**

**Without RAG** (baseline model):

### chrF Score (0-100)

- Character-level F-score```bash```

- Better for morphologically rich languages (Arabic)

- More robust than BLEU# Quick testInput: "The GDP growth rate increased by 3.5%"

- Generally higher scores

python test_gemini.pyOutput: "زاد معدل نمو الناتج المحلي بنسبة 3.5٪"

**RAG typically adds +3 to +10 points**

        (Generic translation, inconsistent terminology)

### What Good Results Look Like

- ✅ BLEU improvement > 5 points# Full evaluation```

- ✅ chrF improvement > 3 points

- ✅ Consistent domain terminologypython compare_rag.py --api_key $GOOGLE_API_KEY --test_file test.csv

- ✅ Natural, fluent translations

**With RAG** (context-aware):

---

# Web UI```

## 🐛 Troubleshooting

streamlit run app.pyInput: "The GDP growth rate increased by 3.5%"

### Out of Memory Error

```Retrieved Context: Similar economic translations from UN corpus

```

RuntimeError: CUDA out of memoryOutput: "ارتفع معدل نمو الناتج المحلي الإجمالي بنسبة 3.5٪"

```

## 📈 Evaluation Metrics        (Domain-specific, consistent with UN terminology)

**Solutions:**

1. Ensure `use_4bit = true` in config.ini```

2. Close other GPU applications

3. Reduce `max_length` to 128The system evaluates translation quality using:

4. Try a smaller model or reduce `batch_size`

---

### Model Download Issues

- **BLEU Score**: Precision-based metric (0-100)

```

HTTPError: 401 Unauthorized- **chrF Score**: Character n-gram F-score (0-100)## 📁 File Structure and Purpose

```



For Llama-2, get a HuggingFace token:

```bash### Expected Results### Core Components

huggingface-cli login

```



Or use Mistral (no token needed):```#### 1. **`rag_agent.py`** - The Orchestrator

```ini

model_name = mistralai/Mistral-7B-Instruct-v0.2📊 RESULTS COMPARISON**Purpose**: Main agentic component that coordinates the entire RAG pipeline.

```

============================================================

### Slow First Run

**Key Responsibilities**:

The first run downloads ~13GB model - this is normal and only happens once.

🔵 WITHOUT RAG (Baseline):- Manages the translation workflow

### Poor Translation Quality

   BLEU Score:  42.35- Coordinates retrieval and generation

Try adjusting in `config.ini`:

```ini   chrF Score:  65.78- Handles different model types (Seq2Seq, Causal LM)

# More deterministic

temperature = 0.1- Implements context-aware prompting



# Retrieve more examples🟢 WITH RAG (Context-Aware):- Batch processing support

top_k = 5

   BLEU Score:  58.92

# Higher similarity threshold

similarity_threshold = 0.6   chrF Score:  78.43**Key Class**: `RAGTranslationAgent`

```

- **Methods**:

---

📈 IMPROVEMENT:  - `translate()`: Single text translation with RAG

## 🎯 Quick Command Reference

   BLEU: +16.57 points (+39.1%)  - `translate_batch()`: Batch translation processing

| Task | Command |

|------|---------|   chrF: +12.65 points (+19.2%)  - `_translate_seq2seq()`: For MarianMT, NLLB models

| Install | `pip install -r requirements.txt` |

| Build database | `python build_vector_db.py --sample 10000` |```  - `_translate_causal()`: For QWEN, LLaMA models

| Quick test | `python test_translation.py` |

| Create test data | `python extract_test_samples.py --num_samples 100` |

| **Run evaluation** | `python compare_rag.py --test_file test_data.csv --sample 50` |

| Web interface | `streamlit run app.py` |## 🎓 Key Features**Agentic Features**:

| Check GPU | `nvidia-smi` |

- Autonomous context retrieval decisions

---

### Why RAG?- Dynamic prompt construction based on retrieved examples

## 📚 Data Source

- Adaptive generation parameters

### Economic v1.csv - UN Parallel Corpus

1. **Domain Adaptation**: Uses economic/financial terminology from corpus

The system uses a parallel corpus containing:

2. **Consistency**: Learns translation style from retrieved examples#### 2. **`vector_db.py`** - The Memory System

- **Source:** English text from UN documents

- **Target:** Professional Arabic translations  3. **Context-Aware**: Better handling of idiomatic expressions**Purpose**: Manages semantic search and retrieval from the parallel corpus.

- **Domain:** Economic, financial, legal, and political terminology

- **Size:** Thousands of high-quality parallel sentence pairs4. **Dynamic**: No model retraining needed for new domains



**Structure:****Key Responsibilities**:

- `en` - English source text

- `ar` - Arabic target translation### Why Gemini API?- Build vector database from `economic v1.csv`

- `domain` - Document domain/category (optional)

- Generate embeddings for source texts

---

1. **Causal Language Model**: Excels at in-context learning- Perform semantic similarity search

## 🔬 System Requirements

2. **Few-Shot Learning**: Effectively uses retrieved examples- Handle both ChromaDB and FAISS backends

### Hardware

- **GPU:** 6GB+ VRAM (4GB might work but tight)3. **Easy Setup**: No local GPU requirements

- **RAM:** 16GB+ system RAM

- **Storage:** ~15GB for models and database4. **Free Tier**: 60 requests/min, 1,500/day**Key Class**: `VectorDBManager`



### Software- **Methods**:

- **CUDA:** 11.8+ or 12.x

- **Python:** 3.8+## 🔍 Comparison with MarianMT  - `build_from_dataframe()`: Index the economic corpus

- **OS:** Linux (tested), Windows/Mac should work

  - `retrieve()`: Semantic search for similar translations

---

| Feature | MarianMT (Previous) | Gemini + RAG (Current) |  - `add_examples()`: Add new translation pairs

## 🚦 Getting Started Checklist

|---------|---------------------|------------------------|  - `_initialize_chromadb()` / `_initialize_faiss()`: Backend setup

- [ ] Install dependencies: `pip install -r requirements.txt`

- [ ] Build vector database: `python build_vector_db.py --sample 10000`| **Architecture** | Seq2Seq Encoder-Decoder | Causal Language Model |

- [ ] Test system: `python test_translation.py`

- [ ] Extract test data: `python extract_test_samples.py --num_samples 100`| **Context Usage** | ❌ Can't use retrieved examples | ✅ Leverages RAG context |**Supported Backends**:

- [ ] Run evaluation: `python compare_rag.py --test_file test_data.csv --sample 50`

- [ ] Check results in `evaluation_results/`| **Domain Adaptation** | ❌ Fixed at training | ✅ Dynamic via retrieval |- **ChromaDB**: Easy persistence, good for <10M documents

- [ ] Try web interface: `streamlit run app.py`

| **Setup** | Complex (quantization issues) | Simple (API key) |- **FAISS**: Ultra-fast search, GPU-accelerated, scalable

---

| **Hardware** | Requires GPU | API-based (no GPU needed) |

## 🎓 Next Steps

| **Translation Quality** | Good | Better with context |#### 3. **`utils.py`** - Helper Functions

1. **Run comparison** to see RAG improvements

2. **Tune parameters** in config.ini (temperature, top_k)**Purpose**: Common utilities for text processing and configuration.

3. **Add domain data** to vector database

4. **Fine-tune model** on your specific domain## 🐛 Troubleshooting

5. **Deploy** to production

**Key Functions**:

---

### API Key Error- `normalize_english_text()`: Clean and normalize English input

## 📄 License

- `normalize_arabic_text()`: Clean and normalize Arabic output

This project is part of the Context-Specific Machine Translation research.

```bash- `format_context_examples()`: Format retrieved examples for prompts

---

# Check if set- `chunk_text()`: Handle long texts

## 🤝 Contributing

echo $GOOGLE_API_KEY- `load_config()`: Configuration management

Contributions, issues, and feature requests are welcome!

- `setup_logging()`: Logging infrastructure

---

# Set it

**Built with ❤️ for accurate, context-aware Arabic translation**

export GOOGLE_API_KEY="your_key_here"### Scripts and Tools

```

#### 4. **`build_vector_db.py`** - Database Builder

### Rate Limit Error**Purpose**: Process `economic v1.csv` and build the searchable vector database.



```python**Usage**:

# Add delay between requests```bash

import time# Build from economic corpus

time.sleep(1)  # 1 second = 60 req/minpython build_vector_db.py --config config.ini

```

# Test with sample

### Import Errorpython build_vector_db.py --sample 1000

```

```bash

# Reinstall package**What It Does**:

pip install --upgrade google-generativeai1. Reads `economic v1.csv` (EN-AR parallel sentences)

```2. Generates embeddings for English sentences

3. Creates vector index (ChromaDB or FAISS)

### Poor Translation Quality4. Stores metadata (EN text, AR translation, domains)

5. Saves to disk for retrieval

1. Increase `top_k_retrieval` (try 5-7)

2. Lower `temperature` (try 0.1-0.2)**Output**: Vector database in `./vector_db/` directory

3. Rebuild vector DB with more samples

#### 5. **`evaluate.py`** - Evaluation Suite

## 📊 Dataset**Purpose**: Measure translation quality with and without RAG.



**economic v1.csv**: 1,799,855 parallel EN-AR sentence pairs**Metrics**:

- Source: UN documents and economic reports- **BLEU**: N-gram overlap score (0-100)

- Domain: Economics, finance, international development- **chrF**: Character-level F-score (0-100)

- Format: CSV with columns: `id`, `en`, `ar`

**Usage**:

## 🚀 API Usage```bash

# Evaluate on test set

**Gemini API Free Tier:**python evaluate.py --config config.ini --test_file test_data.csv

- 60 requests per minute

- 1,500 requests per day# Compare RAG vs baseline

- Free foreverpython evaluate.py --compare_baseline

```

**For 100 test samples:**

- Baseline: 100 requests**Output**:

- RAG: 100 requests- Aggregate metrics (BLEU, chrF)

- **Total: 200 requests** (~3.5 minutes)- Per-example results

- RAG improvement statistics

## 📝 License- Sample translations



This project uses:#### 6. **`app.py`** - Web Interface

- Google Gemini API (subject to Google's terms)**Purpose**: Interactive Streamlit demo for the translation system.

- Open-source dependencies (see requirements.txt)

**Features**:

## 🤝 Contributing- Single text translation

- Batch translation (CSV upload)

To improve the system:- Context visualization (shows retrieved examples)

1. Tune hyperparameters in `config.ini`- Export results to CSV/JSON

2. Try different Gemini models (gemini-1.5-flash, etc.)- Real-time translation

3. Add domain-specific corpora to vector DB- Configuration options

4. Experiment with prompt templates

**Usage**:

## 📚 References```bash

streamlit run app.py

- [Gemini API Documentation](https://ai.google.dev/docs)```

- [FAISS Documentation](https://github.com/facebookresearch/faiss)

- [Sentence-BERT](https://www.sbert.net/)**Interface Sections**:

- [RAG Paper](https://arxiv.org/abs/2005.11401)- Text input area

- Translation button

---- Results display (Arabic output)

- Retrieved context panel

✅ **Ready to translate with context-aware RAG!**- Statistics and metrics



For questions or issues, check the logs in `./logs/` or review the configuration in `config.ini`.#### 7. **`example.py`** - Usage Examples

**Purpose**: Simple demonstrations and quick testing.

**Examples Included**:
- Basic translation
- Batch processing
- Custom configuration
- Error handling
- Performance testing

**Usage**:
```bash
python example.py
```

### Configuration

#### 8. **`config.ini`** - System Configuration
**Purpose**: Centralized configuration for all components.

**Key Sections**:

```ini
[GENERAL]
vector_db_dir = ./vector_db        # Where to store/load vector DB
logs_dir = ./logs                  # Logging directory
corpus_file = ./economic v1.csv    # Path to parallel corpus

[EMBEDDINGS]
model_name = sentence-transformers/paraphrase-multilingual-mpnet-base-v2
device = cuda                      # cuda, cpu, or mps
batch_size = 32                    # Embedding batch size

[VECTOR_DB]
db_type = faiss                    # chromadb or faiss
collection_name = translation_corpus
top_k = 5                          # Number of examples to retrieve

[TRANSLATION]
model_name = Helsinki-NLP/opus-mt-en-ar
device = cuda
max_length = 512
num_beams = 5                      # Beam search width
temperature = 0.7                  # Sampling temperature
top_p = 0.9                        # Nucleus sampling
use_4bit = false                   # 4-bit quantization
use_8bit = false                   # 8-bit quantization
```

#### 9. **`requirements.txt`** - Dependencies
**Purpose**: All Python package dependencies.

**Key Packages**:
```
torch>=2.0.0                       # PyTorch
transformers>=4.30.0               # HuggingFace models
sentence-transformers>=2.2.0       # Embedding models
chromadb>=0.4.0                    # ChromaDB backend
faiss-gpu>=1.7.2                   # FAISS GPU backend
streamlit>=1.25.0                  # Web UI
pandas>=1.5.0                      # Data processing
numpy>=1.24.0                      # Numerical operations
sacrebleu>=2.3.0                   # BLEU metric
```

### Setup Scripts

#### 10. **`setup.fish`** / **`setup.sh`** - Automated Setup
**Purpose**: One-command environment setup for Fish/Bash shells.

**What They Do**:
```bash
./setup.fish  # or ./setup.sh
```

1. Create Python virtual environment
2. Install all dependencies from `requirements.txt`
3. Verify GPU/CUDA setup
4. Create necessary directories
5. Check configuration

### Utility Scripts

#### 11. **`check_gpu.py`** - GPU Diagnostics
**Purpose**: Verify GPU/CUDA setup and provide recommendations.

**Usage**:
```bash
python check_gpu.py
```

**Checks**:
- PyTorch CUDA availability
- GPU device information
- VRAM availability
- FAISS GPU support
- Optimal configuration suggestions

---

## 🚀 Quick Start Guide

### Installation (2 minutes)

```bash
cd RAG_Agent

# Run automated setup
./setup.fish  # or ./setup.sh for bash

# Activate environment
source venv/bin/activate.fish
```

### Build Vector Database (5-10 minutes)

```bash
# Build from economic v1.csv
python build_vector_db.py --config config.ini

# Or test with sample first
python build_vector_db.py --sample 1000
```

This processes the `economic v1.csv` corpus and creates the vector index.

### Test the System (1 minute)

```bash
# Run example script
python example.py
```

### Launch Web Interface (30 seconds)

```bash
# Start Streamlit app
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 💻 Usage Examples

### Python API - Basic Translation

```python
from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager

# Initialize vector database
vector_db = VectorDBManager(
    db_type="faiss",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    db_path="./vector_db"
)

# Initialize RAG agent
agent = RAGTranslationAgent(
    translation_model="Helsinki-NLP/opus-mt-en-ar",
    vector_db_manager=vector_db,
    top_k_retrieval=5
)

# Translate
result = agent.translate(
    "The economic growth rate increased by 3.5%",
    return_context=True
)

print(f"Translation: {result['translation']}")
print(f"Retrieved {len(result['context'])} examples")
```

### Batch Translation

```python
import pandas as pd

# Load test data
df = pd.read_csv("test_data.csv")

# Translate all English texts
results = agent.translate_batch(df['en'].tolist())

# Add translations to dataframe
df['ar_predicted'] = [r['translation'] for r in results]

# Save results
df.to_csv("translations_output.csv", index=False)
```

### Compare With/Without RAG

```python
# Translate with RAG (context-aware)
result_with_rag = agent.translate(
    "The GDP increased significantly",
    use_context=True
)

# Translate without RAG (baseline)
result_without_rag = agent.translate(
    "The GDP increased significantly",
    use_context=False
)

print("With RAG:", result_with_rag['translation'])
print("Without RAG:", result_without_rag['translation'])
```

---

## 📊 Evaluation

### Running Evaluation

```bash
# Evaluate on test set
python evaluate.py --test_file test_data.csv --config config.ini

# Compare RAG vs baseline
python evaluate.py --compare_baseline --output_file results.json
```

### Expected Results

**With RAG** (Context-Aware):
- BLEU: 45-55
- chrF: 65-75
- Better domain terminology
- More consistent translations

**Without RAG** (Baseline):
- BLEU: 35-45
- chrF: 55-65
- Generic translations
- Less consistent

---

## ⚙️ GPU/CUDA Support

### Check GPU Setup

```bash
python check_gpu.py
```

### GPU Configuration

For GPU acceleration, ensure `config.ini` has:

```ini
[EMBEDDINGS]
device = cuda

[TRANSLATION]
device = cuda
use_8bit = false  # Set true if <16GB VRAM
use_4bit = false  # Set true if <12GB VRAM
```

### FAISS GPU Acceleration

FAISS automatically uses GPU if available. Benefits:
- 10-20x faster retrieval
- Handles larger vector databases
- Reduced latency

---

## 🔧 Customization

### Use Different Translation Models

```python
# Option 1: MarianMT (default, fast)
agent = RAGTranslationAgent(
    translation_model="Helsinki-NLP/opus-mt-en-ar",
    vector_db_manager=vector_db
)

# Option 2: NLLB (high quality, multilingual)
agent = RAGTranslationAgent(
    translation_model="facebook/nllb-200-distilled-600M",
    vector_db_manager=vector_db
)

# Option 3: Fine-tuned QWEN (best for domain)
agent = RAGTranslationAgent(
    translation_model="path/to/finetuned-qwen",
    vector_db_manager=vector_db
)
```

### Adjust Retrieval Parameters

```python
# More context examples
agent = RAGTranslationAgent(
    translation_model="Helsinki-NLP/opus-mt-en-ar",
    vector_db_manager=vector_db,
    top_k_retrieval=10  # Retrieve 10 examples instead of 5
)

# Domain-specific retrieval
result = agent.translate(
    "The GDP increased",
    domain="economic",  # Filter by domain
    top_k=7
)
```

### Custom Prompt Templates

```python
custom_prompt = """You are an expert economic translator.

Examples from UN corpus:
{context}

Translate this economic text to Arabic:
{source_text}

Translation:"""

agent = RAGTranslationAgent(
    translation_model="Helsinki-NLP/opus-mt-en-ar",
    vector_db_manager=vector_db,
    prompt_template=custom_prompt
)
```

---

## 🐛 Troubleshooting

### Issue: "Vector database not found"

**Solution**: Build the database first:
```bash
python build_vector_db.py --config config.ini
```

### Issue: "CUDA out of memory"

**Solution**: Enable quantization in `config.ini`:
```ini
[TRANSLATION]
use_8bit = true
```

Or use CPU:
```ini
[TRANSLATION]
device = cpu
```

### Issue: "ChromaDB collection empty"

**Solution**: Rebuild the vector database:
```bash
python build_vector_db.py --clear_existing --config config.ini
```

### Issue: "Poor translation quality"

**Solutions**:
1. Increase `top_k` to retrieve more examples
2. Ensure `economic v1.csv` is properly loaded
3. Try different translation models
4. Check if RAG is enabled (`use_context=True`)

---

## 📈 Performance Tips

### For Speed
- Use FAISS instead of ChromaDB
- Enable GPU acceleration
- Use 8-bit quantization
- Reduce `top_k` to 3-5

### For Quality
- Increase `top_k` to 7-10
- Use larger translation models (NLLB)
- Ensure diverse training corpus
- Fine-tune on domain-specific data

### For Memory Efficiency
- Enable 4-bit quantization
- Use smaller embedding models
- Reduce batch sizes
- Use CPU if limited VRAM

---

## 🔬 How the Agentic System Works

### Agent Decision Flow

```
┌─────────────────────────────────────────────────┐
│         Input: English text to translate        │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ AGENT DECISION 1: Text Analysis                 │
│ • Analyze semantic content                      │
│ • Identify domain (economic, legal, etc.)       │
│ • Determine complexity                          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ AGENT DECISION 2: Retrieval Strategy            │
│ • How many examples to retrieve? (top_k)        │
│ • Should we filter by domain?                   │
│ • What similarity threshold?                    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ ACTION: Query Vector Database                   │
│ • Search economic v1.csv embeddings             │
│ • Rank by semantic similarity                   │
│ • Return top-k most relevant examples           │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ AGENT DECISION 3: Context Selection              │
│ • Evaluate retrieved examples                   │
│ • Filter out low-quality matches                │
│ • Order by relevance                            │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ AGENT DECISION 4: Prompt Construction            │
│ • Format selected examples                      │
│ • Build context-aware prompt                    │
│ • Include domain-specific instructions          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ AGENT DECISION 5: Generation Strategy            │
│ • Select temperature for creativity             │
│ • Choose beam width for quality                 │
│ • Determine max length                          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ ACTION: Generate Translation                     │
│ • Input: Context + Source text                  │
│ • Process: Neural translation with context      │
│ • Output: Arabic translation                    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ AGENT DECISION 6: Post-Processing                │
│ • Validate output quality                       │
│ • Normalize Arabic text                         │
│ • Check for errors                              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Output: Context-aware translation        │
└─────────────────────────────────────────────────┘
```

### Key Agentic Behaviors

1. **Autonomous Reasoning**: Makes decisions without human intervention at each step
2. **Context Awareness**: Understands when to use which examples based on semantic similarity
3. **Adaptive Strategy**: Adjusts parameters based on input characteristics
4. **Goal-Oriented**: Works towards producing the best domain-specific translation
5. **Self-Monitoring**: Validates outputs and can retry with different parameters

---

## 📚 Key Concepts

### RAG (Retrieval-Augmented Generation)
A technique that enhances language models by retrieving relevant information from external knowledge sources before generation. In this case, we retrieve similar translation examples from `economic v1.csv`.

### Vector Database
A specialized database that stores high-dimensional vectors (embeddings) and enables fast similarity search. Our economic corpus is embedded and indexed here.

### Semantic Search
Finding similar texts based on meaning rather than exact word matches. Powered by sentence embeddings.

### Few-Shot Learning
Providing the model with examples (from retrieved context) to guide translation, improving domain-specific accuracy.

---

## 🎓 Citation

If you use this system in research, please cite:

```bibtex
@software{rag_translation_agent,
  title={RAG Translation Agent: Context-Specific Machine Translation to Arabic},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/your-repo}
}
```

---

## 📝 License

This project is licensed under the MIT License. The UN parallel corpus (`economic v1.csv`) may have separate licensing terms.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional translation models
- More evaluation metrics
- Domain expansion beyond economics
- Multi-language support
- Fine-tuning scripts

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review example scripts

---

## 📂 Complete File Structure

```
RAG_Agent/
├── 📄 README.md              ← You are here! Complete documentation
│
├── 🔧 Core Components
│   ├── rag_agent.py          ← RAG orchestrator (agentic logic)
│   ├── vector_db.py          ← Vector database manager
│   └── utils.py              ← Helper functions
│
├── 🛠️ Scripts & Tools
│   ├── build_vector_db.py    ← Build vector DB from economic v1.csv
│   ├── evaluate.py           ← Evaluation suite
│   ├── app.py                ← Streamlit web interface
│   ├── example.py            ← Usage examples
│   └── check_gpu.py          ← GPU diagnostics
│
├── 📊 Data
│   └── economic v1.csv       ← UN parallel corpus (EN-AR)
│
├── ⚙️ Configuration
│   ├── config.ini            ← System configuration
│   ├── requirements.txt      ← Python dependencies
│   ├── setup.fish            ← Setup script (Fish shell)
│   └── setup.sh              ← Setup script (Bash)
│
└── 🗂️ Generated (after setup)
    ├── vector_db/            ← Vector database storage
    ├── logs/                 ← Application logs
    └── venv/                 ← Python virtual environment
```

---

## 🎉 Summary

You now have a complete RAG-based translation system that:

✅ Uses `economic v1.csv` as knowledge base  
✅ Performs semantic retrieval for context  
✅ Acts as an intelligent agent with autonomous decision-making  
✅ Delivers domain-specific, consistent translations  
✅ Supports multiple models and configurations  
✅ Provides web UI and Python API  
✅ Includes comprehensive evaluation tools  
✅ Offers GPU acceleration for performance  

**Ready to translate!** 🚀
