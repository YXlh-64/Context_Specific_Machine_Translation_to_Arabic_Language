# Comprehensive Project Guide - NLP Translation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Folder Structure Explained](#folder-structure-explained)
4. [System Components Deep Dive](#system-components-deep-dive)
5. [Prerequisites & Dependencies](#prerequisites--dependencies)
6. [Complete Setup Guide](#complete-setup-guide)
7. [Running the Project](#running-the-project)
8. [Testing the Services](#testing-the-services)
9. [API Usage Examples](#api-usage-examples)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This is a **production-grade translation assistance pipeline** designed to help translators produce high-quality, domain-specific translations from English to Arabic (with support for French-Arabic). The system consists of three interconnected microservices that work together to provide intelligent translation assistance.

### What This System Does

1. **Glossary Lookup (Phase 1)**: Finds domain-specific terminology matches in your text
2. **RAG Retrieval (Phase 2)**: Finds semantically similar translation examples from a knowledge base
3. **Prompt Construction (Phase 3)**: Combines glossary terms and examples into optimized prompts for LLMs

### Key Features

- 🔍 **Intelligent Glossary Lookup**: Full-text search with n-gram matching (1-4 grams)
- 🧠 **Semantic RAG Retrieval**: LaBSE embeddings with Qdrant vector database
- 📝 **Smart Prompt Construction**: XML/JSON formatted prompts optimized for LLMs
- ⚡ **Production Optimizations**: Connection pooling, multi-layer caching, async operations
- 🛡️ **Robust Error Handling**: Circuit breakers, graceful degradation, comprehensive logging
- 📊 **Metrics & Monitoring**: Real-time performance statistics and health checks

### Supported Domains

- `technology` - Software, AI, computing
- `healthcare` / `health` - Medical, pharmaceutical
- `economic` / `finance` - Finance, business, trade
- `legal` - Law, contracts, regulations
- `education` - Academic, research
- `culture` / `media` - Arts, media, society
- `agriculture` - Agricultural terms
- `history` - Historical content

---

## 🏗️ Architecture & Data Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATION                        │
│         (Your frontend, translation tool, etc.)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSLATION PIPELINE (3 Services)                │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Glossary   │───▶│    RAG       │───▶│   Prompt     │  │
│  │   System     │    │   System     │    │ Construction │  │
│  │  Port 8001   │    │  Port 8002   │    │  Port 8003   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   SQLite     │    │   Qdrant     │    │   Jinja2     │  │
│  │   FTS5 DB    │    │   Vector DB  │    │  Templates   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Redis Cache  │
                    │ (Optional)    │
                    └───────────────┘
```

### Data Flow Process

1. **Input**: Translator provides source text and domain
2. **Glossary Lookup (Phase 1)**: 
   - Extracts n-grams (1-4 word phrases) from text
   - Searches SQLite FTS5 database for domain-specific terminology matches
   - Returns matched terms with their Arabic translations
3. **RAG Retrieval (Phase 2)**:
   - Converts query to embedding using LaBSE model
   - Searches Qdrant vector database for semantically similar translation pairs
   - Returns top-k most relevant examples
4. **Prompt Construction (Phase 3)**:
   - Combines glossary matches + RAG examples
   - Formats into XML/JSON prompt using Jinja2 templates
   - Adds domain-specific instructions and context
5. **Output**: Formatted prompt ready for LLM translation

---

## 📁 Folder Structure Explained

```
NLP-PROJECT/
│
├── DATA/                          # Development/test data
│   └── DEV-EXAMPLES/              # Example CSV files for testing
│       ├── healthcare.csv
│       └── pahse3-prompt-construction-example.csv
│
├── glossary-system/               # PHASE 1: Glossary Lookup Service
│   ├── app/                       # Main application code
│   │   ├── main.py               # FastAPI app entry point (Port 8001)
│   │   ├── api/                  # API routes/endpoints
│   │   │   └── routes.py        # All API endpoint definitions
│   │   ├── core/                 # Core functionality
│   │   │   ├── config.py        # Configuration settings
│   │   │   ├── database.py      # Database schema & operations
│   │   │   ├── connection_pool.py  # SQLite connection pooling
│   │   │   ├── lru_cache.py     # In-memory LRU cache
│   │   │   └── redis.py         # Redis cache client
│   │   ├── models/               # Data models
│   │   │   └── schemas.py       # Pydantic request/response models
│   │   ├── services/             # Business logic
│   │   │   ├── glossary_service.py      # Main glossary lookup logic
│   │   │   ├── optimized_glossary_service.py  # Optimized version
│   │   │   ├── pdf_service.py   # PDF processing
│   │   │   └── cache_service.py # Cache management
│   │   └── utils/                # Utility functions
│   │       ├── text_processor.py # Text processing (n-grams, etc.)
│   │       └── file_utils.py     # File handling utilities
│   │
│   ├── data/                      # Data storage
│   │   ├── glossary.db           # SQLite database (FTS5 enabled)
│   │   ├── RAW DATA/             # Original CSV files
│   │   │   ├── en-ar/           # English-Arabic data
│   │   │   └── fr-ar/           # French-Arabic data
│   │   ├── PROCESSED_DATA/       # Cleaned/processed CSV files
│   │   └── verified_glossary_terms.csv  # Verified terms
│   │
│   ├── scripts/                   # Setup & utility scripts
│   │   ├── init_db.py            # Initialize database schema
│   │   ├── seed_glossary.py      # Populate database from CSV
│   │   ├── correct_csv.py        # CSV correction utilities
│   │   └── generate_test_pdf.py  # Generate test PDFs
│   │
│   ├── tests/                     # Test files
│   │   ├── test_api.py           # API endpoint tests
│   │   └── test_comprehensive.py # Comprehensive integration tests
│   │
│   ├── uploads/                   # PDF upload directory
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Service-specific documentation
│
├── RAG-SYSTEM/                    # PHASE 2: RAG Retrieval Service
│   ├── app/                       # Main application code
│   │   ├── main.py               # FastAPI app entry point (Port 8002)
│   │   ├── api/
│   │   │   └── routes.py        # API endpoints for search
│   │   ├── core/
│   │   │   └── config.py        # Configuration (Qdrant, Redis, etc.)
│   │   ├── models/
│   │   │   └── schemas.py       # Request/response models
│   │   ├── services/             # Core services
│   │   │   ├── embedding_service.py      # LaBSE embedding generation
│   │   │   ├── optimized_embedding_service.py  # Optimized version
│   │   │   ├── retrieval_service.py     # Vector search logic
│   │   │   ├── optimized_retrieval_service.py  # Optimized version
│   │   │   ├── setup_qdrant.py  # Qdrant client & collection setup
│   │   │   ├── data_loader.py   # Load translation data
│   │   │   ├── upload_service.py # Upload data to Qdrant
│   │   │   ├── pipeline.py      # End-to-end pipeline
│   │   │   ├── integration.py   # Integration with Phase 1
│   │   │   └── caching.py       # Redis caching
│   │   └── utils/
│   │       ├── error_handling.py # Error handling utilities
│   │       ├── evaluation.py     # Evaluation metrics
│   │       └── monitoring.py     # Performance monitoring
│   │
│   ├── scripts/                   # Setup & testing scripts
│   │   ├── setup.py              # Setup Qdrant collection
│   │   ├── add_test_sentences.py # Add test data
│   │   ├── test_retrieval.py     # Test retrieval
│   │   └── test_hybrid.py        # Test hybrid search
│   │
│   ├── tests/                     # Test files
│   │   ├── test_api.py           # API tests
│   │   ├── test_retrieval.py     # Retrieval logic tests
│   │   ├── test_comprehensive.py # Comprehensive tests
│   │   └── test_full_integration.py  # Full integration tests
│   │
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Service-specific documentation
│
├── prompt-construction/           # PHASE 3: Prompt Construction Service
│   ├── app/                       # Main application code
│   │   ├── main.py               # FastAPI app entry point (Port 8003)
│   │   ├── api/
│   │   │   └── routes.py        # API endpoints
│   │   ├── core/
│   │   │   └── config.py        # Configuration
│   │   ├── models/
│   │   │   └── schemas.py       # Request/response models
│   │   ├── services/
│   │   │   ├── prompt_service.py        # Prompt construction logic
│   │   │   └── optimized_prompt_service.py  # Optimized version
│   │   └── utils/
│   │       └── token_counter.py  # Token counting for LLMs
│   │
│   ├── scripts/
│   │   ├── test_pipeline.py      # Test the full pipeline
│   │   └── start_server.ps1      # Windows startup script
│   │
│   ├── tests/
│   │   ├── test_api.py           # API tests
│   │   ├── test_prompt_construction.py  # Prompt construction tests
│   │   └── test_comprehensive.py # Comprehensive tests
│   │
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Service-specific documentation
│
├── README.md                       # Main project documentation
└── .gitignore                      # Git ignore rules
```

---

## 🔧 System Components Deep Dive

### 1. Glossary System (Phase 1) - Port 8001

**Purpose**: Domain-specific terminology lookup using full-text search.

#### Key Technologies
- **SQLite with FTS5**: Full-text search engine for fast n-gram matching
- **Connection Pooling**: Manages database connections efficiently
- **Multi-layer Caching**: LRU cache (L1) + Redis (L2) for performance
- **PDF Processing**: Extracts text from PDF documents for batch processing

#### How It Works
1. Receives text and domain
2. Generates n-grams (1-4 word phrases) from input text
3. Searches FTS5 database for matching glossary terms
4. Returns matched terms with Arabic translations and metadata

#### Main Files
- `app/services/glossary_service.py`: Core lookup logic
- `app/core/database.py`: Database schema and FTS5 setup
- `app/utils/text_processor.py`: N-gram generation and text processing

#### Database Schema
```sql
-- Glossary terms table with FTS5 virtual table
CREATE VIRTUAL TABLE glossary_fts USING fts5(
    source_term, 
    target_term, 
    domain, 
    frequency,
    content='glossary_terms',
    content_rowid='id'
);
```

---

### 2. RAG System (Phase 2) - Port 8002

**Purpose**: Semantic similarity search for relevant translation examples.

#### Key Technologies
- **LaBSE Embeddings**: Multilingual sentence embeddings (768 dimensions)
- **Qdrant Vector Database**: Fast similarity search with HNSW indexing
- **MMR (Maximal Marginal Relevance)**: Diversity re-ranking
- **Hybrid Search**: Combines semantic + keyword search

#### How It Works
1. Receives query text and domain
2. Converts query to embedding using LaBSE model
3. Searches Qdrant vector database for similar translation pairs
4. Applies MMR for diversity (optional)
5. Returns top-k most relevant examples with similarity scores

#### Main Files
- `app/services/embedding_service.py`: LaBSE embedding generation
- `app/services/retrieval_service.py`: Vector search logic
- `app/services/setup_qdrant.py`: Qdrant client and collection management

#### Vector Database Structure
- **Collection**: `translation_memory` (default)
- **Vectors**: 768-dimensional LaBSE embeddings
- **Payload**: Source text, target text, domain, language pair
- **Index**: HNSW (Hierarchical Navigable Small World) for fast search

---

### 3. Prompt Construction (Phase 3) - Port 8003

**Purpose**: Build optimized LLM prompts from glossary matches and examples.

#### Key Technologies
- **Jinja2 Templates**: Template engine for prompt formatting
- **Tiktoken**: Token counting for LLM context management
- **Input Sanitization**: XSS and injection prevention

#### How It Works
1. Receives glossary matches + RAG examples + source text
2. Selects best examples (top-k, diversity filtering)
3. Formats into XML/JSON using Jinja2 templates
4. Adds domain-specific instructions
5. Estimates token count
6. Returns formatted prompt

#### Prompt Format Example (XML)
```xml
<translation_prompt>
  <source_sentence>Machine learning models require training data</source_sentence>
  <domain>technology</domain>
  <glossary>
    <term source="machine learning" target="التعلم الآلي" />
  </glossary>
  <examples>
    <example score="0.85">
      <source>Training data is essential for ML</source>
      <target>بيانات التدريب ضرورية للتعلم الآلي</target>
    </example>
  </examples>
</translation_prompt>
```

---

## 📦 Prerequisites & Dependencies

### System Requirements

1. **Python 3.10+** (required)
2. **Redis** (optional, for distributed caching)
3. **Qdrant** (required for RAG system)

### External Services

#### Redis (Optional but Recommended)
- **Purpose**: Distributed caching for improved performance
- **Default Port**: 6379
- **Installation**:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install redis-server
  
  # macOS
  brew install redis
  
  # Or use Docker
  docker run -d -p 6379:6379 redis:7-alpine
  ```

#### Qdrant (Required for RAG System)
- **Purpose**: Vector database for semantic search
- **Default Port**: 6333
- **Installation**:
  ```bash
  # Using Docker (Recommended)
  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
  
  # Or download binary from https://qdrant.tech/documentation/guides/installation/
  ```

### Python Dependencies

Each service has its own `requirements.txt`. Key dependencies:

**Glossary System:**
- FastAPI, Uvicorn (web framework)
- SQLAlchemy (database ORM)
- spaCy, NLTK (NLP processing)
- pdfplumber (PDF extraction)
- Redis (caching)

**RAG System:**
- FastAPI, Uvicorn
- sentence-transformers (LaBSE embeddings)
- torch (PyTorch for ML models)
- qdrant-client (vector database client)
- Redis (caching)

**Prompt Construction:**
- FastAPI, Uvicorn
- Jinja2 (templating)
- tiktoken (token counting)
- requests/httpx (HTTP client)

---

## 🚀 Complete Setup Guide

### Step 1: Clone and Navigate to Project

```bash
cd /home/aya/Desktop/ENSIA 4Y/S1/NLP/Project/NLP-PROJECT
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
# Install dependencies for all three services
pip install -r glossary-system/requirements.txt
pip install -r RAG-SYSTEM/requirements.txt
pip install -r prompt-construction/requirements.txt
```

**Note**: If you encounter issues, install them one by one:
```bash
cd glossary-system && pip install -r requirements.txt
cd ../RAG-SYSTEM && pip install -r requirements.txt
cd ../prompt-construction && pip install -r requirements.txt
cd ..
```

### Step 4: Download spaCy Models (for Glossary System)

```bash
# Download English model
python -m spacy download en_core_web_sm

# Download Arabic model (if available)
# python -m spacy download ar_core_news_sm
# If not available, the system will use a simple tokenizer
```

### Step 5: Start External Services

#### Start Redis (Optional)
```bash
# If installed via package manager
redis-server

# Or using Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

#### Start Qdrant (Required for RAG System)
```bash
# Using Docker (Recommended)
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest

# Verify Qdrant is running
curl http://localhost:6333/health
```

### Step 6: Initialize Glossary Database

```bash
cd glossary-system

# Initialize database schema
python scripts/init_db.py

# Seed database with glossary terms from CSV files
python scripts/seed_glossary.py

cd ..
```

**Note**: The database file will be created at `glossary-system/data/glossary.db`

### Step 7: Setup Qdrant Collection (for RAG System)

```bash
cd RAG-SYSTEM

# Setup Qdrant collection and upload translation data
python scripts/setup.py

# Or manually add test sentences
python scripts/add_test_sentences.py

cd ..
```

**Note**: You need translation data (CSV files with source-target pairs) to populate Qdrant. The script will create the collection and upload embeddings.

### Step 8: Configure Environment Variables (Optional)

Create `.env` files in each service directory if you need custom configuration:

**glossary-system/.env:**
```env
DATABASE_URL=file:data/glossary.db
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

**RAG-SYSTEM/.env:**
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=translation_memory
REDIS_HOST=localhost
REDIS_PORT=6379
MODEL_NAME=sentence-transformers/LaBSE
```

**prompt-construction/.env:**
```env
PORT=8003
DEFAULT_PROMPT_FORMAT=xml
DEFAULT_DOMAIN=general
```

---

## ▶️ Running the Project

### Method 1: Run All Services in Separate Terminals

#### Terminal 1 - Glossary System (Port 8001)
```bash
cd glossary-system
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

#### Terminal 2 - RAG System (Port 8002)
```bash
cd RAG-SYSTEM
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

#### Terminal 3 - Prompt Construction (Port 8003)
```bash
cd prompt-construction
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

### Method 2: Run Services in Background (Linux/Mac)

```bash
# Start Glossary System
cd glossary-system
nohup uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload > glossary.log 2>&1 &

# Start RAG System
cd ../RAG-SYSTEM
nohup uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload > rag.log 2>&1 &

# Start Prompt Construction
cd ../prompt-construction
nohup uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload > prompt.log 2>&1 &
```

### Verify Services Are Running

```bash
# Check Glossary System
curl http://localhost:8001/health

# Check RAG System
curl http://localhost:8002/health

# Check Prompt Construction
curl http://localhost:8003/health
```

### Access API Documentation

Once services are running, access interactive API docs:

- **Glossary System**: http://localhost:8001/docs
- **RAG System**: http://localhost:8002/docs
- **Prompt Construction**: http://localhost:8003/docs

---

## 🧪 Testing the Services

### Run Unit Tests

```bash
# Glossary System tests
cd glossary-system
pytest tests/ -v

# RAG System tests
cd ../RAG-SYSTEM
pytest tests/ -v

# Prompt Construction tests
cd ../prompt-construction
pytest tests/ -v
```

### Manual API Testing

#### Test Glossary System

```bash
curl -X POST http://localhost:8001/api/v1/translate/sentence \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of artificial intelligence",
    "source_lang": "en",
    "target_lang": "ar",
    "domain": "technology"
  }'
```

#### Test RAG System

```bash
curl -X POST http://localhost:8002/api/v1/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep neural networks for image classification",
    "domain": "technology",
    "top_k": 5
  }'
```

#### Test Prompt Construction

```bash
curl -X POST http://localhost:8003/api/v1/prompt/construct \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "Machine learning models require training data",
    "glossary_matches": [
      {"source_term": "machine learning", "target_term": "التعلم الآلي", "n_gram_size": 2}
    ],
    "similar_examples": [
      {
        "source_text": "Training data is essential for ML",
        "target_text": "بيانات التدريب ضرورية للتعلم الآلي",
        "score": 0.85
      }
    ],
    "domain": "technology",
    "source_lang": "en",
    "target_lang": "ar",
    "format": "xml"
  }'
```

---

## 📡 API Usage Examples

### Complete Pipeline Example

Here's how to use all three services together:

```python
import requests

# Step 1: Glossary Lookup
glossary_response = requests.post(
    "http://localhost:8001/api/v1/translate/sentence",
    json={
        "text": "Machine learning models require training data",
        "source_lang": "en",
        "target_lang": "ar",
        "domain": "technology"
    }
)
glossary_data = glossary_response.json()
glossary_matches = glossary_data["glossary_matches"]

# Step 2: RAG Retrieval
rag_response = requests.post(
    "http://localhost:8002/api/v1/search/semantic",
    json={
        "query": "Machine learning models require training data",
        "domain": "technology",
        "top_k": 5
    }
)
rag_data = rag_response.json()
similar_examples = rag_data["results"]

# Step 3: Prompt Construction
prompt_response = requests.post(
    "http://localhost:8003/api/v1/prompt/construct",
    json={
        "sentence": "Machine learning models require training data",
        "glossary_matches": glossary_matches,
        "similar_examples": similar_examples,
        "domain": "technology",
        "source_lang": "en",
        "target_lang": "ar",
        "format": "xml"
    }
)
prompt_data = prompt_response.json()
final_prompt = prompt_data["prompt"]

print("Final Prompt:")
print(final_prompt)
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Failed (Glossary System)

**Error**: `unable to open database file`

**Solution**:
```bash
# Ensure database exists
cd glossary-system
python scripts/init_db.py
python scripts/seed_glossary.py

# Check file permissions
ls -la data/glossary.db
chmod 644 data/glossary.db
```

#### 2. Redis Connection Refused

**Error**: `Connection refused to redis://localhost:6379`

**Solution**:
```bash
# Start Redis
redis-server

# Or check if Redis is running
redis-cli ping
# Should return: PONG

# If using Docker
docker ps | grep redis
```

**Note**: Redis is optional. The system will work without it, but caching will be disabled.

#### 3. Qdrant Not Available (RAG System)

**Error**: `Failed to connect to Qdrant at localhost:6333`

**Solution**:
```bash
# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Verify it's running
curl http://localhost:6333/health

# Check if collection exists
curl http://localhost:6333/collections
```

#### 4. Model Download Issues (RAG System)

**Error**: `OSError: Can't load tokenizer`

**Solution**:
```bash
# LaBSE model will download automatically on first use
# If you have network issues, download manually:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"
```

#### 5. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using the port
lsof -i :8001  # For port 8001
lsof -i :8002  # For port 8002
lsof -i :8003  # For port 8003

# Kill the process
kill -9 <PID>

# Or use different ports
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

#### 6. Import Errors

**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
```bash
# Make sure you're in the correct directory
cd glossary-system  # or RAG-SYSTEM or prompt-construction

# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 7. spaCy Model Not Found

**Error**: `Can't find model 'en_core_web_sm'`

**Solution**:
```bash
# Download the model
python -m spacy download en_core_web_sm

# If that doesn't work, install via pip
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

#### 8. Qdrant Collection Not Found

**Error**: `Collection 'translation_memory' not found`

**Solution**:
```bash
cd RAG-SYSTEM
python scripts/setup.py

# This will create the collection and upload data
```

#### 9. Out of Memory (Embeddings)

**Error**: `CUDA out of memory` or `RAM exhausted`

**Solution**:
- Reduce batch size in `RAG-SYSTEM/app/core/config.py`:
  ```python
  BATCH_SIZE = 16  # Reduce from 32
  ```
- Use CPU instead of GPU (if using GPU):
  ```python
  # In embedding_service.py, force CPU
  model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
  ```

#### 10. PDF Processing Fails

**Error**: `Failed to extract text from PDF`

**Solution**:
- Ensure `pdfplumber` is installed: `pip install pdfplumber`
- Check PDF is not corrupted or password-protected
- Verify file size is within limits (default: 50MB)

---

## 📊 Performance Tips

### Optimization Settings

1. **Enable Redis Caching**: Significantly improves repeat query performance
2. **Connection Pooling**: Already enabled in Glossary System
3. **Embedding Cache**: RAG system caches embeddings automatically
4. **Batch Processing**: Use batch endpoints for multiple sentences

### Monitoring

Check service health and metrics:

```bash
# Glossary System health
curl http://localhost:8001/api/v1/health/services

# RAG System stats
curl http://localhost:8002/api/v1/stats

# Cache statistics
curl http://localhost:8002/api/v1/cache/stats
```

---

## 🎓 Next Steps

1. **Add Your Data**: 
   - Add glossary terms to `glossary-system/data/RAW DATA/`
   - Add translation pairs to RAG system via `scripts/add_test_sentences.py`

2. **Customize Templates**: 
   - Edit prompt templates in `prompt-construction/app/services/`

3. **Integrate with LLM**: 
   - Use the constructed prompts with your LLM service (OpenAI, local model, etc.)

4. **Deploy**: 
   - Use Docker Compose for production deployment
   - Set up monitoring and logging
   - Configure HTTPS/TLS

---

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **LaBSE Model**: https://huggingface.co/sentence-transformers/LaBSE
- **SQLite FTS5**: https://www.sqlite.org/fts5.html

---

## 💡 Tips for Development

1. **Use `--reload` flag**: Enables auto-reload on code changes
2. **Check logs**: Services log to console, check for errors
3. **Test incrementally**: Test each service independently before integration
4. **Use API docs**: Interactive docs at `/docs` endpoint are very helpful
5. **Monitor performance**: Check response times in headers (`X-Process-Time-Ms`)

---

**Happy Translating! 🚀**
