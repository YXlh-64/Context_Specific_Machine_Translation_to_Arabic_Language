# Quick Reference Guide

## 🚀 Quick Start

```bash
# 1. Run setup script
./quick_start.sh

# 2. Start all services
./start_all_services.sh

# 3. Access API docs
# Open browser: http://localhost:8001/docs
```

## 📍 Service Ports

| Service | Port | URL | Docs |
|---------|------|-----|------|
| Glossary System | 8001 | http://localhost:8001 | /docs |
| RAG System | 8002 | http://localhost:8002 | /docs |
| Prompt Construction | 8003 | http://localhost:8003 | /docs |

## 🔧 Prerequisites

- **Python 3.10+**
- **Redis** (optional, port 6379)
- **Qdrant** (required, port 6333)
- **Docker** (for Qdrant)

## 📦 Installation

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r glossary-system/requirements.txt
pip install -r RAG-SYSTEM/requirements.txt
pip install -r prompt-construction/requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🐳 Start External Services

```bash
# Redis (optional)
redis-server
# OR
docker run -d -p 6379:6379 redis:7-alpine

# Qdrant (required)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## 🗄️ Database Setup

```bash
# Initialize Glossary DB
cd glossary-system
python scripts/init_db.py
python scripts/seed_glossary.py

# Setup Qdrant Collection
cd ../RAG-SYSTEM
python scripts/setup.py
```

## ▶️ Running Services

### Option 1: All at once (background)
```bash
./start_all_services.sh
```

### Option 2: Separate terminals
```bash
# Terminal 1
cd glossary-system
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2
cd RAG-SYSTEM
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

# Terminal 3
cd prompt-construction
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

## 🧪 Test Services

```bash
# Health checks
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Run tests
cd glossary-system && pytest tests/
cd ../RAG-SYSTEM && pytest tests/
cd ../prompt-construction && pytest tests/
```

## 📡 API Examples

### Glossary Lookup
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

### RAG Search
```bash
curl -X POST http://localhost:8002/api/v1/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep neural networks",
    "domain": "technology",
    "top_k": 5
  }'
```

### Prompt Construction
```bash
curl -X POST http://localhost:8003/api/v1/prompt/construct \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "Machine learning models require training data",
    "glossary_matches": [{"source_term": "machine learning", "target_term": "التعلم الآلي"}],
    "similar_examples": [{"source_text": "...", "target_text": "...", "score": 0.85}],
    "domain": "technology",
    "format": "xml"
  }'
```

## 🛑 Stop Services

```bash
# Stop all
./stop_all_services.sh

# Or manually
kill $(lsof -ti:8001)
kill $(lsof -ti:8002)
kill $(lsof -ti:8003)
```

## 📁 Key Directories

```
glossary-system/
  ├── data/glossary.db          # SQLite database
  ├── data/RAW DATA/            # CSV source files
  └── scripts/seed_glossary.py  # Populate DB

RAG-SYSTEM/
  ├── scripts/setup.py          # Setup Qdrant
  └── scripts/add_test_sentences.py  # Add data

prompt-construction/
  └── app/services/             # Prompt templates
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | `lsof -ti:8001 \| xargs kill` |
| Database not found | Run `python scripts/init_db.py` |
| Qdrant not found | Start Docker: `docker run -d -p 6333:6333 qdrant/qdrant` |
| Redis connection refused | Start Redis or ignore (optional) |
| Module not found | Activate venv: `source venv/bin/activate` |

## 📚 Documentation

- **Full Guide**: `COMPREHENSIVE_GUIDE.md`
- **Main README**: `README.md`
- **API Docs**: http://localhost:8001/docs (each service)

## 🎯 Common Workflows

### 1. Translate a sentence
```python
# Step 1: Glossary lookup
glossary = requests.post("http://localhost:8001/api/v1/translate/sentence", json={...})

# Step 2: RAG search
rag = requests.post("http://localhost:8002/api/v1/search/semantic", json={...})

# Step 3: Build prompt
prompt = requests.post("http://localhost:8003/api/v1/prompt/construct", json={...})
```

### 2. Process PDF
```python
# Upload PDF
response = requests.post("http://localhost:8001/api/v1/translate/pdf", files={...})
session_id = response.json()["session_id"]

# Process sentences
results = requests.post(f"http://localhost:8001/api/v1/session/{session_id}/process/batch", json={...})
```

## 💡 Tips

- Use `--reload` flag for development (auto-reload on changes)
- Check logs in `logs/` directory when running in background
- Redis is optional but improves performance significantly
- Qdrant needs data - run `setup.py` to populate collection

---

**For detailed information, see `COMPREHENSIVE_GUIDE.md`**
