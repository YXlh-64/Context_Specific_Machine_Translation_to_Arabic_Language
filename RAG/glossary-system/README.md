# Glossary Lookup System - Production Ready API

A production-ready FastAPI system for glossary-based translation assistance with AI-powered term extraction.

## 🚀 Features

### Core Capabilities
- ✅ **Sentence Translation**: Process individual sentences for glossary term matches
- ✅ **PDF Processing**: Upload and batch-process PDF documents
- ✅ **Session Management**: Efficient Redis caching for repeated lookups
- ✅ **Multi-domain Support**: Health, Agriculture, History, Finance, Legal, Technology
- ✅ **Database Management**: Full CRUD operations for glossary terms
- ✅ **Full-Text Search**: Fast FTS5-powered search across all terms
- ✅ **Database Tools**: Reset, delete, and rebuild database on demand

---

## 📋 Table of Contents
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Database Setup](#-database-setup)
- [Running the Server](#-running-the-server)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🏗 Architecture

```
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──────┐  ┌──▼───────┐
│ SQLite   │  │  Redis   │
│  FTS5    │  │  Cache   │
│ Database │  │  Layer   │
└──────────┘  └──────────┘
```

**Flow**:
1. **L1 Cache**: In-memory LRU cache for ultra-fast lookups (<1ms)
2. **L2 Cache**: Redis cache for cross-session data (2-5ms)
3. **Database**: SQLite with FTS5 full-text search (10-50ms)
4. **Multi-layer**: Automatic cache hierarchy for optimal performance

---

## 📦 Prerequisites

### Required Software
- **Python**: 3.9+
- **Redis**: 5.0+ (running on localhost:6379)
- **pip**: Package installer

### System Requirements
- Windows 10/11 (or Linux/macOS with appropriate adjustments)
- 2GB RAM minimum
- 500MB disk space

---

## ⚙️ Installation

### 1. Clone/Download Project
```powershell
cd c:\Users\Star_info\Desktop\NLP-PROJECT\PROJECT\glossary-system
```

### 2. Create Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download Spacy Model
```powershell
python -m spacy download en_core_web_sm
```

Optional (for Arabic, if available):
```powershell
python -m spacy download ar_core_news_sm
```

---

## 🔧 Configuration

### Environment Variables (`.env`)
The project includes a `.env` file with the following configuration:

```dotenv
# Project Settings
PROJECT_NAME=Glossary Lookup System

# Database
DATABASE_URL=file:data/glossary.db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL_SECONDS=7200

# File Storage
UPLOAD_DIR=uploads
MAX_PDF_SIZE_MB=50
```

### Supported Domains
- `health` - Medical and healthcare terms
- `agriculture` - Farming and agricultural terminology
- `history` - Historical terms and concepts
- `finance` - Financial and economic terms
- `legal` - Legal terminology
- `technology` - Technical and IT terms

### Supported Languages
- `en` - English
- `ar` - Arabic
- `fr` - French

---

## 💾 Database Setup

### Initialize Database (First Time)
```powershell
python scripts/init_db.py
```

This creates:
- `data/glossary.db` - SQLite database
- `glossary_terms` table - Main glossary storage
- `glossary_fts` table - FTS5 full-text search index
- Auto-sync triggers
- Sample seed data (15 terms across multiple domains)

**After running the seed script** (`seed_from_raw_data.py`):
- **3,870+ terms** across 5 domains (economy, media, finance, health, technology)
- Full English-Arabic glossary coverage for economic and media domains

### Manage Database via API
You can now manage the database through API endpoints or the web interface:
- Add/search/delete terms
- View statistics
- Reset or delete database

### Seed Database with CSV Data

#### Option 1: From Verified CSV File
If you have a pre-processed CSV file (`data/verified_glossary_terms.csv`):
```powershell
python scripts/seed_glossary.py
```

#### Option 2: From Raw CSV Files (Recommended)
Automatically process and seed from raw CSV files in `data/RAW DATA/`:
```powershell
python scripts/seed_from_raw_data.py
```

This script:
- ✅ Reads all CSV files from `data/RAW DATA/` subfolders
- ✅ Automatically detects language pairs from folder names (`en-ar`, `fr-ar`)
- ✅ Extracts domains from filenames (`economic_english_arabic.csv` → `economy`)
- ✅ Calculates n-gram sizes and sets default frequencies
- ✅ Handles multiple column name variations
- ✅ Uses `INSERT OR IGNORE` to avoid duplicates

**Supported folder structure**:
```
data/RAW DATA/
├── en-ar/
│   ├── economic_english_arabic.csv
│   └── media_english_arabic.csv
└── fr-ar/
    └── culture_french_arabic.csv
```

**Expected CSV format** (flexible column names):
```csv
English,Arabic
artificial intelligence,الذكاء الاصطناعي
machine learning,التعلم الآلي
```

**Clear database before reseeding**:
```powershell
python scripts/seed_from_raw_data.py --clear
```

---

## 🚀 Running the Server

### Start Redis (Required)
Ensure Redis is running on localhost:6379:
```powershell
# If Redis is installed as a service
Start-Service Redis

# Or manually (if using WSL or Docker)
redis-server
```

### Start FastAPI Server
```powershell
uvicorn app.main:app --reload --port 8001
```

**Server will be available at**: http://127.0.0.1:8001

### Access Documentation
- **Swagger UI**: http://127.0.0.1:8001/docs
- **ReDoc**: http://127.0.0.1:8001/redoc
- **OpenAPI JSON**: http://127.0.0.1:8001/openapi.json

---

## 🌐 API Endpoints

### Health & Status

#### `GET /api/v1/health/services`
Check service health (Redis, configuration).

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "redis": {
      "status": "connected",
      "host": "localhost",
      "port": 6379
    }
  },
  "config": {
    "max_pdf_size_mb": 50,
    "cache_ttl_seconds": 7200,
    "allowed_domains": ["health", "technology", ...],
    "allowed_langs": ["en", "ar", "fr"]
  }
}
```

---

### Sentence Translation

#### `POST /api/v1/translate/sentence`
Translate a single sentence using glossary lookup (with AI fallback).

**Request Body**:
```json
{
  "text": "Artificial intelligence and machine learning are transforming healthcare.",
  "source_lang": "en",
  "target_lang": "ar",
  "domain": "technology"
}
```

**Response**:
```json
{
  "glossary_matches": [
    {
      "source_term": "artificial intelligence",
      "target_term": "الذكاء الاصطناعي",
      "n_gram_size": 2,
      "frequency": 20,
      "cache_hit": true
    },
    {
      "source_term": "machine learning",
      "target_term": "التعلم الآلي",
      "n_gram_size": 2,
      "frequency": 15,
      "cache_hit": true
    }
  ],
  "match_count": 2,
  "source_sentence": "Artificial intelligence...",
  "domain": "technology",
  "processing_time_ms": 3.64
}
```

**Notes**:
- Results are cached for fast subsequent lookups
- Multi-layer caching: L1 (LRU) < 1ms, L2 (Redis) 2-5ms
- Database queries optimized with FTS5 index

---

### PDF Processing

#### 1. `POST /api/v1/translate/pdf`
Upload PDF and create session.

**Form Data**:
- `file`: PDF file (max 50MB)
- `source_lang`: `en` | `ar` | `fr`
- `target_lang`: `en` | `ar` | `fr`
- `domain`: Domain name
- `auto_process`: `true` | `false` (optional, start processing immediately)

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "initialized",
  "domain": "technology",
  "source_lang": "en",
  "target_lang": "ar",
  "glossary_terms_loaded": 457,
  "total_pages": 15,
  "cache_expires_in_seconds": 7200,
  "auto_processing": false
}
```

#### 2. `GET /api/v1/session/{session_id}`
Get session status.

**Response**:
```json
{
  "session_id": "550e8400-...",
  "status": "initialized",
  "domain": "technology",
  "source_lang": "en",
  "target_lang": "ar",
  "glossary_count": 457,
  "total_pages": 15,
  "processed_sentences": 0,
  "created_at": "2025-12-06T10:30:00"
}
```

#### 3. `GET /api/v1/session/{session_id}/extract`
Extract text from PDF.

**Query Parameters**:
- `pages`: Comma-separated page numbers (optional, e.g., `1,2,5`)

**Response**:
```json
{
  "session_id": "550e8400-...",
  "extracted_text": "Full text from PDF...",
  "total_pages": 15,
  "pages_extracted": [1, 2, 3, ...]
}
```

#### 4. `GET /api/v1/session/{session_id}/glossary-terms`
Get ALL glossary terms found in PDF.

**Query Parameters**:
- `pages`: Comma-separated page numbers (optional)
- `include_context`: `true` | `false` (include example sentences)

**Response**:
```json
{
  "session_id": "550e8400-...",
  "terms": [
    {
      "source_term": "artificial intelligence",
      "target_term": "الذكاء الاصطناعي",
      "occurrences": 5,
      "example_sentences": [
        "Artificial intelligence is transforming industries.",
        "..."
      ]
    }
  ],
  "total_unique_terms": 25,
  "pages_scanned": [1, 2, 3, ...],
  "processing_time_ms": 450.2
}
```

#### 5. `POST /api/v1/session/{session_id}/process/sentence`
Process single sentence within session.

**Request Body**:
```json
{
  "sentence": "Machine learning algorithms are improving."
}
```

**Response**: Same as sentence translation endpoint.

#### 6. `POST /api/v1/session/{session_id}/process/batch`
Process entire PDF or specific pages.

**Query Parameters**:
- `pages`: Comma-separated page numbers (optional)
- `batch_size`: Sentences per batch (1-100, default: 10)
- `async_mode`: `true` | `false` (process in background)

**Response**:
```json
{
  "session_id": "550e8400-...",
  "status": "completed",
  "total_pages": 15,
  "total_sentences": 342,
  "total_matches": 128,
  "results": [
    {
      "sentence": "...",
      "matches": [...]
    }
  ],
  "processing_time_ms": 5432.1
}
```

#### 7. `DELETE /api/v1/session/{session_id}`
Cleanup session and release resources.

**Response**:
```json
{
  "status": "success",
  "session_id": "550e8400-...",
  "message": "Session cleaned up successfully"
}
```

---

### Database Management Endpoints

#### `GET /api/v1/database/stats`
Get database statistics including term counts by domain and language.

**Response**:
```json
{
  "total_terms": 500,
  "by_domain": {
    "health": 150,
    "technology": 200,
    "finance": 150
  },
  "by_language_pair": [
    {"source": "en", "target": "ar", "count": 400},
    {"source": "en", "target": "fr", "count": 100}
  ],
  "database_size_mb": 2.5,
  "database_path": "data/glossary.db"
}
```

#### `POST /api/v1/database/terms`
Add a new term to the glossary.

**Query Parameters**:
- `source_term`: Source term (required)
- `target_term`: Target term (required)
- `source_lang`: Source language (default: "en")
- `target_lang`: Target language (default: "ar")
- `domain`: Domain (default: "general")
- `frequency`: Frequency count (default: 1)

**Response**:
```json
{
  "status": "success",
  "message": "Term added successfully",
  "term": {
    "id": 501,
    "source_term": "neural network",
    "target_term": "شبكة عصبية",
    "source_lang": "en",
    "target_lang": "ar",
    "domain": "technology",
    "n_gram_size": 2,
    "frequency": 1
  }
}
```

#### `GET /api/v1/database/search`
Search for terms using FTS5 full-text search.

**Query Parameters**:
- `query`: Search query (required)
- `limit`: Max results (default: 20, max: 1000)

**Response**:
```json
{
  "query": "neural",
  "total_results": 5,
  "results": [
    {
      "source_term": "neural network",
      "target_term": "شبكة عصبية",
      "source_lang": "en",
      "target_lang": "ar",
      "domain": "technology",
      "n_gram_size": 2,
      "frequency": 10
    }
  ]
}
```

#### `POST /api/v1/database/reset`
Reset database by recreating schema (deletes all terms).

⚠️ **WARNING**: This is destructive and cannot be undone!

**Response**:
```json
{
  "status": "success",
  "message": "Database reset successfully",
  "database_path": "data/glossary.db"
}
```

#### `DELETE /api/v1/database`
Delete the entire database file.

⚠️ **WARNING**: This is irreversible!

**Response**:
```json
{
  "status": "success",
  "message": "Database deleted successfully",
  "deleted_path": "data/glossary.db"
}
```

---

### Admin Endpoints

#### `GET /api/v1/sessions`
List all active sessions.

#### `DELETE /api/v1/sessions`
⚠️ **DANGER**: Close all sessions and delete all files.

#### `GET /api/v1/session/{session_id}/stats`
Get cache statistics for session.

---

## 📘 Usage Examples

### Example 1: Simple Sentence Translation
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/translate/sentence" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "Quantum computing is revolutionary.", "source_lang": "en", "target_lang": "ar", "domain": "technology"}'
```

### Example 2: Upload PDF with Auto-Processing
```powershell
$form = @{
    file = Get-Item "document.pdf"
    source_lang = "en"
    target_lang = "ar"
    domain = "technology"
    auto_process = "true"
}

Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/translate/pdf" `
  -Method POST `
  -Form $form
```

### Example 3: Process PDF Manually
```powershell
# 1. Upload PDF
$upload = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/translate/pdf" `
  -Method POST -Form @{
    file = Get-Item "doc.pdf"
    source_lang = "en"
    target_lang = "ar"
    domain = "health"
  }

$sessionId = $upload.session_id

# 2. Extract text from specific pages
$text = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/session/$sessionId/extract?pages=1,2,3"

# 3. Get all glossary terms in PDF
$terms = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/session/$sessionId/glossary-terms?include_context=true"

# 4. Process all sentences
$results = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/session/$sessionId/process/batch" -Method POST

# 5. Cleanup
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/session/$sessionId" -Method DELETE
```

### Example 4: Database Management
```powershell
# Get database statistics
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/database/stats"

# Add a new term
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/database/terms?source_term=cloud computing&target_term=الحوسبة السحابية&domain=technology" -Method POST

# Search for terms
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/database/search?query=cloud&limit=10"
```

---

## 🧪 Testing

### Run Unit Tests
```powershell
pytest tests/ -v
```

### Test API Endpoints
```powershell
python tests/test_api.py
```

### Manual Testing
Use the interactive Swagger UI at http://127.0.0.1:8001/docs

---

## 📁 Project Structure

```
glossary-system/
├── .env                          # Environment configuration
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── app/
│   ├── main.py                   # FastAPI application entry point
│   ├── api/
│   │   └── routes.py             # All API endpoints
│   ├── core/
│   │   ├── config.py             # Settings & environment variables
│   │   ├── database.py           # SQLite connection manager
│   │   ├── redis.py              # Redis connection manager
│   │   ├── connection_pool.py    # [NEW] Thread-safe SQLite connection pooling
│   │   └── lru_cache.py          # [NEW] Multi-layer LRU + Redis cache
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   ├── services/
│   │   ├── cache_service.py      # Redis caching logic
│   │   ├── glossary_service.py   # Main glossary lookup service
│   │   ├── optimized_glossary_service.py  # Production-optimized service
│   │   └── pdf_service.py        # PDF processing & session management
│   └── utils/
│       ├── file_utils.py         # File handling utilities
│       └── text_processor.py     # Text tokenization & processing
│
├── data/
│   ├── glossary.db               # SQLite database (FTS5)
│   └── verified_glossary_terms.csv # CSV seed data
│
├── scripts/
│   ├── init_db.py                # Initialize database schema
│   ├── migrate_add_origin.py     # Add 'origin' column migration
│   ├── seed_glossary.py          # Seed database from CSV
│   ├── check_agent_terms.py      # Check AI-discovered terms
│   └── generate_test_pdf.py      # Generate test PDF
│
├── tests/
│   ├── test_api.py               # API endpoint tests
│   └── test_comprehensive.py     # [NEW] Full test suite with edge cases
│
├── uploads/                      # Uploaded PDF files (session-based)
└── outputs/                      # Processing outputs (if needed)
```

---

## ⚡ Performance Optimizations (NEW)

### Implemented Optimizations

| Feature | Description | Impact |
|---------|-------------|--------|
| **Connection Pooling** | Thread-safe SQLite pool with recycling | 5x throughput |
| **LRU Cache (L1)** | In-memory cache with O(1) access | <1ms lookups |
| **Redis Cache (L2)** | Distributed cache for sessions | Cross-request sharing |
| **Async Operations** | Non-blocking I/O with ThreadPoolExecutor | Better concurrency |
| **Batch Queries** | N-gram lookups in single transaction | 3x faster |
| **FTS5 Search** | Full-text search with ranking | Fast term lookup |

### Cache Layers

```
Request → L1 LRU Cache → L2 Redis Cache → SQLite Database
              ↑              ↑                  ↑
           <1ms           5-10ms            20-50ms
```

### Configuration

Add to `.env`:
```env
# Connection Pool
POOL_SIZE=5
POOL_TIMEOUT=30
MAX_CONNECTIONS=10

# LRU Cache
LRU_CACHE_SIZE=10000
CACHE_TTL_SECONDS=3600
```

---

## 🔍 Troubleshooting


### Redis Connection Errors
**Error**: `Connection refused` or `Redis disconnected`

**Solutions**:
1. Check if Redis is running:
   ```powershell
   Test-NetConnection localhost -Port 6379
   ```

2. Start Redis:
   ```powershell
   Start-Service Redis
   # or
   redis-server
   ```

3. Verify `.env` configuration:
   ```
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

### Database Errors
**Error**: Database file not found or corrupted

**Solution**: Initialize or reset the database:
```powershell
# Initialize new database
python scripts/init_db.py

# Or reset via API
Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/database/reset" -Method POST
```

### Port Already in Use
**Error**: `[WinError 10013] Address already in use`

**Solution**:
```powershell
# Find process using port 8001
netstat -ano | findstr :8001

# Kill process (replace PID)
Stop-Process -Id <PID> -Force

# Or use different port
uvicorn app.main:app --reload --port 8002
```

### Spacy Model Missing
**Error**: `Can't find model 'en_core_web_sm'`

**Solution**:
```powershell
python -m spacy download en_core_web_sm
```

### PDF Upload Fails
**Error**: `File must be a PDF document`

**Checks**:
1. File extension is `.pdf`
2. File size < 50MB (configurable in `.env`)
3. File is not corrupted

---

## 📊 Performance Metrics

| Operation | L1 Cache (LRU) | L2 Cache (Redis) | Database |
|-----------|----------------|------------------|----------|
| Glossary lookup | <1 ms | 2-5 ms | 10-50 ms |
| Full-text search | N/A | N/A | 20-100 ms |
| PDF batch processing (100 sentences) | N/A | N/A | 500-2000 ms |
| Add new term | N/A | N/A | 5-15 ms |

**Cache Hit Ratio**: ~85-95% with proper warming

---

## 🔐 Security Considerations

### Production Deployment
1. **API Key**: Store `LLM_API_KEY` securely (environment variable, vault)
2. **CORS**: Update `allow_origins` in `app/main.py` to whitelist specific domains
3. **File Upload**: Validate PDF files thoroughly (antivirus scan recommended)
4. **Rate Limiting**: Add rate limiting middleware for production
5. **HTTPS**: Use reverse proxy (nginx/Caddy) with SSL certificate

### Environment Variables
Never commit `.env` to version control. Use `.env.example` as template.

---

## 📝 API Version

**Current Version**: `v1`

All endpoints are prefixed with `/api/v1/`

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

This project is proprietary. All rights reserved.

---

## 👥 Support

For issues or questions:
- Check logs: Server logs show detailed error messages
- Review Swagger docs: http://127.0.0.1:8001/docs
- Inspect Redis: Use `redis-cli` to check cached data

---

## 🎯 Quick Start Checklist

- [ ] Python 3.9+ installed
- [ ] Redis running on localhost:6379
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Spacy model downloaded (`python -m spacy download en_core_web_sm`)
- [ ] Database initialized (`python scripts/init_db.py`)
- [ ] `.env` file configured
- [ ] Server running (`uvicorn app.main:app --reload --port 8001`)
- [ ] Test endpoint: http://127.0.0.1:8001/api/v1/health/services
- [ ] Optional: Access web interface at http://localhost:8501 (Streamlit)

---

## 🌐 Web Interface

The system includes a comprehensive Streamlit web interface for easy management:

```powershell
# Start the web interface (separate terminal)
cd ../interface
streamlit run app.py
```

Access at: http://localhost:8501

### Interface Features:
- 📊 **Database Manager** - Add, search, view statistics, reset/delete database
- 🧪 **API Tester** - Test all endpoints with interactive forms
- 📈 **Service Dashboard** - Monitor service health and logs
- 🔄 **Session Manager** - View and manage PDF processing sessions

---

**Status**: ✅ Production Ready | 💾 Database-Powered | ⚡ Fast & Reliable
