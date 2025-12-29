# Phase 3: Prompt Construction Service

## Overview
This service assembles glossary terms (Phase 1) and fuzzy matches (Phase 2) into structured prompts ready to be passed to an LLM for translation. This service **DOES NOT** perform translation - it only constructs the prompt.

The constructed prompt is designed to be handed off to another team/service for the actual LLM translation.

## Features
- Dynamic prompt assembly with glossary terms and translation examples
- Token management to stay within model limits  
- Multi-domain support (medical, legal, education, technology, economic, general)
- Multiple prompt formats: XML, JSON, Markdown, Plain text
- Domain-specific tone guidance
- System message generation for LLM configuration

## API Endpoints

All endpoints use the `/api/v1` prefix.

### POST /api/v1/prompt/construct
Construct a translation prompt from glossary matches and fuzzy matches.

**Request:**
```json
{
    "source_sentence": "Machine learning is transforming technology.",
    "glossary_matches": [
        {"source_term": "machine learning", "target_term": "التعلم الآلي", "domain": "technology"}
    ],
    "fuzzy_matches": [
        {"source": "Deep learning transforms industries.", "target": "التعلم العميق يحول الصناعات.", "similarity_percentage": 85.0}
    ],
    "domain": "technology",
    "source_lang": "en",
    "target_lang": "ar",
    "prompt_format": "xml",
    "include_system_message": true
}
```

**Response:**
```json
{
    "prompt": "<translation_task>...</translation_task>",
    "system_message": "You are an expert translator...",
    "token_count": 450,
    "format": "xml",
    "domain": "technology"
}
```

### POST /api/v1/prompt/preview
Preview the prompt in all available formats.

### GET /api/v1/health
Health check endpoint.

### GET /api/v1/info
Service information and available options.

### GET /api/v1/domains
List all available domain types.

### GET /api/v1/formats
List all available prompt formats.

## Quick Start

```bash
cd prompt-construction
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

## Environment Variables

```env
PORT=8003
HOST=0.0.0.0

# Prompt settings
MAX_GLOSSARY_TERMS=5
MAX_FUZZY_MATCHES=7
MAX_PROMPT_TOKENS=4096
DEFAULT_DOMAIN=general

LOG_LEVEL=INFO
```

## Usage Example

```python
import requests

# Construct a prompt from glossary and RAG results
response = requests.post(
    "http://127.0.0.1:8003/api/v1/prompt/construct",
    json={
        "source_sentence": "Patients with severe symptoms must be admitted to intensive care units.",
        "glossary_matches": [
            {"source_term": "intensive care units", "target_term": "وحدات العناية المركزة", "domain": "medical"},
            {"source_term": "severe symptoms", "target_term": "الأعراض الشديدة", "domain": "medical"},
            {"source_term": "patients", "target_term": "المرضى", "domain": "medical"}
        ],
        "fuzzy_matches": [
            {
                "source": "Individuals with critical conditions should be transferred to ICU.",
                "target": "يجب نقل الأفراد ذوي الحالات الحرجة إلى العناية المركزة.",
                "similarity_percentage": 85.5,
                "domain": "medical"
            }
        ],
        "domain": "medical",
        "source_lang": "en",
        "target_lang": "ar",
        "prompt_format": "xml",
        "include_system_message": true
    }
)

result = response.json()

# Get the constructed prompt (hand off to LLM team)
prompt = result["prompt"]
system_message = result.get("system_message")
token_count = result["token_count"]

print(f"Constructed prompt ({token_count} tokens)")
print(prompt)
```

## Supported Domains
- `medical` - Healthcare and clinical terminology
- `health` - General health topics
- `legal` - Legal and regulatory content
- `education` - Educational and academic content
- `technology` - Technical and IT content
- `economic` - Financial and business content
- `general` - General purpose translation

## Supported Formats
- `xml` - Structured XML format (recommended for structured LLM input)
- `json` - JSON format
- `markdown` - Markdown format
- `plain` - Plain text format

## Output
The service outputs:
1. **prompt** - The fully constructed prompt ready for an LLM
2. **system_message** (optional) - A system message for configuring the LLM
3. **token_count** - Number of tokens in the prompt
4. **format** - The format used
5. **domain** - The domain used

This output can be directly passed to an LLM service for translation.

## ⚡ Performance Optimizations (NEW)

### Implemented Optimizations

| Feature | Description | Impact |
|---------|-------------|--------|
| **Template Caching** | Compiled Jinja2 templates in memory | 100x render speedup |
| **Prompt Memoization** | LRU cache for constructed prompts | <1ms for repeats |
| **Input Sanitization** | XSS/injection prevention | Security hardening |
| **Token Estimation** | tiktoken-based estimation | Accurate limits |
| **Async Construction** | Non-blocking prompt building | Better concurrency |

### New Service Files

```
app/services/
├── prompt_service.py              # Original service
└── optimized_prompt_service.py    # [NEW] Production service
    ├── InputSanitizer             # XML/JSON injection prevention
    ├── TokenEstimator             # Accurate token counting
    ├── PromptCache                # LRU response cache
    └── OptimizedPromptConstructor # Main service class
```

### Security Features

**Input Sanitization** prevents:
- XML injection attacks (`<![CDATA[...]]>`)
- JSON structure breaking
- XSS in rendered prompts
- Unicode exploits

### Cache Configuration

Add to `.env`:
```env
# Prompt Cache
PROMPT_CACHE_SIZE=10000
TEMPLATE_CACHE_TTL=3600

# Token Limits
MAX_PROMPT_TOKENS=4096
TOKEN_ENCODING=cl100k_base
```

## Running Tests

```bash
cd prompt-construction

# Run all tests
pytest tests/ -v

# Run comprehensive test suite
pytest tests/test_comprehensive.py -v

# Run with coverage
pytest tests/ -v --cov=app

# Run security tests only
pytest tests/test_comprehensive.py -v -k "security"
```

## 📁 Project Structure

```
prompt-construction/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   │   └── schemas.py
│   └── services/
│       ├── prompt_service.py           # Original service
│       └── optimized_prompt_service.py # [NEW] Production service
├── templates/
│   ├── xml_prompt.j2
│   ├── json_prompt.j2
│   └── markdown_prompt.j2
├── tests/
│   ├── test_api.py
│   └── test_comprehensive.py           # [NEW] Full test suite
└── requirements.txt
```
