# DeepL Integration Guide

## Overview

The translation system now supports **two translation services**:

1. **DeepL** (Free, Default) - Professional translation API with generous free tier
2. **OpenRouter** (Paid, Optional) - Multiple LLM models with diverse translation variants

## Why DeepL?

- **FREE**: 500,000 characters/month (no credit card required)
- **High Quality**: Professional-grade neural machine translation
- **Fast**: Faster than LLM-based translation
- **Reliable**: Consistent output without prompt engineering complexity
- **Supported Languages**: EN ↔ AR, FR ↔ AR, and many more

## Quick Start

### 1. Get Your Free DeepL API Key

1. Visit: https://www.deepl.com/pro-api
2. Sign up for a free account (no credit card needed)
3. Get your API key from the account page

### 2. Configure the Service

Create or update `.env` file in the `app/` directory:

```bash
# Use DeepL as default (free)
TRANSLATION_SERVICE=deepl
DEEPL_API_KEY=your_deepl_api_key_here

# Optional: Keep OpenRouter for advanced use cases
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3. Start the Server

```bash
cd app
python -m flask run --port=5002
```

## API Usage

### Basic Translation Request (DeepL)

```json
POST /api/translate
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "source_language": "en",
  "target_language": "ar",
  "translation_service": "deepl"
}
```

**Response:**
```json
{
  "translation": "مرحباً، كيف حالك؟",
  "translations": ["مرحباً، كيف حالك؟"],
  "source_language": "en",
  "target_language": "ar",
  "metadata": {
    "service": "deepl",
    "num_variants": 1
  }
}
```

### Using OpenRouter (Multiple Variants)

```json
POST /api/translate
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "source_language": "en",
  "target_language": "ar",
  "translation_service": "openrouter",
  "num_variants": 3
}
```

**Response:**
```json
{
  "translation": "مرحباً، كيف حالك؟",
  "translations": [
    "مرحباً، كيف حالك؟",
    "أهلاً، كيف الحال؟",
    "السلام عليكم، كيف حالك؟"
  ],
  "source_language": "en",
  "target_language": "ar",
  "metadata": {
    "service": "openrouter",
    "num_variants": 3
  }
}
```

## Service Comparison

| Feature | DeepL (Free) | OpenRouter (Paid) |
|---------|--------------|-------------------|
| **Cost** | FREE (500k chars/month) | Pay per token |
| **Speed** | Fast (~1-2 seconds) | Slower (~3-10 seconds) |
| **Variants** | 1 translation | 1-10 translations |
| **Quality** | Professional-grade | Depends on model |
| **Languages** | 30+ languages | Depends on model |
| **Setup** | Simple API key | API key + model selection |
| **Best For** | Production, high volume | Custom needs, experimentation |

## Supported Language Pairs

Both services support:
- **English → Arabic** (en → ar)
- **French → Arabic** (fr → ar)
- **Arabic → English** (ar → en)
- **Arabic → French** (ar → fr)

DeepL also supports many other language pairs (DE, ES, IT, JA, etc.)

## Configuration Details

### Environment Variables

```bash
# Service Selection (default: deepl)
TRANSLATION_SERVICE=deepl  # or 'openrouter'

# DeepL Configuration
DEEPL_API_KEY=your_key_here

# OpenRouter Configuration (optional)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4-turbo
OPENROUTER_FALLBACK_MODEL=meta-llama/llama-3.2-3b-instruct:free
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | *required* | Text to translate |
| `source_language` | string | `"en"` | Source language code |
| `target_language` | string | `"ar"` | Target language code |
| `translation_service` | string | `"deepl"` | `"deepl"` or `"openrouter"` |
| `num_variants` | integer | `1` for DeepL, `3` for OpenRouter | Number of translation variants |
| `domain` | string | `"general"` | Context domain (for OpenRouter) |
| `auto_chunk` | boolean | `true` | Auto-chunk long texts |

## PDF Translation

Both services support PDF translation with automatic chunking for long documents:

```json
POST /api/upload-file
Content-Type: multipart/form-data

file: [your_pdf_file]
source_language: en
target_language: ar
translation_service: deepl
```

The system automatically:
- Extracts text from PDF
- Chunks long documents (>15k characters)
- Translates each chunk
- Combines results

## Error Handling

### DeepL Errors

```json
{
  "error": "DeepL API quota exceeded",
  "detail": "You've used your free 500,000 characters/month limit"
}
```

**Solution:** Wait until next month or upgrade to paid DeepL plan

### OpenRouter Errors

```json
{
  "error": "Translation API error",
  "detail": "PAYMENT_REQUIRED"
}
```

**Solution:** Add credits to your OpenRouter account

## Migration from OpenRouter

If you're currently using OpenRouter and want to switch to DeepL:

1. Get DeepL API key (free)
2. Update `.env`:
   ```bash
   TRANSLATION_SERVICE=deepl
   DEEPL_API_KEY=your_deepl_key
   ```
3. Restart the server
4. Frontend automatically adapts (no changes needed)

**Note:** DeepL returns 1 translation instead of 3 variants. If your frontend expects multiple variants, it will receive an array with one element.

## Frontend Integration

The frontend doesn't need changes! Just set the service parameter:

```typescript
// Using DeepL (default, free)
const response = await fetch('/api/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: inputText,
    source_language: 'en',
    target_language: 'ar',
    translation_service: 'deepl'  // Free!
  })
});

// Using OpenRouter (paid, multiple variants)
const response = await fetch('/api/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: inputText,
    source_language: 'en',
    target_language: 'ar',
    translation_service: 'openrouter',
    num_variants: 3
  })
});
```

## Testing

### Test DeepL Integration

```bash
cd app
python3 << 'EOF'
import requests
import json

response = requests.post('http://localhost:5002/api/translate', 
    json={
        'text': 'Hello, how are you?',
        'source_language': 'en',
        'target_language': 'ar',
        'translation_service': 'deepl'
    })

print(json.dumps(response.json(), ensure_ascii=False, indent=2))
EOF
```

### Test OpenRouter (for comparison)

```bash
cd app
python3 << 'EOF'
import requests
import json

response = requests.post('http://localhost:5002/api/translate', 
    json={
        'text': 'Hello, how are you?',
        'source_language': 'en',
        'target_language': 'ar',
        'translation_service': 'openrouter',
        'num_variants': 3
    })

print(json.dumps(response.json(), ensure_ascii=False, indent=2))
EOF
```

## Troubleshooting

### "DeepL API key not set"

**Problem:** DEEPL_API_KEY environment variable missing

**Solution:**
```bash
# Add to .env file
echo "DEEPL_API_KEY=your_key_here" >> .env

# Or export in terminal
export DEEPL_API_KEY=your_key_here
```

### "DeepL API key is invalid"

**Problem:** Wrong API key or not activated

**Solution:**
1. Check your API key at https://www.deepl.com/account
2. Make sure you activated the API access
3. Verify you're using the correct key (not the website login)

### "Invalid translation_service"

**Problem:** Wrong service name in request

**Solution:** Use `"deepl"` or `"openrouter"` (lowercase)

```json
{
  "translation_service": "deepl"  // ✅ Correct
  "translation_service": "DeepL"  // ❌ Wrong
  "translation_service": "deep_l" // ❌ Wrong
}
```

## Cost Analysis

### DeepL Free Tier

- **Limit:** 500,000 characters/month
- **Cost:** FREE
- **Example:** 
  - Average translation: 100 characters
  - Monthly capacity: 5,000 translations
  - Daily capacity: ~165 translations

### OpenRouter Pricing (Example: GPT-4 Turbo)

- **Input:** $10 / 1M tokens (~$0.01 per translation)
- **Output:** $30 / 1M tokens (~$0.03 per translation)
- **Total:** ~$0.04 per 100-character translation

**Bottom Line:** DeepL is FREE for most use cases, OpenRouter costs ~$0.04 per translation.

## Best Practices

1. **Default to DeepL** for production (free, fast, reliable)
2. **Use OpenRouter** only when you need:
   - Multiple translation variants
   - Custom prompt engineering
   - Specific domain adaptation
   - Experimentation with different models

3. **Monitor your usage:**
   - DeepL dashboard: https://www.deepl.com/account
   - OpenRouter dashboard: https://openrouter.ai/activity

4. **Handle errors gracefully:**
   ```python
   try:
       result = translate(text, service='deepl')
   except QuotaExceeded:
       # Fallback to OpenRouter or inform user
       result = translate(text, service='openrouter', num_variants=1)
   ```

## Advanced: Paid DeepL Features

If you outgrow the free tier, DeepL Pro offers:

- **Starter:** €5.49/month (unlimited API, 1 user)
- **Advanced:** €25.49/month (unlimited API, CAT tool integration)
- **Ultimate:** €60.49/month (priority support)

To use paid DeepL, update the API URL in `rag_service.py`:

```python
DEEPL_API_URL = 'https://api.deepl.com/v2/translate'  # Paid tier
# Instead of: 'https://api-free.deepl.com/v2/translate'  # Free tier
```

## Summary

✅ **DeepL is now the default** (free, fast, professional quality)  
✅ **OpenRouter is optional** (paid, multiple variants, flexible)  
✅ **No frontend changes needed** (backward compatible)  
✅ **Easy to switch** between services per request  

Get your free DeepL key and start translating! 🚀
