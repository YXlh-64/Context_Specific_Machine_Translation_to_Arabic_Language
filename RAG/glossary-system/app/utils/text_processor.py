import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Try to load spaCy, but make it optional for Python 3.14 compatibility
nlp_en = None
try:
    import spacy
    nlp_en = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy English model: en_core_web_sm")
except ImportError as e:
    logger.warning(f"spaCy not available: {e}. Using fallback tokenizer.")
except Exception as e:
    logger.warning(f"Failed to load spaCy English model: {e}. Using fallback tokenizer.")

def normalize_text(text: str, lang: str) -> str:
    """
    Checklist: Normalization, Whitespace handling, Arabic specifics
    
    Args:
        text: Input text to normalize
        lang: Language code ('en', 'ar', 'fr')
        
    Returns:
        Normalized lowercase text
    """
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    if not text:
        return ""
        
    text = re.sub(r'\s+', ' ', text) # Normalize internal whitespace
    
    # if lang == 'ar':
    #     # Basic Arabic normalization (Example: Alef normalization)
    #     text = re.sub(r'[أإآ]', 'ا', text)
    #     text = re.sub(r'ة', 'ه', text)
    #     # Remove diacritics could be added here
        
    return text.lower()

def tokenize(text: str, lang: str) -> List[str]:
    """
    Checklist: Tokenization
    
    Args:
        text: Input text to tokenize
        lang: Language code ('en', 'ar', 'fr')
        
    Returns:
        List of tokens
    """
    if not text or not text.strip():
        return []
    
    if lang == 'en' and nlp_en:
        try:
            doc = nlp_en(text)
            return [token.text for token in doc if not token.is_space]
        except Exception as e:
            logger.warning(f"spaCy tokenization failed: {e}. Using fallback.")
            return text.split()

    # Fallback / Simple Split for other langs if model missing
    return text.split()

def generate_ngrams(tokens: List[str], min_n: int = 1, max_n: int = 5) -> List[dict]:
    """
    Checklist: N-gram Generation (1-10 words)
    
    Args:
        tokens: List of tokens
        min_n: Minimum n-gram size (default: 1)
        max_n: Maximum n-gram size (default: 5)
        
    Returns:
        List of dicts with text and metadata, sorted by n-gram size (largest first)
    """
    if not tokens:
        return []
    
    # Validate parameters
    if min_n < 1:
        min_n = 1
    if max_n < min_n:
        max_n = min_n
    if max_n > len(tokens):
        max_n = len(tokens)
    
    ngrams = []
    
    # Loop from max_n down to min_n (priority)
    for n in range(max_n, min_n - 1, -1):
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i : i + n]
            term = ' '.join(chunk)
            ngrams.append({
                'text': term,
                'n_size': n,
                'start_idx': i  # Crucial for overlap removal later
            })
    
    
    
    
    return ngrams