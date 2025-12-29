"""
Token Counter Utility
Handles token counting for prompts
"""

import re
from typing import Optional


class TokenCounter:
    """
    Estimates token count for prompts.
    Uses heuristic approach: ~4 characters per token for English,
    ~2-3 characters per token for Arabic.
    """
    
    # Average characters per token by language
    CHARS_PER_TOKEN = {
        "en": 4.0,
        "ar": 2.5,  # Arabic uses fewer tokens due to UTF-8 encoding
        "default": 3.5
    }
    
    @classmethod
    def count_tokens(cls, text: str, lang: str = "default") -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: The text to count tokens for
            lang: Language code (en, ar, or default)
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        chars_per_token = cls.CHARS_PER_TOKEN.get(lang, cls.CHARS_PER_TOKEN["default"])
        
        # Count characters (excluding whitespace for more accuracy)
        char_count = len(text)
        
        # Estimate tokens
        estimated = int(char_count / chars_per_token)
        
        # Add overhead for special characters and formatting
        special_chars = len(re.findall(r'[\n\t<>{}[\]]', text))
        estimated += special_chars
        
        return max(1, estimated)
    
    @classmethod
    def count_mixed_language(cls, text: str, primary_lang: str = "en") -> int:
        """
        Count tokens for mixed language text (e.g., English with Arabic examples).
        Uses weighted average based on character distribution.
        
        Args:
            text: The text to count tokens for
            primary_lang: Primary language of the text
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Count Arabic characters (Arabic Unicode range)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))
        total_chars = len(text.replace(" ", "").replace("\n", ""))
        
        if total_chars == 0:
            return 0
        
        # Calculate weighted average
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
        english_ratio = 1 - arabic_ratio
        
        weighted_chars_per_token = (
            cls.CHARS_PER_TOKEN["ar"] * arabic_ratio +
            cls.CHARS_PER_TOKEN["en"] * english_ratio
        )
        
        estimated = int(len(text) / weighted_chars_per_token)
        
        return max(1, estimated)
    
    @classmethod
    def estimate_cost(
        cls,
        input_tokens: int,
        output_tokens: int = 500,
        model: str = "gpt-4"
    ) -> float:
        """
        Estimate API cost for a request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Expected output tokens (default 500)
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Pricing per 1M tokens (as of 2024)
        PRICING = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "default": {"input": 5.0, "output": 15.0}
        }
        
        pricing = PRICING.get(model.lower(), PRICING["default"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)


# Convenience functions
def count_tokens(text: str, lang: str = "default") -> int:
    """Count tokens for text."""
    return TokenCounter.count_tokens(text, lang)


def count_mixed_tokens(text: str) -> int:
    """Count tokens for mixed language text."""
    return TokenCounter.count_mixed_language(text)


def estimate_cost(input_tokens: int, model: str = "gpt-4") -> float:
    """Estimate API cost."""
    return TokenCounter.estimate_cost(input_tokens, model=model)
