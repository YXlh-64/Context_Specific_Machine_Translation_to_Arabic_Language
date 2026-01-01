"""
Translation Service supporting multiple backends
Supports: OpenRouter (default), DeepL (disabled)
"""

import os
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Translation Service Configuration
TRANSLATION_SERVICE = os.getenv('TRANSLATION_SERVICE', 'openrouter').lower()  # 'deepl' or 'openrouter'

# DeepL API Configuration (Free alternative) - DISABLED
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY', '')  # Free API key from deepl.com
DEEPL_API_URL = 'https://api-free.deepl.com/v2/translate'  # Free tier endpoint
# For paid accounts, use: 'https://api.deepl.com/v2/translate'

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
# Using Llama 3.3 70B for high-quality translations (excellent for EN/FR to AR)
# Alternative models: 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'google/gemini-pro-1.5'
# Free fallback: 'meta-llama/llama-3.2-3b-instruct:free' (known working free model)
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct')
OPENROUTER_FALLBACK_MODEL = os.getenv('OPENROUTER_FALLBACK_MODEL', 'meta-llama/llama-3.2-3b-instruct:free')

class PromptingTranslationService:
    """Service to handle translations using DeepL (default/free) or OpenRouter (paid)"""
    
    def __init__(self, service_type: str = None):
        # Determine which service to use
        self.service_type = service_type or TRANSLATION_SERVICE
        
        # DeepL configuration
        self.deepl_key = DEEPL_API_KEY
        self.deepl_url = DEEPL_API_URL
        
        # OpenRouter configuration
        self.openrouter_key = OPENROUTER_API_KEY
        self.openrouter_url = OPENROUTER_API_URL
        self.openrouter_model = OPENROUTER_MODEL
        self.fallback_model = OPENROUTER_FALLBACK_MODEL
        
        # Log which service is being used
        if self.service_type == 'deepl':
            if not self.deepl_key:
                logger.warning("DeepL API key not set. Get free key at: https://www.deepl.com/pro-api")
            else:
                logger.info("Using DeepL translation service (free)")
        elif self.service_type == 'openrouter':
            if not self.openrouter_key:
                logger.error("OpenRouter API key not set but service is selected")
            else:
                logger.info("Using OpenRouter translation service (paid)")
        else:
            logger.warning(f"Unknown service type: {self.service_type}. Defaulting to DeepL.")
            self.service_type = 'deepl'

    def _clean_translation(self, text: str) -> str:
        """Clean translation text to remove explanatory prefixes and extract only the translation"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```") and text.endswith("```"):
            # Remove the code block markers
            lines = text.split('\n')
            if len(lines) > 2:
                # Remove first and last lines
                text = '\n'.join(lines[1:-1]).strip()
        
        # Remove common explanatory prefixes (case-insensitive)
        prefixes_to_remove = [
            "here are three translation variants",
            "here are translation variants",
            "here is the translation",
            "here are the translations",
            "the translation is",
            "translations:",
            "translation:",
            "variant 1:",
            "variant 2:",
            "variant 3:",
            "variant:",
            "here are",
            "here is",
            "arabic translation:",
            "french translation:",
            "english translation:",
            "formal translation:",
            "conversational translation:",
            "natural translation:",
            "concise translation:",
        ]
        
        text_lower = text.lower()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                # Find where the actual translation starts (after colon or newline)
                for sep in [":", "\n", "-", "—", "."]:
                    if sep in text:
                        parts = text.split(sep, 1)
                        if len(parts) > 1 and parts[1].strip():
                            text = parts[1].strip()
                            break
                break
        
        # Remove numbered list markers (e.g., "1. ", "2. ", etc.)
        if text and len(text) > 3 and text[0].isdigit() and text[1:3] in ['. ', ') ', '- ']:
            text = text[3:].strip()
        
        # Remove bullet points and dashes at the start
        if text.startswith(('- ', '* ', '• ', '· ')):
            text = text[2:].strip()
        
        # Remove explanatory text in parentheses at the start
        if text.startswith('(') and ')' in text:
            paren_end = text.index(')')
            if paren_end < 50:  # Only if the parenthetical is relatively short
                text = text[paren_end + 1:].strip()
        
        # Remove any remaining explanatory lines at the start
        lines = text.split('\n')
        cleaned_lines = []
        skip_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip explanatory lines (but only the first few lines)
            if i < 2 and any(phrase in line_lower for phrase in [
                'here', 'the translation', 'variant', 'style:', 'formal:', 
                'conversational:', 'natural:', 'concise:', 'note:', 'explanation:'
            ]):
                skip_count += 1
                continue
            
            cleaned_lines.append(line)
        
        # If we cleaned everything away, return the original stripped text
        if not cleaned_lines:
            return text.strip()
        
        # Join the cleaned lines
        result = '\n'.join(cleaned_lines).strip()
        
        # Final cleanup: remove any leading/trailing quotes
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1].strip()
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1].strip()
        
        return result

    def _calculate_max_tokens(self, text: str, is_fallback: bool = False) -> int:
        """Calculate max_tokens based on input text length and available credits"""
        # Estimate: Arabic translations are typically 1.2-1.5x the length of source text
        # English: ~4 chars per token, Arabic: ~2-3 chars per token
        
        base_length = len(text)
        
        if is_fallback:
            # For free models, be more conservative
            # Estimate tokens needed: source text tokens + buffer for translation
            estimated_source_tokens = base_length // 4  # ~4 chars per token
            estimated_output_tokens = int(estimated_source_tokens * 2)  # 2x for Arabic expansion
            # Cap at 1000 tokens for free models
            max_tokens = max(100, min(estimated_output_tokens, 1000))
        else:
            # For paid models, be more generous
            # Calculate based on text length with proper buffer
            estimated_source_tokens = base_length // 4  # ~4 chars per token
            estimated_output_tokens = int(estimated_source_tokens * 2)  # 2x for translation expansion
            
            # Set reasonable limits based on text length
            if base_length < 500:
                # Short text: 200-500 tokens
                max_tokens = max(200, min(estimated_output_tokens, 500))
            elif base_length < 2000:
                # Medium text: 500-2000 tokens
                max_tokens = max(500, min(estimated_output_tokens, 2000))
            else:
                # Long text (PDFs): 2000-4000 tokens
                max_tokens = max(1000, min(estimated_output_tokens, 4000))
        
        return max_tokens

    def _is_valid_translation(self, text: str, target_lang: str, source_text: str, source_lang: str = None) -> bool:
        """
        Validate if the text looks like an actual translation and not explanatory text or a response
        """
        if not text or len(text) < 2:
            return False
        
        text_lower = text.lower()
        source_lower = source_text.lower()
        
        # Reject if it contains common explanatory phrases (in English or French)
        explanatory_phrases = [
            'here are', 'here is', 'the translation', 'translations:', 'translation:',
            'variant', 'style:', 'formal:', 'conversational:', 'natural:', 'concise:',
            'i have translated', 'i\'ve translated', 'this translates to',
            'please find', 'the above', 'note:', 'explanation:', 'literally:',
            'this can be translated', 'alternative translation',
            'voici', 'voilà', 'la traduction', 'traduction:', # French phrases
        ]
        
        for phrase in explanatory_phrases:
            if phrase in text_lower:
                return False
        
        # Special check: If source is a greeting question, reject common answers
        # Support for EN, FR, and AR questions
        greeting_questions = [
            'how are you', 'how are u', 'how do you do',  # English
            'comment allez-vous', 'comment vas-tu', 'comment ça va', 'ça va',  # French
            'كيف حالك', 'كيف الحال', 'كيف أنت'  # Arabic
        ]
        
        if target_lang == 'ar':
            # Check for common answer patterns when source is a greeting question
            if any(q in source_lower for q in greeting_questions):
                # These are common responses in Arabic, not translations
                response_patterns = [
                    'أنا جيد',  # I'm fine
                    'أنا بخير',  # I'm well
                    'بخير',  # fine
                    'الحمد لله',  # thanks to God (when used alone)
                    'جيد',  # good
                    'تمام',  # okay
                ]
                for pattern in response_patterns:
                    # Only reject if it doesn't contain the question word
                    if pattern in text and 'كيف' not in text:
                        print(f"[Validation] Rejected answer instead of translation: {text[:50]}")
                        return False
        
        elif target_lang in ['en', 'fr']:
            # Check for answer patterns in English/French
            if any(q in source_lower for q in greeting_questions):
                answer_patterns_en = [
                    "i'm fine", "i'm good", "i'm well", "i am fine", "i am good",
                    "fine thanks", "good thanks", "very well"
                ]
                answer_patterns_fr = [
                    "je vais bien", "ça va bien", "très bien", "je suis bien",
                    "bien merci", "ça va merci"
                ]
                
                patterns_to_check = answer_patterns_en if target_lang == 'en' else answer_patterns_fr
                
                for pattern in patterns_to_check:
                    # Only reject if it doesn't contain question words
                    question_words = ['how', 'comment'] if target_lang == 'en' else ['comment', 'how']
                    if pattern in text_lower and not any(qw in text_lower for qw in question_words):
                        print(f"[Validation] Rejected answer instead of translation: {text[:50]}")
                        return False
        
        # Validate target language characters
        if target_lang == 'ar':
            # Should contain at least some Arabic characters
            has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
            if not has_arabic:
                print(f"[Validation] Rejected: No Arabic characters in AR translation")
                return False
            
            # Should NOT be mostly English/French when translating TO Arabic
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if latin_chars > arabic_chars:
                print(f"[Validation] Rejected: More Latin than Arabic chars in AR translation")
                return False
                
        elif target_lang == 'fr':
            # Should contain at least some French/Latin characters
            has_latin = any(char.isalpha() and ord(char) < 0x0600 for char in text)
            if not has_latin:
                print(f"[Validation] Rejected: No Latin characters in FR translation")
                return False
            
            # Should NOT be mostly Arabic when translating TO French
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if arabic_chars > latin_chars:
                print(f"[Validation] Rejected: More Arabic than Latin chars in FR translation")
                return False
                
        elif target_lang == 'en':
            # Should contain at least some English/Latin characters
            has_latin = any(char.isalpha() and ord(char) < 0x0600 for char in text)
            if not has_latin:
                print(f"[Validation] Rejected: No Latin characters in EN translation")
                return False
            
            # Should NOT be mostly Arabic when translating TO English
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if arabic_chars > latin_chars:
                print(f"[Validation] Rejected: More Arabic than Latin chars in EN translation")
                return False
        
        # Reject if it's too short (likely incomplete)
        if len(text) < 3:
            return False
        
        # Reject if it's unreasonably long compared to source (> 5x length)
        if len(text) > len(source_text) * 5:
            return False
        
        # Reject if it starts with obvious instructional markers
        if text_lower.startswith(('translate', 'translation', 'here', 'the ', 'variant', 'note')):
            return False
        
        return True

    def _call_openrouter(self, system_message: str, prompt: str, num_variants: int = 1, 
                        temperature: float = 0.7, model: Optional[str] = None, 
                        max_tokens: Optional[int] = None) -> List[str]:
        """Call OpenRouter API to get translation variants"""
        model_to_use = model or self.openrouter_model
        is_fallback = model is not None and model == self.fallback_model
        print(f"--- Calling OpenRouter API (model: {model_to_use}) ---")
        
        if not self.openrouter_key:
            print("Error: OPENROUTER_API_KEY is not set.")
            logger.error("Attempted to call OpenRouter without an API key.")
            raise Exception("OpenRouter API key is missing.")

        # Calculate max_tokens if not provided
        if max_tokens is None:
            max_tokens = self._calculate_max_tokens(prompt, is_fallback=is_fallback)

        try:
            payload = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "n": num_variants,
                "max_tokens": max_tokens
            }
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000", # Optional, but good practice
                "X-Title": "Context-Specific Translation" # Optional
            }
            
            print(f"[OpenRouter] Requesting {num_variants} translation(s) for model: {model_to_use} (max_tokens: {max_tokens})")
            response = requests.post(self.openrouter_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            raw_translations = [choice['message']['content'].strip() for choice in result.get('choices', [])]
            
            # Clean translations to remove explanatory text
            translations = [self._clean_translation(t) for t in raw_translations]
            
            print(f"[OpenRouter] Success! Received {len(translations)} translation(s).")
            if translations:
                print(f"[OpenRouter] First translation preview: {translations[0][:100]}...")

            return translations
            
        except requests.exceptions.HTTPError as e:
            # Check if it's a payment/credit error (402) or model not found (404)
            if e.response:
                status_code = e.response.status_code
                error_detail = {}
                try:
                    error_detail = e.response.json()
                except Exception:
                    pass
                
                error_msg = str(e)
                
                if status_code == 402:
                    # Payment required
                    if 'credits' in str(error_detail).lower() or 'payment' in str(error_detail).lower():
                        logger.warning(f"Payment required for {model_to_use}, will try fallback model")
                        print(f"[OpenRouter] Payment required for {model_to_use}. Trying fallback model: {self.fallback_model}")
                        raise Exception("PAYMENT_REQUIRED")  # Special exception for fallback
                    else:
                        logger.error(f"OpenRouter API HTTP error: {e}")
                        raise Exception(f"Translation API error: {error_msg}")
                elif status_code == 404:
                    # Model not found
                    logger.warning(f"Model {model_to_use} not found (404), will try fallback model")
                    print(f"[OpenRouter] Model {model_to_use} not found. Trying fallback model: {self.fallback_model}")
                    raise Exception("MODEL_NOT_FOUND")  # Special exception for fallback
                else:
                    logger.error(f"OpenRouter API HTTP error: {e}")
                    print(f"Error calling OpenRouter: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_detail = e.response.json()
                            logger.error(f"OpenRouter error details: {error_detail}")
                            print(f"OpenRouter error details: {error_detail}")
                        except Exception:
                            logger.error(f"OpenRouter error response: {e.response.text}")
                            print(f"OpenRouter error response: {e.response.text}")
                    raise Exception(f"Translation API error: {error_msg}")
            else:
                logger.error(f"OpenRouter API error: {e}")
                print(f"Error calling OpenRouter: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"OpenRouter error details: {error_detail}")
                        print(f"OpenRouter error details: {error_detail}")
                    except Exception:
                        logger.error(f"OpenRouter error response: {e.response.text}")
                        print(f"OpenRouter error response: {e.response.text}")
                raise Exception(f"Translation API error: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            print(f"Error calling OpenRouter: {e}")
            raise Exception(f"Translation API error: {str(e)}")

    def _call_deepl(self, text: str, source_lang: str, target_lang: str) -> str:
        """Call DeepL API for translation (free alternative)"""
        if not self.deepl_key:
            raise Exception("DeepL API key not set. Get a free key at https://www.deepl.com/pro-api")
        
        # Map our language codes to DeepL's format
        # DeepL uses: EN, FR, AR, DE, ES, etc. (all uppercase)
        deepl_source = source_lang.upper()
        deepl_target = target_lang.upper()
        
        # DeepL requires specific codes for some languages
        # EN can be EN-US or EN-GB for target, but just EN for source
        if deepl_target == 'EN':
            deepl_target = 'EN-US'  # Default to US English
        
        print(f"--- Calling DeepL API ---")
        print(f"Source: {deepl_source}, Target: {deepl_target}")
        
        try:
            payload = {
                'auth_key': self.deepl_key,
                'text': text,
                'source_lang': deepl_source,
                'target_lang': deepl_target,
            }
            
            response = requests.post(
                self.deepl_url,
                data=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'translations' in result and len(result['translations']) > 0:
                translation = result['translations'][0]['text']
                print(f"[DeepL] Success: {translation[:100]}...")
                return translation
            else:
                logger.error(f"DeepL unexpected response format: {result}")
                raise Exception("DeepL returned unexpected response format")
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else None
            
            if status_code == 403:
                error_msg = "DeepL API key is invalid or not authorized"
                logger.error(error_msg)
                raise Exception(error_msg)
            elif status_code == 456:
                error_msg = "DeepL API quota exceeded. You've used your free 500,000 characters/month limit."
                logger.error(error_msg)
                raise Exception(error_msg)
            else:
                error_msg = f"DeepL API error (HTTP {status_code})"
                logger.error(f"{error_msg}: {e}")
                
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"DeepL error details: {error_detail}")
                        print(f"DeepL error details: {error_detail}")
                    except Exception:
                        logger.error(f"DeepL error response: {e.response.text}")
                        print(f"DeepL error response: {e.response.text}")
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepL API error: {e}")
            print(f"Error calling DeepL: {e}")
            raise Exception(f"DeepL translation error: {str(e)}")

    def translate(self, text: str, source_lang: str, target_lang: str, 
                  domain: Optional[str] = 'general', num_variants: int = 3) -> Dict:
        """
        Translates text using the configured service (DeepL by default, or OpenRouter).
        
        DeepL: Returns 1 translation (professional quality, fast, free)
        OpenRouter: Returns multiple translation variants (customizable, requires credits)
        """
        logger.info(f"Starting translation for: '{text[:50]}...' ({source_lang} -> {target_lang})")
        print(f"\n===== Translation Request =====")
        print(f"Service: {self.service_type.upper()}")
        print(f"Text: {text!r}")
        print(f"Source: {source_lang}, Target: {target_lang}, Domain: {domain}")
        
        # Route to appropriate service
        if self.service_type == 'deepl':
            return self._translate_with_deepl(text, source_lang, target_lang, domain)
        else:
            # Default to OpenRouter for backward compatibility
            print(f"Requesting {num_variants} diverse translation variants")
            return self._translate_with_openrouter(text, source_lang, target_lang, domain, num_variants)
    
    def _translate_with_deepl(self, text: str, source_lang: str, target_lang: str, domain: str) -> Dict:
        """Translate using DeepL API (free, single high-quality translation)"""
        try:
            translation = self._call_deepl(text, source_lang, target_lang)
            
            # DeepL returns clean translations, but validate anyway
            if not self._is_valid_translation(translation, target_lang, text, source_lang):
                logger.warning(f"DeepL translation failed validation: {translation[:100]}")
                raise Exception("Translation validation failed")
            
            # DeepL provides one high-quality translation
            return {
                'translations': [translation],
                'service': 'deepl',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
            raise Exception(f"Translation failed: {str(e)}")
    
    def _translate_with_openrouter(self, text: str, source_lang: str, target_lang: str, 
                                    domain: str, num_variants: int = 3) -> Dict:
        """
        Generates multiple diverse, high-quality translations using OpenRouter.
        For 3 variants, we make separate calls with different prompts/temperatures to ensure diversity.
        """
        translations = []
        
        # For multiple variants, we'll make separate calls with different approaches
        # to ensure true diversity rather than just sampling variations
        if num_variants > 1:
            # Create examples to show the model what we want (language-specific)
            example_text = ""
            if source_lang == 'en' and target_lang == 'ar':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"Hello, how are you?\"\n"
                    "Correct output: \"مرحباً، كيف حالك؟\"\n"
                    "(Note: NOT \"أنا بخير\" - that would be ANSWERING the question, not translating it)\n"
                )
            elif source_lang == 'fr' and target_lang == 'ar':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"Bonjour, comment allez-vous?\"\n"
                    "Correct output: \"مرحباً، كيف حالك؟\"\n"
                    "(Note: NOT \"أنا بخير\" - that would be ANSWERING the question, not translating it)\n"
                )
            elif source_lang == 'ar' and target_lang == 'en':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"مرحباً، كيف حالك؟\"\n"
                    "Correct output: \"Hello, how are you?\"\n"
                    "(Note: NOT \"I'm fine\" - that would be ANSWERING the question, not translating it)\n"
                )
            elif source_lang == 'ar' and target_lang == 'fr':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"مرحباً، كيف حالك؟\"\n"
                    "Correct output: \"Bonjour, comment allez-vous?\"\n"
                    "(Note: NOT \"Je vais bien\" - that would be ANSWERING the question, not translating it)\n"
                )
            
            # Variant 1: Standard formal translation
            system_message_1 = (
                f"You are a professional translator. Your task is to translate text from {source_lang} to {target_lang}. "
                f"You will receive a text that needs to be translated. "
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. Translate the EXACT meaning of the given text\n"
                f"2. PRESERVE the original formatting: newlines, paragraphs, bullet points, numbering\n"
                f"3. Do NOT respond to the text or answer questions in it\n"
                f"4. Do NOT have a conversation - just translate\n"
                f"5. Output ONLY the {target_lang} translation of the text\n"
                f"6. Keep the same structure and line breaks as the original\n"
                f"7. No explanations, no prefixes, no comments - ONLY the translation"
                f"{example_text}"
            )
            prompt_1 = f"Translate this {source_lang} text to {target_lang} (formal style). PRESERVE all line breaks and formatting:\n\n\"{text}\""
            
            # Variant 2: Natural, conversational translation
            system_message_2 = (
                f"You are a professional translator. Your task is to translate text from {source_lang} to {target_lang}. "
                f"You will receive a text that needs to be translated. "
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. Translate the EXACT meaning of the given text\n"
                f"2. PRESERVE the original formatting: newlines, paragraphs, bullet points, numbering\n"
                f"3. Do NOT respond to the text or answer questions in it\n"
                f"4. Do NOT have a conversation - just translate\n"
                f"5. Output ONLY the {target_lang} translation of the text\n"
                f"6. Use a natural, conversational tone in the translation\n"
                f"7. Keep the same structure and line breaks as the original\n"
                f"8. No explanations, no prefixes, no comments - ONLY the translation"
                f"{example_text}"
            )
            prompt_2 = f"Translate this {source_lang} text to {target_lang} (natural, conversational style). PRESERVE all line breaks and formatting:\n\n\"{text}\""
            
            # Variant 3: Concise, clear translation
            system_message_3 = (
                f"You are a professional translator. Your task is to translate text from {source_lang} to {target_lang}. "
                f"You will receive a text that needs to be translated. "
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. Translate the EXACT meaning of the given text\n"
                f"2. PRESERVE the original formatting: newlines, paragraphs, bullet points, numbering\n"
                f"3. Do NOT respond to the text or answer questions in it\n"
                f"4. Do NOT have a conversation - just translate\n"
                f"5. Output ONLY the {target_lang} translation of the text\n"
                f"6. Use a concise, clear style in the translation\n"
                f"7. Keep the same structure and line breaks as the original\n"
                f"8. No explanations, no prefixes, no comments - ONLY the translation"
                f"{example_text}"
            )
            prompt_3 = f"Translate this {source_lang} text to {target_lang} (concise style). PRESERVE all line breaks and formatting:\n\n\"{text}\""
            
            # Make parallel calls with different temperatures for additional diversity
            try:
                # Use slightly different temperatures to encourage diversity
                variant_1 = self._call_openrouter(system_message_1, prompt_1, num_variants=1, temperature=0.3)
                variant_2 = self._call_openrouter(system_message_2, prompt_2, num_variants=1, temperature=0.7)
                variant_3 = self._call_openrouter(system_message_3, prompt_3, num_variants=1, temperature=0.9)
                
                # Log what we got from each call
                print(f"[Translation] Variant 1 result: {len(variant_1) if variant_1 else 0} translation(s)")
                print(f"[Translation] Variant 2 result: {len(variant_2) if variant_2 else 0} translation(s)")
                print(f"[Translation] Variant 3 result: {len(variant_3) if variant_3 else 0} translation(s)")
                
                # Collect all translations
                all_variants = []
                if variant_1:
                    all_variants.extend(variant_1)
                if variant_2:
                    all_variants.extend(variant_2)
                if variant_3:
                    all_variants.extend(variant_3)
                
                # Filter out non-translation text and empty strings
                filtered_translations = []
                for trans in all_variants:
                    trans_clean = trans.strip()
                    # Use our validation function to ensure it's a real translation
                    if self._is_valid_translation(trans_clean, target_lang, text):
                        filtered_translations.append(trans_clean)
                
                print(f"[Translation] After validation: {len(filtered_translations)} valid translation(s)")
                
                # Remove exact duplicates (case-sensitive) but keep similar ones for diversity
                seen_exact = set()
                unique_translations = []
                for trans in filtered_translations:
                    # Only remove exact duplicates, not normalized ones
                    if trans not in seen_exact:
                        seen_exact.add(trans)
                        unique_translations.append(trans)
                
                # If we have fewer than requested, try to get more by being less strict
                if len(unique_translations) < num_variants and len(filtered_translations) >= num_variants:
                    # Use filtered translations directly if we have enough
                    unique_translations = filtered_translations[:num_variants]
                elif len(unique_translations) < num_variants:
                    # If still not enough, pad with what we have (even if duplicates)
                    while len(unique_translations) < num_variants and len(filtered_translations) > 0:
                        # Add from filtered if we haven't used it yet
                        for trans in filtered_translations:
                            if trans not in unique_translations:
                                unique_translations.append(trans)
                                break
                        else:
                            # If all are duplicates, add them anyway to reach 3
                            if filtered_translations:
                                unique_translations.append(filtered_translations[0])
                            break
                
                translations = unique_translations[:num_variants]
                print(f"[Translation] Final count: {len(translations)} translation(s) after filtering")
                
                # If we still don't have enough translations, make additional calls
                if len(translations) < num_variants:
                    logger.warning(f"Only got {len(translations)} translations, need {num_variants}. Making additional calls...")
                    # Try making additional calls with different temperatures
                    additional_temps = [0.5, 0.6, 0.8]
                    for i, temp in enumerate(additional_temps):
                        if len(translations) >= num_variants:
                            break
                        try:
                            # Use a different system message style for diversity
                            additional_system = (
                                f"You are a professional translator. Your task is to translate text from {source_lang} to {target_lang}. "
                                f"CRITICAL: Do NOT respond to the text or answer questions. Just translate the text itself. "
                                f"PRESERVE all formatting: line breaks, paragraphs, bullet points, numbering. "
                                f"Output ONLY the {target_lang} translation."
                            )
                            additional_prompt = f"Translate this {source_lang} text to {target_lang}. Keep all formatting:\n\n\"{text}\""
                            additional_variant = self._call_openrouter(additional_system, additional_prompt, num_variants=1, temperature=temp)
                            if additional_variant:
                                for trans in additional_variant:
                                    trans_clean = trans.strip()
                                    if (self._is_valid_translation(trans_clean, target_lang, text) and
                                        trans_clean not in translations):
                                        translations.append(trans_clean)
                                        if len(translations) >= num_variants:
                                            break
                        except Exception as e:
                            logger.warning(f"Additional call {i+1} failed: {e}")
                            continue
                    
                    translations = translations[:num_variants]
                    print(f"[Translation] After additional calls: {len(translations)} translation(s)")
                
            except Exception as e:
                error_str = str(e)
                # Check if it's a payment error or model not found, then try fallback model
                if "PAYMENT_REQUIRED" in error_str or "402" in error_str or "MODEL_NOT_FOUND" in error_str or "404" in error_str:
                    logger.warning(f"Primary model failed, switching to free fallback model: {self.fallback_model}")
                    print(f"[Translation] Switching to free model: {self.fallback_model}")
                    try:
                        # Try with fallback model using individual calls (more reliable than single call)
                        # This ensures we get actual translations, not explanatory text
                        variant_1_fallback = self._call_openrouter(system_message_1, prompt_1, num_variants=1, temperature=0.3, model=self.fallback_model)
                        variant_2_fallback = self._call_openrouter(system_message_2, prompt_2, num_variants=1, temperature=0.7, model=self.fallback_model)
                        variant_3_fallback = self._call_openrouter(system_message_3, prompt_3, num_variants=1, temperature=0.9, model=self.fallback_model)
                        
                        print(f"[Translation] Fallback Variant 1: {len(variant_1_fallback) if variant_1_fallback else 0}")
                        print(f"[Translation] Fallback Variant 2: {len(variant_2_fallback) if variant_2_fallback else 0}")
                        print(f"[Translation] Fallback Variant 3: {len(variant_3_fallback) if variant_3_fallback else 0}")
                        
                        # Collect all translations
                        all_variants = []
                        if variant_1_fallback:
                            all_variants.extend(variant_1_fallback)
                        if variant_2_fallback:
                            all_variants.extend(variant_2_fallback)
                        if variant_3_fallback:
                            all_variants.extend(variant_3_fallback)
                        
                        # Filter and deduplicate
                        filtered_translations = []
                        for trans in all_variants:
                            trans_clean = trans.strip()
                            if self._is_valid_translation(trans_clean, target_lang, text):
                                filtered_translations.append(trans_clean)
                        
                        # Remove exact duplicates only
                        seen_exact = set()
                        unique_translations = []
                        for trans in filtered_translations:
                            if trans not in seen_exact:
                                seen_exact.add(trans)
                                unique_translations.append(trans)
                        
                        # Ensure we have at least num_variants
                        if len(unique_translations) < num_variants:
                            # Add from filtered if needed
                            for trans in filtered_translations:
                                if len(unique_translations) >= num_variants:
                                    break
                                if trans not in unique_translations:
                                    unique_translations.append(trans)
                        
                        translations = unique_translations[:num_variants]
                        print(f"[Translation] Fallback final count: {len(translations)} translation(s)")
                        logger.info(f"Successfully used fallback model: {self.fallback_model}")
                    except Exception as fallback_error:
                        logger.error(f"Fallback model also failed: {fallback_error}")
                        # Try individual calls with fallback model (retry)
                        try:
                            variant_1 = self._call_openrouter(system_message_1, prompt_1, num_variants=1, temperature=0.3, model=self.fallback_model)
                            variant_2 = self._call_openrouter(system_message_2, prompt_2, num_variants=1, temperature=0.7, model=self.fallback_model)
                            variant_3 = self._call_openrouter(system_message_3, prompt_3, num_variants=1, temperature=0.9, model=self.fallback_model)
                            
                            all_variants = []
                            if variant_1:
                                all_variants.extend(variant_1)
                            if variant_2:
                                all_variants.extend(variant_2)
                            if variant_3:
                                all_variants.extend(variant_3)
                            
                            # Filter
                            filtered_translations = []
                            for trans in all_variants:
                                trans_clean = trans.strip()
                                if self._is_valid_translation(trans_clean, target_lang, text):
                                    filtered_translations.append(trans_clean)
                            
                            # Remove exact duplicates only
                            seen_exact = set()
                            unique_translations = []
                            for trans in filtered_translations:
                                if trans not in seen_exact:
                                    seen_exact.add(trans)
                                    unique_translations.append(trans)
                            
                            # Ensure we have enough
                            if len(unique_translations) < num_variants:
                                for trans in filtered_translations:
                                    if len(unique_translations) >= num_variants:
                                        break
                                    if trans not in unique_translations:
                                        unique_translations.append(trans)
                            
                            translations = unique_translations[:num_variants]
                        except Exception as final_error:
                            logger.error(f"All translation attempts failed: {final_error}")
                            translations = []
                else:
                    logger.error(f"Failed to get diverse translations: {e}")
                    # Fallback: try individual calls with original prompts
                    try:
                        variant_1 = self._call_openrouter(system_message_1, prompt_1, num_variants=1, temperature=0.3)
                        variant_2 = self._call_openrouter(system_message_2, prompt_2, num_variants=1, temperature=0.7)
                        variant_3 = self._call_openrouter(system_message_3, prompt_3, num_variants=1, temperature=0.9)
                        
                        all_variants = []
                        if variant_1:
                            all_variants.extend(variant_1)
                        if variant_2:
                            all_variants.extend(variant_2)
                        if variant_3:
                            all_variants.extend(variant_3)
                        
                        # Filter
                        filtered_translations = []
                        for trans in all_variants:
                            trans_clean = trans.strip()
                            if self._is_valid_translation(trans_clean, target_lang, text):
                                filtered_translations.append(trans_clean)
                        
                        # Remove exact duplicates only
                        seen_exact = set()
                        unique_translations = []
                        for trans in filtered_translations:
                            if trans not in seen_exact:
                                seen_exact.add(trans)
                                unique_translations.append(trans)
                        
                        # Ensure we have enough
                        if len(unique_translations) < num_variants:
                            for trans in filtered_translations:
                                if len(unique_translations) >= num_variants:
                                    break
                                if trans not in unique_translations:
                                    unique_translations.append(trans)
                        
                        translations = unique_translations[:num_variants]
                    except Exception as fallback_error:
                        logger.error(f"Fallback translation also failed: {fallback_error}")
                        translations = []
        else:
            # Single translation
            # Create examples to show the model what we want (language-specific)
            example_text = ""
            if source_lang == 'en' and target_lang == 'ar':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"Hello, how are you?\"\n"
                    "Correct output: \"مرحباً، كيف حالك؟\"\n"
                    "(Note: NOT \"أنا بخير\" - that would be ANSWERING the question, not translating it)\n"
                )
            elif source_lang == 'fr' and target_lang == 'ar':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"Bonjour, comment allez-vous?\"\n"
                    "Correct output: \"مرحباً، كيف حالك؟\"\n"
                    "(Note: NOT \"أنا بخير\" - that would be ANSWERING the question, not translating it)\n"
                )
            elif source_lang == 'ar' and target_lang == 'en':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"مرحباً، كيف حالك؟\"\n"
                    "Correct output: \"Hello, how are you?\"\n"
                    "(Note: NOT \"I'm fine\" - that would be ANSWERING the question, not translating it)\n"
                )
            elif source_lang == 'ar' and target_lang == 'fr':
                example_text = (
                    "\n\nExamples of correct translation:\n"
                    "Text to translate: \"مرحباً، كيف حالك؟\"\n"
                    "Correct output: \"Bonjour, comment allez-vous?\"\n"
                    "(Note: NOT \"Je vais bien\" - that would be ANSWERING the question, not translating it)\n"
                )
            
            system_message = (
                f"You are a professional translator. Your task is to translate text from {source_lang} to {target_lang}. "
                f"You will receive a text that needs to be translated. "
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. Translate the EXACT meaning of the given text\n"
                f"2. PRESERVE the original formatting: newlines, paragraphs, bullet points, numbering\n"
                f"3. Do NOT respond to the text or answer questions in it\n"
                f"4. Do NOT have a conversation - just translate\n"
                f"5. Output ONLY the {target_lang} translation of the text\n"
                f"6. Keep the same structure and line breaks as the original\n"
                f"7. No explanations, no prefixes, no comments - ONLY the translation"
                f"{example_text}"
            )
            prompt = f"Translate this {source_lang} text to {target_lang}. PRESERVE all line breaks and formatting:\n\n\"{text}\""
            
            try:
                translations = self._call_openrouter(
                    system_message=system_message,
                    prompt=prompt,
                    num_variants=1,
                    temperature=0.5
                )
                # Validate the single translation
                if translations:
                    validated = [t for t in translations if self._is_valid_translation(t, target_lang, text)]
                    translations = validated if validated else translations
            except Exception as e:
                error_str = str(e)
                # Check if it's a payment error or model not found, then try fallback model
                if "PAYMENT_REQUIRED" in error_str or "402" in error_str or "MODEL_NOT_FOUND" in error_str or "404" in error_str:
                    logger.warning(f"Primary model failed, trying fallback model: {self.fallback_model}")
                    try:
                        translations = self._call_openrouter(
                            system_message=system_message,
                            prompt=prompt,
                            num_variants=1,
                            temperature=0.5,
                            model=self.fallback_model
                        )
                        # Validate the fallback translation
                        if translations:
                            validated = [t for t in translations if self._is_valid_translation(t, target_lang, text)]
                            translations = validated if validated else translations
                    except Exception as fallback_error:
                        logger.error(f"Fallback model also failed: {fallback_error}")
                        translations = []
                else:
                    logger.error(f"Failed to get translation from OpenRouter: {e}")
                    translations = []

        # Final validation: ensure all translations are valid
        if translations:
            validated_final = [t for t in translations if self._is_valid_translation(t, target_lang, text)]
            if validated_final:
                translations = validated_final
                print(f"[Translation] Final validation: {len(translations)} valid translation(s)")
            else:
                # If all failed validation, log the issue but keep what we have
                print(f"[Translation] WARNING: All {len(translations)} translations failed final validation")
                logger.warning(f"All translations failed validation. Keeping original responses: {translations[:3]}")
        
        # If we have no translations at all, log an error
        if not translations:
            logger.error(f"No translations generated for: '{text[:50]}...'")
            print(f"[Translation] ERROR: No translations generated")

        print(f"===== Translation Finished: {len(translations)} variant(s) =====\n")
        
        return {
            "translations": translations,
            "original_text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "domain": domain,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global service instances (cached by service type)
_prompting_services = {}

def get_prompting_service(service_type: str = None) -> PromptingTranslationService:
    """
    Get or create a translation service instance
    
    Args:
        service_type: 'deepl' (free, default) or 'openrouter' (paid)
    
    Returns:
        PromptingTranslationService instance configured for the specified service
    """
    global _prompting_services
    
    # Default to environment variable or 'deepl'
    if service_type is None:
        service_type = TRANSLATION_SERVICE
    
    # Create service instance if not cached
    if service_type not in _prompting_services:
        _prompting_services[service_type] = PromptingTranslationService(service_type=service_type)
    
    return _prompting_services[service_type]
