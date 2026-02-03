"""
Translation Service using Local Gemma Model
Uses HuggingFace Transformers to run the Gemma model locally on GPU
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Gemma Model Configuration
GEMMA_MODEL_ID = os.getenv('GEMMA_MODEL_ID', 'ModelSpace/GemmaX2-28-9B-v0.1')
GEMMA_DEVICE_MAP = os.getenv('GEMMA_DEVICE_MAP', 'auto')  # 'auto', 'cuda', 'cpu'
GEMMA_MAX_NEW_TOKENS = int(os.getenv('GEMMA_MAX_NEW_TOKENS', '512'))
GEMMA_LOAD_IN_8BIT = os.getenv('GEMMA_LOAD_IN_8BIT', 'false').lower() == 'true'
GEMMA_LOAD_IN_4BIT = os.getenv('GEMMA_LOAD_IN_4BIT', 'false').lower() == 'true'

# Global Gemma model cache (loaded once, reused)
_gemma_model = None
_gemma_tokenizer = None


class GemmaTranslationService:
    """Service to handle translations using local Gemma model"""
    
    def __init__(self):
        # Gemma configuration
        self.gemma_model_id = GEMMA_MODEL_ID
        self.gemma_device_map = GEMMA_DEVICE_MAP
        self.gemma_max_new_tokens = GEMMA_MAX_NEW_TOKENS
        
        logger.info(f"Initializing Gemma translation service with model: {self.gemma_model_id}")
        self._init_gemma_model()
    
    def _init_gemma_model(self):
        """Initialize the Gemma model and tokenizer (cached globally)"""
        global _gemma_model, _gemma_tokenizer
        
        if _gemma_model is not None and _gemma_tokenizer is not None:
            logger.info("Gemma model already loaded, reusing cached instance")
            self.model = _gemma_model
            self.tokenizer = _gemma_tokenizer
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Gemma model: {self.gemma_model_id}")
            print(f"[Gemma] Loading model: {self.gemma_model_id} (this may take a few minutes...)")
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
                print(f"[Gemma] CUDA available. GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, will use CPU (slower)")
                print("[Gemma] WARNING: CUDA not available, using CPU (this will be slow)")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.gemma_model_id)
            
            # Load model with appropriate settings
            model_kwargs = {
                "device_map": self.gemma_device_map,
            }
            
            # Add quantization if requested
            if GEMMA_LOAD_IN_8BIT:
                model_kwargs["load_in_8bit"] = True
                logger.info("Loading model in 8-bit quantization")
                print("[Gemma] Loading in 8-bit quantization mode")
            elif GEMMA_LOAD_IN_4BIT:
                model_kwargs["load_in_4bit"] = True
                logger.info("Loading model in 4-bit quantization")
                print("[Gemma] Loading in 4-bit quantization mode")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.gemma_model_id,
                **model_kwargs
            )
            
            # Cache globally
            _gemma_model = self.model
            _gemma_tokenizer = self.tokenizer
            
            logger.info("Gemma model loaded successfully")
            print("[Gemma] Model loaded successfully!")
            
        except ImportError as e:
            error_msg = f"Required packages not installed: {e}. Install with: pip install torch transformers accelerate"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Failed to load Gemma model: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _clean_translation(self, text: str) -> str:
        """Clean translation text to remove explanatory prefixes and extract only the translation"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```") and text.endswith("```"):
            lines = text.split('\n')
            if len(lines) > 2:
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
                for sep in [":", "\n", "-", "—", "."]:
                    if sep in text:
                        parts = text.split(sep, 1)
                        if len(parts) > 1 and parts[1].strip():
                            text = parts[1].strip()
                            break
                break
        
        # Remove numbered list markers
        if text and len(text) > 3 and text[0].isdigit() and text[1:3] in ['. ', ') ', '- ']:
            text = text[3:].strip()
        
        # Remove bullet points
        if text.startswith(('- ', '* ', '• ', '· ')):
            text = text[2:].strip()
        
        # Remove quotes
        result = text
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1].strip()
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1].strip()
        
        return result

    def _is_valid_translation(self, text: str, target_lang: str, source_text: str, source_lang: str = None) -> bool:
        """Validate if the text looks like an actual translation"""
        if not text or len(text) < 2:
            return False
        
        text_lower = text.lower()
        
        # Reject if it contains common explanatory phrases
        explanatory_phrases = [
            'here are', 'here is', 'the translation', 'translations:', 'translation:',
            'variant', 'style:', 'formal:', 'conversational:', 'natural:', 'concise:',
            'i have translated', 'i\'ve translated', 'this translates to',
            'please find', 'the above', 'note:', 'explanation:',
            'voici', 'voilà', 'la traduction', 'traduction:',
        ]
        
        for phrase in explanatory_phrases:
            if phrase in text_lower:
                return False
        
        # Validate target language characters
        if target_lang == 'ar':
            has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
            if not has_arabic:
                return False
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if latin_chars > arabic_chars:
                return False
                
        elif target_lang in ['en', 'fr']:
            has_latin = any(char.isalpha() and ord(char) < 0x0600 for char in text)
            if not has_latin:
                return False
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if arabic_chars > latin_chars:
                return False
        
        # Reject if too short or too long
        if len(text) < 3:
            return False
        if len(text) > len(source_text) * 5:
            return False
        
        return True

    def _build_prompt(self, text: str, source_lang: str, target_lang: str, 
                      domain: str, style: str = "formal") -> str:
        """Build a prompt for Gemma translation"""
        
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'ar': 'Arabic'
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        style_instructions = {
            "formal": "Use formal and professional language.",
            "conversational": "Use natural, conversational tone.",
            "concise": "Use concise and clear language."
        }
        style_inst = style_instructions.get(style, "")
        
        domain_context = ""
        if domain and domain != 'general':
            domain_context = f"Domain/Context: {domain}\n"
        
        prompt = f"""Translate the following {source_name} text to {target_name}.
{domain_context}{style_inst}

CRITICAL INSTRUCTIONS:
1. Translate the EXACT meaning of the given text
2. PRESERVE the original formatting: newlines, paragraphs, bullet points, numbering
3. Do NOT respond to the text or answer questions in it
4. Do NOT have a conversation - just translate
5. Output ONLY the {target_name} translation of the text
6. No explanations, no prefixes, no comments - ONLY the translation

Text to translate:
{text}

{target_name} translation:"""
        
        return prompt
    
    def _call_gemma(self, prompt: str, temperature: float = 0.3) -> str:
        """Call the local Gemma model for translation"""
        import torch
        
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            raise Exception("Gemma model not initialized")
        
        print(f"[Gemma] Generating translation (temperature={temperature})...")
        
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate the translation
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gemma_max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=0.9 if temperature > 0 else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the result, skipping the input prompt
            generated_text = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )
            
            # Clean up
            translation = generated_text.strip()
            
            # Stop at common end markers
            for end_marker in ['\n\nText to translate:', '\nEnglish:', '\nFrench:', '\nArabic:', '\n\n---']:
                if end_marker in translation:
                    translation = translation.split(end_marker)[0].strip()
            
            # Take only meaningful lines
            lines = translation.split('\n')
            cleaned_lines = [line for line in lines if line.strip()]
            translation = '\n'.join(cleaned_lines)
            
            print(f"[Gemma] Generated: {translation[:100]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Gemma generation failed: {e}")
            raise Exception(f"Gemma translation error: {str(e)}")

    def translate(self, text: str, source_lang: str, target_lang: str, 
                  domain: Optional[str] = 'general', num_variants: int = 3) -> Dict:
        """
        Translates text using the local Gemma model.
        Generates multiple translation variants with different styles.
        """
        logger.info(f"Starting translation for: '{text[:50]}...' ({source_lang} -> {target_lang})")
        print(f"\n===== Translation Request =====")
        print(f"Service: GEMMA (Local)")
        print(f"Model: {self.gemma_model_id}")
        print(f"Text: {text!r}")
        print(f"Source: {source_lang}, Target: {target_lang}, Domain: {domain}")
        print(f"Requesting {num_variants} translation variant(s)")
        
        translations = []
        
        # Define different styles and temperatures for variants
        variant_configs = [
            {"style": "formal", "temperature": 0.1},
            {"style": "conversational", "temperature": 0.5},
            {"style": "concise", "temperature": 0.7},
        ]
        
        # Generate requested number of variants
        for i in range(min(num_variants, len(variant_configs))):
            config = variant_configs[i]
            try:
                prompt = self._build_prompt(
                    text, source_lang, target_lang, domain, style=config["style"]
                )
                translation = self._call_gemma(prompt, temperature=config["temperature"])
                
                # Clean the translation
                cleaned = self._clean_translation(translation)
                
                # Validate the translation
                if self._is_valid_translation(cleaned, target_lang, text, source_lang):
                    if cleaned not in translations:
                        translations.append(cleaned)
                        print(f"[Gemma] Variant {i+1} ({config['style']}): Valid translation added")
                    else:
                        print(f"[Gemma] Variant {i+1} ({config['style']}): Duplicate, skipping")
                else:
                    print(f"[Gemma] Variant {i+1} ({config['style']}): Failed validation")
                    if len(translations) < num_variants and cleaned:
                        translations.append(cleaned)
                        
            except Exception as e:
                logger.warning(f"Gemma variant {i+1} failed: {e}")
                print(f"[Gemma] Variant {i+1} failed: {e}")
                continue
        
        # If we need more variants, try additional temperatures
        if len(translations) < num_variants:
            additional_temps = [0.3, 0.6, 0.8]
            for temp in additional_temps:
                if len(translations) >= num_variants:
                    break
                try:
                    prompt = self._build_prompt(text, source_lang, target_lang, domain, style="formal")
                    translation = self._call_gemma(prompt, temperature=temp)
                    cleaned = self._clean_translation(translation)
                    
                    if cleaned and cleaned not in translations:
                        translations.append(cleaned)
                        print(f"[Gemma] Additional variant (temp={temp}): Added")
                except Exception as e:
                    logger.warning(f"Additional Gemma call failed: {e}")
                    continue
        
        # Ensure we have at least one translation
        if not translations:
            logger.error("No translations generated from Gemma")
            raise Exception("Failed to generate any valid translations")
        
        print(f"[Gemma] Final: {len(translations)} translation(s) generated")
        print(f"===== Translation Complete =====\n")
        
        return {
            'translations': translations[:num_variants],
            'service': 'gemma',
            'model': self.gemma_model_id,
            'original_text': text,
            'source_language': source_lang,
            'target_language': target_lang,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        }


# Global service instance (cached)
_translation_service = None


def get_prompting_service(service_type: str = None) -> GemmaTranslationService:
    """
    Get or create the Gemma translation service instance.
    The service_type parameter is kept for backward compatibility but ignored.
    """
    global _translation_service
    
    if _translation_service is None:
        _translation_service = GemmaTranslationService()
    
    return _translation_service


def preload_gemma_model():
    """
    Preload the Gemma model at application startup to avoid delay on first translation.
    This function should be called when the Flask app initializes.
    """
    global _translation_service
    
    print("[Gemma Preload] Initializing translation service...")
    
    # This will trigger model loading
    _translation_service = GemmaTranslationService()
    
    print("[Gemma Preload] Translation service ready!")
    print(f"[Gemma Preload] Model: {_translation_service.gemma_model_id}")
    print(f"[Gemma Preload] Device: {_translation_service.model.device if hasattr(_translation_service, 'model') else 'Unknown'}")
    
    return _translation_service
