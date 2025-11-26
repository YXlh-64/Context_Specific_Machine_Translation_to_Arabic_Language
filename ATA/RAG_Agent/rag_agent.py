"""
RAG Translation Agent - Local Llama Implementation

Context-aware English-to-Arabic translation using local Llama model with RAG.
Optimized for 3.6GB VRAM using 4-bit quantization.
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from typing import Dict, List, Optional, Any
from loguru import logger
import warnings

warnings.filterwarnings("ignore")


class RAGTranslationAgent:
    """
    RAG-based translation agent using local Llama model.
    Supports translation with and without retrieval context.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        vector_db_manager = None,
        device: str = "cuda",
        max_length: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k_retrieval: int = 3,
        use_4bit: bool = True,
        **kwargs
    ):
        """
        Initialize RAG Translation Agent with local Llama model.
        
        Args:
            model_name: HuggingFace model name or local path
            vector_db_manager: VectorDBManager instance for RAG
            device: Device to run model on
            max_length: Maximum generation length
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k_retrieval: Number of examples to retrieve for context
            use_4bit: Use 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.vector_db_manager = vector_db_manager
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k_retrieval = top_k_retrieval
        
        logger.info(f"Initializing RAG Translation Agent with model: {model_name}")
        logger.info(f"Device: {device}, 4-bit quantization: {use_4bit}")
        
        # Configure 4-bit quantization for VRAM efficiency
        if use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization (NF4) for memory efficiency")
        else:
            quantization_config = None
            logger.warning("4-bit quantization disabled or CUDA not available")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        logger.info("Loading model (this may take a minute)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not use_4bit else None,
            low_cpu_mem_usage=True
        )
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        logger.info(f"  VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    def _build_prompt(
        self, 
        source_text: str, 
        context_examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Build translation prompt with optional context examples.
        
        Args:
            source_text: English text to translate
            context_examples: List of similar translation examples
            
        Returns:
            Formatted prompt string
        """
        # System instruction
        system_msg = (
            "You are a professional English-to-Arabic translator specializing in "
            "economic, financial, and political texts. Provide accurate, fluent, "
            "and contextually appropriate translations."
        )
        
        # Build prompt
        if context_examples and len(context_examples) > 0:
            # With RAG context
            context_str = "\n\nHere are similar translation examples for reference:\n"
            for i, ex in enumerate(context_examples, 1):
                context_str += f"\nExample {i}:\n"
                context_str += f"English: {ex['en']}\n"
                context_str += f"Arabic: {ex['ar']}\n"
            
            prompt = f"""<s>[INST] <<SYS>>
{system_msg}
<</SYS>>

{context_str}

Now, translate the following English text to Arabic:

English: {source_text}

Arabic: [/INST]"""
        else:
            # Without context (baseline)
            prompt = f"""<s>[INST] <<SYS>>
{system_msg}
<</SYS>>

Translate the following English text to Arabic:

English: {source_text}

Arabic: [/INST]"""
        
        return prompt
    
    def _generate_translation(self, prompt: str) -> str:
        """
        Generate translation using the Llama model.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated Arabic translation
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length * 2  # Allow space for context
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation (text after "Arabic:")
        if "Arabic:" in generated_text:
            translation = generated_text.split("Arabic:")[-1].strip()
        elif "[/INST]" in generated_text:
            translation = generated_text.split("[/INST]")[-1].strip()
        else:
            translation = generated_text.strip()
        
        # Clean up common artifacts
        translation = translation.replace("[INST]", "").replace("[/INST]", "")
        translation = translation.replace("<s>", "").replace("</s>", "")
        translation = translation.strip()
        
        return translation
    
    def translate(
        self,
        source_text: str,
        use_context: bool = True,
        return_context: bool = False,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Translate English text to Arabic with optional RAG context.
        
        Args:
            source_text: English text to translate
            use_context: Whether to use RAG context
            return_context: Whether to return retrieved examples
            top_k: Number of examples to retrieve (overrides default)
            
        Returns:
            Dictionary with translation and metadata
        """
        result = {
            "translation": "",
            "used_context": False,
            "num_retrieved": 0
        }
        
        # Retrieve context if requested
        context_examples = []
        if use_context and self.vector_db_manager is not None:
            try:
                k = top_k if top_k is not None else self.top_k_retrieval
                retrieved = self.vector_db_manager.search(
                    query=source_text,
                    top_k=k
                )
                
                if retrieved:
                    context_examples = retrieved
                    result["used_context"] = True
                    result["num_retrieved"] = len(context_examples)
                    
                    if return_context:
                        result["retrieved_examples"] = [
                            {
                                "en": ex.get("en", ""),
                                "ar": ex.get("ar", ""),
                                "similarity": ex.get("score", 0.0)
                            }
                            for ex in context_examples
                        ]
                    
                    logger.debug(f"Retrieved {len(context_examples)} context examples")
            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")
        
        # Build prompt
        prompt = self._build_prompt(source_text, context_examples)
        
        # Generate translation
        try:
            translation = self._generate_translation(prompt)
            result["translation"] = translation
            logger.debug(f"Translation generated: {translation[:50]}...")
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            result["translation"] = ""
            result["error"] = str(e)
        
        return result
    
    def batch_translate(
        self,
        source_texts: List[str],
        use_context: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts (sequential processing for memory efficiency).
        
        Args:
            source_texts: List of English texts to translate
            use_context: Whether to use RAG context
            **kwargs: Additional arguments for translate()
            
        Returns:
            List of translation results
        """
        results = []
        for text in source_texts:
            result = self.translate(text, use_context=use_context, **kwargs)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "vram_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "vram_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }


# Backward compatibility alias
GeminiRAGAgent = RAGTranslationAgent
