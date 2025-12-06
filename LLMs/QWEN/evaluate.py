"""
Evaluation and Inference Script for Fine-tuned QWEN Models
Supports translation inference and evaluation using BLEU scores
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
import sacrebleu
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QWENTranslator:
    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        domain: str = "technology",
        use_4bit: bool = True
    ):
        """
        Initialize QWEN translator
        
        Args:
            model_path: Path to fine-tuned model (LoRA adapters)
            base_model: Base QWEN model name
            domain: Domain of the model
            use_4bit: Use 4-bit quantization
        """
        self.model_path = model_path
        self.base_model = base_model
        self.domain = domain
        self.use_4bit = use_4bit
        
        self.tokenizer = None
        self.model = None
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Configure quantization
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bnb_config else torch.float16
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base, self.model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def translate(self, text: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Translate English text to Arabic
        
        Args:
            text: English text to translate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Translated Arabic text
        """
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": f"You are an expert translator specializing in {self.domain} domain translations between English and Arabic."
            },
            {
                "role": "user",
                "content": f"Translate the following English text to Arabic. Domain: {self.domain}\n\nText: {text}"
            }
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation (remove prompt)
        # The response is after the last assistant marker
        translation = generated_text.split("assistant")[-1].strip()
        
        return translation
    
    def batch_translate(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Translate multiple texts"""
        translations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            batch_translations = [self.translate(text) for text in batch]
            translations.extend(batch_translations)
        
        return translations

class ModelEvaluator:
    def __init__(self, model_path: str, base_model: str, domain: str, test_data_path: str):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to fine-tuned model
            base_model: Base QWEN model name
            domain: Domain name
            test_data_path: Path to test dataset
        """
        self.translator = QWENTranslator(model_path, base_model, domain)
        self.domain = domain
        self.test_data_path = test_data_path
    
    def load_test_data(self) -> Tuple[List[str], List[str]]:
        """Load test dataset"""
        logger.info(f"Loading test data from {self.test_data_path}")
        
        dataset = load_dataset('json', data_files={'test': self.test_data_path})['test']
        
        sources = []
        references = []
        
        for example in dataset:
            messages = example['messages']
            # Extract English text from user message
            user_msg = [msg for msg in messages if msg['role'] == 'user'][0]['content']
            english_text = user_msg.split("Text: ")[-1].strip()
            
            # Extract Arabic reference from assistant message
            assistant_msg = [msg for msg in messages if msg['role'] == 'assistant'][0]['content']
            
            sources.append(english_text)
            references.append(assistant_msg)
        
        logger.info(f"Loaded {len(sources)} test examples")
        return sources, references
    
    def evaluate(self, sample_size: int = None) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting evaluation for {self.domain} domain")
        
        # Load test data
        sources, references = self.load_test_data()
        
        # Limit sample size if specified
        if sample_size and sample_size < len(sources):
            sources = sources[:sample_size]
            references = references[:sample_size]
            logger.info(f"Evaluating on {sample_size} samples")
        
        # Generate translations
        logger.info("Generating translations...")
        translations = self.translator.batch_translate(sources)
        
        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(translations, [references])
        
        # Calculate character-level BLEU (better for Arabic)
        chrf = sacrebleu.corpus_chrf(translations, [references])
        
        results = {
            'domain': self.domain,
            'num_samples': len(sources),
            'bleu_score': bleu.score,
            'chrf_score': chrf.score,
            'examples': []
        }
        
        # Add some example translations
        num_examples = min(5, len(sources))
        for i in range(num_examples):
            results['examples'].append({
                'source': sources[i],
                'reference': references[i],
                'translation': translations[i]
            })
        
        logger.info(f"Evaluation complete!")
        logger.info(f"BLEU Score: {bleu.score:.2f}")
        logger.info(f"chrF Score: {chrf.score:.2f}")
        
        return results

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned QWEN models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                        help="Base QWEN model")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["technology", "economic", "education"],
                        help="Domain of the model")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data (default: data/prepared/{domain}_test.jsonl)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (default: results/{domain}_results.json)")
    parser.add_argument("--translate_text", type=str, default=None,
                        help="Translate a specific text (interactive mode)")
    
    args = parser.parse_args()
    
    # Set default paths
    if args.test_data is None:
        args.test_data = f"data/prepared/{args.domain}_test.jsonl"
    
    if args.output_file is None:
        os.makedirs("results", exist_ok=True)
        args.output_file = f"results/{args.domain}_results.json"
    
    # Interactive translation mode
    if args.translate_text:
        translator = QWENTranslator(args.model_path, args.base_model, args.domain)
        translation = translator.translate(args.translate_text)
        print(f"\nSource: {args.translate_text}")
        print(f"Translation: {translation}")
        return
    
    # Evaluation mode
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        base_model=args.base_model,
        domain=args.domain,
        test_data_path=args.test_data
    )
    
    results = evaluator.evaluate(sample_size=args.sample_size)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
