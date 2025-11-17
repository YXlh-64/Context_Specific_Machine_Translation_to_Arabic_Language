"""
QWEN Fine-tuning Script for Domain-Specific Translation
Supports independent fine-tuning for each domain using QLoRA for efficiency
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Data Collator for proper padding
@dataclass
class DataCollatorForCausalLM:
    """
    Custom data collator that properly handles padding for causal language modeling
    """
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad input_ids and labels
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            # Pad input_ids with pad_token_id
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            # Pad labels with -100 (ignore index)
            padded_lbls = lbls + [-100] * padding_length
            padded_labels.append(padded_lbls)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(ids) + [0] * padding_length
            attention_mask.append(mask)
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }
        
        return batch

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name: str = field(
        default="Qwen/Qwen2-1.5B-Instruct",
        metadata={"help": "QWEN model to fine-tune. Options: Qwen2-1.5B-Instruct, Qwen2-7B-Instruct"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization for memory efficiency"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for parameter-efficient fine-tuning"}
    )

@dataclass
class DataArguments:
    """Arguments for data configuration"""
    data_dir: str = field(
        default="data/prepared",
        metadata={"help": "Directory containing prepared data"}
    )
    domain: str = field(
        default="technology",
        metadata={"help": "Domain to train on: technology, economic, or education"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )

@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration"""
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )

class QWENTrainer:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        lora_args: LoRAArguments,
        output_dir: str
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """Load QWEN model and tokenizer with quantization if specified"""
        logger.info(f"Loading model: {self.model_args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization
        if self.model_args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("Using 4-bit quantization")
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bnb_config else torch.float16,
        )
        
        logger.info(f"Model loaded successfully. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
    def prepare_model_for_training(self):
        """Apply LoRA and prepare model for training"""
        if self.model_args.use_lora:
            logger.info("Preparing model with LoRA")
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_args.lora_r,
                lora_alpha=self.lora_args.lora_alpha,
                lora_dropout=self.lora_args.lora_dropout,
                target_modules=self.lora_args.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model.config.use_cache = False
    
    def load_datasets(self):
        """Load prepared datasets for the specified domain"""
        logger.info(f"Loading datasets for domain: {self.data_args.domain}")
        
        data_files = {
            'train': str(Path(self.data_args.data_dir) / f"{self.data_args.domain}_train.jsonl"),
            'validation': str(Path(self.data_args.data_dir) / f"{self.data_args.domain}_val.jsonl"),
            'test': str(Path(self.data_args.data_dir) / f"{self.data_args.domain}_test.jsonl")
        }
        
        # Check if files exist
        for split, file_path in data_files.items():
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        dataset = load_dataset('json', data_files=data_files)
        
        logger.info(f"Loaded datasets - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
        
        return dataset
    
    def format_chat_template(self, example):
        """Format example using QWEN chat template"""
        # Apply chat template
        messages = example['messages']
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {'text': formatted_text}
    
    def tokenize_function(self, examples):
        """Tokenize examples"""
        outputs = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.data_args.max_seq_length,
            padding=False,  # Don't pad here, let data collator handle it
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        # Copy input_ids to labels - they should have the same structure
        outputs['labels'] = outputs['input_ids'].copy()
        
        return outputs
    
    def train(self, training_args: TrainingArguments):
        """Execute training"""
        logger.info("Starting training process")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        self.prepare_model_for_training()
        
        # Load and prepare datasets
        dataset = self.load_datasets()
        
        # Format with chat template
        dataset = dataset.map(
            self.format_chat_template,
            remove_columns=dataset['train'].column_names
        )
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing dataset"
        )
        
        # Use custom data collator for proper padding
        data_collator = DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8  # Pad to multiple of 8 for efficiency
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Beginning training...")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("Training complete!")
        
        return trainer

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune QWEN for domain-specific translation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                        help="QWEN model to use")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/prepared",
                        help="Directory with prepared data")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["technology", "economic", "education"],
                        help="Domain to train on")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for the model")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation steps")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"models/{args.domain}_model"
    
    # Create argument objects
    model_args = ModelArguments(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_lora=args.use_lora
    )
    
    data_args = DataArguments(
        data_dir=args.data_dir,
        domain=args.domain,
        max_seq_length=args.max_seq_length
    )
    
    lora_args = LoRAArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="tensorboard",
        save_total_limit=2,
        remove_unused_columns=True,
    )
    
    # Initialize trainer
    qwen_trainer = QWENTrainer(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        output_dir=args.output_dir
    )
    
    # Train
    qwen_trainer.train(training_args)

if __name__ == "__main__":
    main()