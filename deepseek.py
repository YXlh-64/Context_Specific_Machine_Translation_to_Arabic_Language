# Suppress dependency warnings
import warnings
warnings.filterwarnings('ignore')

print("📦 Installing packages...")

# CRITICAL FIX for Kaggle CUDA 12.4 compatibility
# Use pre-built wheel that works on Kaggle
print("🔧 Installing bitsandbytes (pre-built for CUDA 12.1)...")
!pip uninstall -y bitsandbytes
!pip install -q https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-any.whl

# Install other packages with compatible versions
print("📦 Installing transformers, peft, accelerate, datasets...")
!pip install -q transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 datasets

# Verify GPU is available
import torch
print(f"\n🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"🔧 CUDA Version: {torch.version.cuda}")
else:
    print("❌ No GPU available! Enable GPU in Settings → Accelerator → GPU T4")

# Verify bitsandbytes installation
print(f"\n🔍 Testing bitsandbytes...")
try:
    import bitsandbytes as bnb
    print(f"✅ bitsandbytes {bnb.__version__} loaded successfully!")
    
    # Test actual quantization functionality
    test_tensor = torch.randn(10, 10).cuda()
    from bitsandbytes.nn import Linear4bit
    test_layer = Linear4bit(10, 10)
    print(f"✅ 4-bit quantization support: Working!")
    
    print(f"\n🎉 All packages installed successfully!")
    print(f"   Ready for QLoRA fine-tuning on T4 GPU")
    
except Exception as e:
    print(f"❌ Critical error: {e}")
    print("\n⚠️  This version of bitsandbytes doesn't work on Kaggle CUDA 12.4")
    print("\n💡 RECOMMENDED SOLUTION:")
    print("   Switch to Google Colab which has better bitsandbytes support")
    print("   - Same notebooks work on Colab (auto-detects environment)")
    print("   - No manual file upload/download needed")
    print("   - More stable package ecosystem")
    raise
import os
from pathlib import Path

# Auto-detect environment (Kaggle vs Colab)
if os.path.exists('/kaggle/working'):
    # Running on Kaggle
    print("📍 Running on Kaggle")
    
    # Check for input data from previous notebook
    # Option 1: Added as Kaggle dataset input
    possible_input_paths = [
        Path('/kaggle/input'),
    ]
    
    # Find the JSONL files
    train_file = None
    dev_file = None
    
    for base_path in possible_input_paths:
        if base_path.exists():
            # Search for JSONL files recursively
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file == 'train.jsonl' and train_file is None:
                        train_file = os.path.join(root, file)
                    if file == 'dev.jsonl' and dev_file is None:
                        dev_file = os.path.join(root, file)
    
    # Set output paths (Kaggle working directory)
    MODEL_OUTPUT_PATH = "/kaggle/working/Models/deepseek_lora_1.3b"
    RESULTS_PATH = "/kaggle/working/Results"
    
    if train_file is None:
        print("\n❌ ERROR: Training data not found!")
        print("\n📌 TO FIX THIS:")
        print("   1. Run notebook 01 (data preprocessing) first")
        print("   2. Download the output files from the Output tab:")
        print("      - train.jsonl")
        print("      - dev.jsonl (optional)")
        print("   3. In THIS notebook:")
        print("      - Click '+ Add Input' button")
        print("      - Upload the JSONL files as a dataset")
        print("      - Re-run this cell")
        raise FileNotFoundError("Training data not found. See instructions above.")
    


# Create output directories
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Verify data files exist
print("\n📂 Checking data files...")
if os.path.exists(train_file):
    print(f"✅ Training data found: {train_file}")
    # Count lines
    with open(train_file) as f:
        num_train = sum(1 for _ in f)
    print(f"   📊 {num_train:,} training examples")
else:
    print(f"❌ Training data NOT found: {train_file}")
    raise FileNotFoundError(f"Training file not found at {train_file}")

if os.path.exists(dev_file):
    print(f"✅ Dev data found: {dev_file}")
    with open(dev_file) as f:
        num_dev = sum(1 for _ in f)
    print(f"   📊 {num_dev:,} dev examples")
else:
    print(f"⚠️  Dev data not found (optional): {dev_file}")
    dev_file = None

print(f"\n💾 Model will be saved to: {MODEL_OUTPUT_PATH}")
print(f"📊 Results will be saved to: {RESULTS_PATH}")
from datasets import load_dataset

# Load training data
print("📊 Loading datasets...")

dataset_files = {"train": train_file}
if dev_file:
    dataset_files["validation"] = dev_file

dataset = load_dataset("json", data_files=dataset_files)

print(f"✅ Loaded {len(dataset['train']):,} training examples")
if 'validation' in dataset:
    print(f"✅ Loaded {len(dataset['validation']):,} validation examples")

# Show sample
print("\n📝 Sample training example:")
sample = dataset['train'][0]
print(f"Messages: {len(sample['messages'])} turns")
for msg in sample['messages']:
    print(f"  - {msg['role']}: {msg['content'][:100]}...")
    
print(f"\n✅ Data loaded successfully!")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Model configuration
model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# QLoRA configuration - 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"📦 Loading {model_name} with 4-bit quantization...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

print(f"✅ Model loaded! Memory usage:")
print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# LoRA configuration
lora_config = LoraConfig(
    r=16,                          # Rank of LoRA matrices
    lora_alpha=32,                 # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
    lora_dropout=0.05,             # Dropout for LoRA layers
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n✅ LoRA configuration applied!")
print(f"   Only training {model.num_parameters() / 1e6:.2f}M parameters")

def formatting_func(example):
    """
    Format training examples as chat prompts.
    Input: {"messages": [{"role": "system/user/assistant", "content": "..."}]}
    Output: Tokenized text
    """
    # Reconstruct conversation from messages
    conversation = ""
    for msg in example['messages']:
        role = msg['role']
        content = msg['content']
        
        if role == "system":
            conversation += f"<|system|>\n{content}\n"
        elif role == "user":
            conversation += f"<|user|>\n{content}\n"
        elif role == "assistant":
            conversation += f"<|assistant|>\n{content}<|endoftext|>"
    
    # Tokenize
    return tokenizer(
        conversation,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

# Apply tokenization
print("🔄 Tokenizing datasets...")
print("   This may take a few minutes...")

tokenized_train = dataset["train"].map(
    formatting_func,
    remove_columns=dataset["train"].column_names,
    batched=False,
    desc="Tokenizing train"
)

if "validation" in dataset:
    tokenized_val = dataset["validation"].map(
        formatting_func,
        remove_columns=dataset["validation"].column_names,
        batched=False,
        desc="Tokenizing validation"
    )
else:
    tokenized_val = None

print(f"\n✅ Tokenization complete!")
print(f"   Training samples: {len(tokenized_train):,}")
if tokenized_val:
    print(f"   Validation samples: {len(tokenized_val):,}")


# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    
    # Training configuration
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    
    # Optimizer
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    
    # Checkpointing - CRITICAL for handling disconnections!
    save_strategy="steps",
    save_steps=500,                 # Save every 500 steps
    save_total_limit=3,             # Keep only 3 most recent checkpoints
    
    # Evaluation
    evaluation_strategy="steps" if tokenized_val else "no",
    eval_steps=500 if tokenized_val else None,
    
    # Logging
    logging_dir=os.path.join(MODEL_OUTPUT_PATH, "logs"),
    logging_steps=50,
    report_to="none",               # Disable wandb/tensorboard
    
    # Memory optimization
    fp16=True,                      # Mixed precision training
    gradient_checkpointing=True,    # Trade compute for memory
    
    # Other
    load_best_model_at_end=False,
    push_to_hub=False,
)

print("✅ Training configuration:")
print(f"   Output: {MODEL_OUTPUT_PATH}")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Checkpoints every: {training_args.save_steps} steps")
print(f"   Total training steps: ~{len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

# Check for existing checkpoints
if os.path.exists(MODEL_OUTPUT_PATH):
    checkpoints = [d for d in os.listdir(MODEL_OUTPUT_PATH) if d.startswith("checkpoint-")]
else:
    checkpoints = []

if checkpoints:
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = os.path.join(MODEL_OUTPUT_PATH, checkpoints[-1])
    print(f"🔄 Found existing checkpoint: {checkpoints[-1]}")
    print(f"   Will resume training from step {checkpoints[-1].split('-')[-1]}")
    resume_from_checkpoint = latest_checkpoint
else:
    print("🆕 No existing checkpoints found. Starting fresh training.")
    resume_from_checkpoint = None

from transformers import DataCollatorForLanguageModeling

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masked language modeling
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("\n🚀 Starting training...")
print("=" * 60)

# Train!
train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

print("\n" + "=" * 60)
print("✅ Training completed!")
print(f"   Final loss: {train_result.training_loss:.4f}")
print(f"   Total steps: {train_result.global_step}")
print(f"   Training time: {train_result.metrics['train_runtime'] / 3600:.2f} hours")

# Save model
final_model_path = os.path.join(os.path.dirname(MODEL_OUTPUT_PATH), "deepseek_lora_1.3b_final")
os.makedirs(final_model_path, exist_ok=True)

print(f"💾 Saving final model to: {final_model_path}")

trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Save training metrics
import json
metrics_path = os.path.join(final_model_path, "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(train_result.metrics, f, indent=2)

print("\n✅ Model saved successfully!")
print(f"   Location: {final_model_path}")
print(f"   Files: adapter_model.bin, adapter_config.json, tokenizer files")
print(f"   Size: ~200-300 MB")

if os.path.exists('/kaggle/working'):
    print("\n📥 ON KAGGLE:")
    print("   Go to Output tab to download your trained model!")
else:
    print("\n✅ ON COLAB:")
    print("   Model saved to Google Drive automatically!")
