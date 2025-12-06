"""
Batch training script to train all domain-specific models sequentially
"""

import subprocess
import sys
from pathlib import Path

def train_domain(domain: str, epochs: int = 3, batch_size: int = 4):
    """Train a model for a specific domain"""
    print(f"\n{'='*60}")
    print(f"Training {domain.upper()} domain model")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "train_qwen.py",
        "--domain", domain,
        "--data_dir", "data/prepared",
        "--output_dir", f"models/{domain}_model",
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", str(batch_size),
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-4",
        "--warmup_ratio", "0.03",
        "--logging_steps", "10",
        "--save_steps", "100",
        "--eval_steps", "100",
        "--use_4bit",
        "--use_lora"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error training {domain} model")
        return False
    
    print(f"\n{domain.capitalize()} model training complete!")
    return True

def main():
    domains = ["technology", "economic", "education"]
    
    print("="*60)
    print("QWEN Domain-Specific Fine-tuning Pipeline")
    print("="*60)
    print("\nThis script will train models for all domains sequentially.")
    print("Make sure you have prepared the data first using prepare_data.py\n")
    
    input("Press Enter to continue...")
    
    for domain in domains:
        success = train_domain(domain, epochs=3, batch_size=4)
        if not success:
            print(f"\nStopping pipeline due to error in {domain} training")
            break
    
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Evaluate models: python evaluate.py --model_path models/technology_model --domain technology")
    print("2. Launch Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    main()
