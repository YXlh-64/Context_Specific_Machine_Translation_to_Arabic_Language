"""
Data Preparation Script for QWEN Fine-tuning
Loads domain-specific English-Arabic parallel corpora and prepares them for training
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import random

class DataPreparator:
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize data preparator
        
        Args:
            data_dir: Path to the Data/english-arabic directory
            output_dir: Path to save prepared datasets
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.domains = {
            'technology': 'technology.csv',
            'economic': 'economic v1.csv',
            'education': 'Education_Research.csv'
        }
        
    def load_domain_data(self, domain: str) -> pd.DataFrame:
        """Load data for a specific domain"""
        file_path = self.data_dir / self.domains[domain]
        print(f"Loading {domain} data from {file_path}...")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"Loaded {len(df)} examples for {domain} domain")
            return df
        except Exception as e:
            print(f"Error loading {domain} data: {e}")
            return None
    
    def format_for_qwen(self, row: pd.Series, domain: str) -> Dict:
        """
        Format a single example for QWEN fine-tuning
        Using instruction-following format with domain context
        """
        english_text = str(row['english']).strip()
        arabic_text = str(row['arabic']).strip()
        
        # Create instruction-based format for translation
        instruction = f"Translate the following English text to Arabic. Domain: {domain}"
        
        # QWEN chat format
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert translator specializing in {domain} domain translations between English and Arabic."
                },
                {
                    "role": "user",
                    "content": f"{instruction}\n\nText: {english_text}"
                },
                {
                    "role": "assistant",
                    "content": arabic_text
                }
            ]
        }
        
        return conversation
    
    def prepare_domain_dataset(self, domain: str, test_size: float = 0.1, val_size: float = 0.1, max_samples: int = None):
        """
        Prepare train/val/test splits for a specific domain
        
        Args:
            domain: Domain name (technology, economic, education)
            test_size: Proportion for test set
            val_size: Proportion for validation set
            max_samples: Maximum number of samples to use (for testing)
        """
        print(f"\n{'='*60}")
        print(f"Preparing {domain.upper()} domain dataset")
        print(f"{'='*60}")
        
        df = self.load_domain_data(domain)
        if df is None or len(df) == 0:
            print(f"No data available for {domain}")
            return
        
        # Remove any rows with missing values
        df = df.dropna(subset=['english', 'arabic'])
        
        # Limit samples if specified (useful for quick testing)
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            print(f"Limited to {max_samples} samples for testing")
        
        # Split into train, validation, and test
        train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=(test_size / (test_size + val_size)), random_state=42)
        
        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Format data for QWEN
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            formatted_data = []
            for _, row in split_df.iterrows():
                formatted_data.append(self.format_for_qwen(row, domain))
            
            # Save as JSONL
            output_file = self.output_dir / f"{domain}_{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in formatted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(formatted_data)} examples to {output_file}")
        
        # Save statistics
        stats = {
            'domain': domain,
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'avg_english_length': df['english'].str.len().mean(),
            'avg_arabic_length': df['arabic'].str.len().mean()
        }
        
        stats_file = self.output_dir / f"{domain}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Saved statistics to {stats_file}")
    
    def prepare_all_domains(self, test_size: float = 0.1, val_size: float = 0.1, max_samples: int = None):
        """Prepare datasets for all domains"""
        print(f"\n{'#'*60}")
        print("Starting data preparation for all domains")
        print(f"{'#'*60}")
        
        for domain in self.domains.keys():
            try:
                self.prepare_domain_dataset(domain, test_size, val_size, max_samples)
            except Exception as e:
                print(f"Error preparing {domain} domain: {e}")
                continue
        
        print(f"\n{'#'*60}")
        print("Data preparation complete!")
        print(f"{'#'*60}")
        print(f"Output directory: {self.output_dir}")

def main():
    """Main function to run data preparation"""
    # Set paths relative to the script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../../Data/english-arabic"
    output_dir = script_dir / "data/prepared"
    
    # Initialize preparator
    preparator = DataPreparator(data_dir=str(data_dir), output_dir=str(output_dir))
    
    # Prepare all domains
    # Set max_samples=1000 for quick testing, or None for full dataset
    preparator.prepare_all_domains(test_size=0.1, val_size=0.1, max_samples=None)

if __name__ == "__main__":
    main()
