"""
Extract Test Samples from Economic V1 CSV

Creates a smaller test dataset for evaluation by randomly sampling
from the full economic v1.csv dataset.
"""

import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract test samples from economic v1.csv")
    parser.add_argument(
        "--input",
        type=str,
        default="./economic v1.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test_samples.csv",
        help="Path to output test CSV file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to extract"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("📊 Extracting Test Samples")
    print("="*60)
    
    # Load data
    print(f"\n📁 Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Total rows: {len(df):,}")
    
    # Validate columns
    if 'en' not in df.columns or 'ar' not in df.columns:
        print("❌ Error: CSV must have 'en' and 'ar' columns")
        return
    
    # Remove empty rows
    df = df.dropna(subset=['en', 'ar'])
    df = df[df['en'].str.strip() != '']
    df = df[df['ar'].str.strip() != '']
    print(f"   Valid rows: {len(df):,}")
    
    # Sample
    if args.num_samples > len(df):
        print(f"⚠️  Warning: Requested {args.num_samples} samples but only {len(df)} available")
        args.num_samples = len(df)
    
    test_df = df.sample(n=args.num_samples, random_state=args.seed)
    
    # Save
    test_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Saved {len(test_df)} samples to: {args.output}")
    print(f"   Random seed: {args.seed}")
    
    # Show samples
    print("\n📋 Sample Preview (first 3):")
    print("="*60)
    for i, row in test_df.head(3).iterrows():
        print(f"\n{i+1}.")
        print(f"EN: {row['en'][:80]}...")
        print(f"AR: {row['ar'][:80]}...")


if __name__ == "__main__":
    main()
