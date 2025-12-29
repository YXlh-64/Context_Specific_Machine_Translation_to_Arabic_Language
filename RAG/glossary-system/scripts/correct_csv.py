"""
Convert Raw CSV Data to Standardized CSV Format
Restructures CSV files from RAW DATA folder and saves cleaned versions to PROCESSED DATA
"""

import pandas as pd
import os
import glob
from pathlib import Path
import logging

# Configuration
RAW_DATA_DIR = "../data/RAW DATA"
PROCESSED_DATA_DIR = "../data/PROCESSED_DATA"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_language_pair(folder_name):
    """Parse language pair from folder name (e.g., 'en-ar' -> 'en', 'ar')"""
    if '-' in folder_name:
        source, target = folder_name.split('-')
        return source, target
    return 'en', 'ar'  # default

def parse_domain_from_filename(filename):
    """Extract domain from filename (e.g., 'economic_english_arabic.csv' -> 'economic')"""
    name = Path(filename).stem.lower()
    
    # Map common variations to standard domains
    domain_mapping = {
        'finance': 'finance',
        'economic': 'economy',
        'media': 'media',
        'legal': 'legal',
        'tech': 'technology',
        'politics': 'politics'
    }

    # Simple keyword matching if exact match fails
    for key, value in domain_mapping.items():
        if key in name:
            return value
            
    return name

def process_csv_file(csv_path, source_lang, target_lang, domain):
    """Process a single CSV file and return a DataFrame in the correct format"""
    try:
        # Read CSV
        df = pd.read_csv(csv_path, encoding='utf-8')

        # Handle different column name variations
        source_col = None
        target_col = None

        possible_source_cols = ['English', 'english', 'French', 'french', 'source', 'Source', 'term_source']
        possible_target_cols = ['Arabic', 'arabic', 'target', 'Target', 'term_target']

        for col in df.columns:
            if col in possible_source_cols:
                source_col = col
            elif col in possible_target_cols:
                target_col = col

        if not source_col or not target_col:
            logger.warning(f"Could not identify columns in {csv_path}. Columns: {list(df.columns)}")
            return pd.DataFrame()

        # Rename columns to standard format
        df = df.rename(columns={source_col: 'source_term', target_col: 'target_term'})
        
        # Keep only necessary columns
        clean_df = df[['source_term', 'target_term']].copy()

        # Clean data: drop NaNs and empty strings
        clean_df.dropna(subset=['source_term', 'target_term'], inplace=True)
        clean_df['source_term'] = clean_df['source_term'].astype(str).str.strip()
        clean_df['target_term'] = clean_df['target_term'].astype(str).str.strip()
        clean_df = clean_df[clean_df['source_term'] != 'nan']
        clean_df = clean_df[clean_df['target_term'] != 'nan']

        # Add metadata columns
        clean_df['source_lang'] = source_lang
        clean_df['target_lang'] = target_lang
        clean_df['domain'] = domain
        
        # Calculate n-gram size
        clean_df['n_gram_size'] = clean_df['source_term'].apply(lambda x: len(x.split()))

        logger.info(f"Processed {len(clean_df)} terms from {csv_path}")
        return clean_df

    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        return pd.DataFrame()

def generate_standardized_csvs():
    """Main function to process raw files and output clean CSVs"""

    # Ensure output directory exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        logger.info(f"Created output directory: {PROCESSED_DATA_DIR}")

    # Find all CSV files in RAW DATA directory
    csv_pattern = os.path.join(RAW_DATA_DIR, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    if not csv_files:
        logger.warning(f"No CSV files found in {RAW_DATA_DIR}")
        return

    logger.info(f"Found {len(csv_files)} CSV files to process")

    processed_count = 0

    # Process each CSV file
    for csv_file in csv_files:
        csv_path = Path(csv_file)

        # Parse metadata
        parent_folder = csv_path.parent.name
        source_lang, target_lang = parse_language_pair(parent_folder)
        domain = parse_domain_from_filename(csv_path.name)

        logger.info(f"Processing {csv_file.split('/')[-1]} -> Domain: {domain}")

        # Get cleaned DataFrame
        df = process_csv_file(csv_file, source_lang, target_lang, domain)

        if not df.empty:
            # Create a unique filename based on domain and original name
            output_filename = f"{domain}_{source_lang}_{target_lang}_{csv_path.stem}_cleaned.csv"
            output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
            
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Saved: {output_path}")
            processed_count += 1

    logger.info("-" * 30)
    logger.info(f"Processing Complete.")
    logger.info(f"Total files processed and saved: {processed_count}")
    logger.info("-" * 30)

if __name__ == "__main__":
    generate_standardized_csvs()