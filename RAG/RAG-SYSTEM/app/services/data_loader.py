"""
Data Loader Module
Load and clean translation data from CSV files
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_translation_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load all domain CSVs and combine into single DataFrame
    
    Expected CSV formats:
    - Format 1: id, english, arabic (healthcare.csv)
    - Format 2: id, arabic, english (technology.csv)
    - Format 3: source, target columns
    
    Args:
        data_dir: Directory containing translation CSVs
        
    Returns:
        Combined and cleaned DataFrame
    """
    if data_dir is None:
        data_dir = str(settings.TRANSLATION_DATA_DIR)
    
    data_path = Path(data_dir)
    all_data = []
    
    logger.info(f"Loading translation data from: {data_path}")
    
    # Define CSV files and their mappings
    csv_mappings = [
        # (file_path, source_col, target_col, domain, lang_pair)
        (data_path / "healthcare.csv", "english", "arabic", "health", "en-ar")
    ]
    
    # Also check pipeline folder
    # pipeline_path = data_path / "pipeline"
    # if pipeline_path.exists():
    #     csv_mappings.append(
    #         (pipeline_path / "culture_media_translation_examples.csv", "english", "arabic", "history", "en-ar")
    #     )
    
    for csv_info in csv_mappings:
        file_path = csv_info[0]
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        try:
            df = load_single_csv(file_path, csv_info)
            if df is not None and len(df) > 0:
                all_data.append(df)
                logger.info(f"Loaded {len(df)} pairs from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        logger.warning("No translation data loaded!")
        return pd.DataFrame(columns=['source', 'target', 'domain', 'language_pair', 'source_lang', 'target_lang'])
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Clean the data
    combined_df = clean_translation_data(combined_df)
    
    # Statistics
    logger.info(f"\nTotal translation pairs: {len(combined_df)}")
    logger.info(f"By domain:\n{combined_df['domain'].value_counts().to_string()}")
    logger.info(f"By language pair:\n{combined_df['language_pair'].value_counts().to_string()}")
    
    return combined_df


def load_single_csv(file_path: Path, csv_info: tuple) -> Optional[pd.DataFrame]:
    """Load a single CSV file with proper column mapping"""
    
    _, source_col, target_col, domain, lang_pair = csv_info
    
    # Parse language pair
    source_lang, target_lang = lang_pair.split("-")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Handle special formats
    if source_col is None:
        # Special format like medical.csv with query/field structure
        df = parse_special_format(df, domain, lang_pair)
        if df is not None:
            return df
        return None
    
    # Check if columns exist (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()
    
    # Try to find source and target columns
    source_col_lower = source_col.lower()
    target_col_lower = target_col.lower()
    
    if source_col_lower not in df.columns or target_col_lower not in df.columns:
        # Try alternative column names
        alt_mappings = {
            'english': ['english', 'eng', 'en', 'source', 'src'],
            'arabic': ['arabic', 'ara', 'ar', 'target', 'tgt'],
            'french': ['french', 'fra', 'fr']
        }
        
        source_found = None
        target_found = None
        
        for col in df.columns:
            if any(alt in col for alt in alt_mappings.get(source_col_lower, [source_col_lower])):
                source_found = col
            if any(alt in col for alt in alt_mappings.get(target_col_lower, [target_col_lower])):
                target_found = col
        
        if source_found and target_found:
            source_col_lower = source_found
            target_col_lower = target_found
        else:
            logger.warning(f"Could not find columns in {file_path.name}: {df.columns.tolist()}")
            return None
    
    # Create standardized DataFrame
    result_df = pd.DataFrame({
        'source': df[source_col_lower],
        'target': df[target_col_lower],
        'domain': domain,
        'language_pair': lang_pair,
        'source_lang': source_lang,
        'target_lang': target_lang
    })
    
    return result_df


def parse_special_format(df: pd.DataFrame, domain: str, lang_pair: str) -> Optional[pd.DataFrame]:
    """Parse special CSV formats like medical.csv with embedded examples"""
    
    source_lang, target_lang = lang_pair.split("-")
    pairs = []
    
    df.columns = df.columns.str.lower().str.strip()
    
    if 'query' in df.columns:
        for _, row in df.iterrows():
            text = str(row.get('query', ''))
            
            # Extract English/Arabic pairs from text
            lines = text.split('\n')
            english_text = None
            arabic_text = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('English:'):
                    english_text = line.replace('English:', '').strip()
                elif line.startswith('Arabic:'):
                    arabic_text = line.replace('Arabic:', '').strip()
                    
                    # Save pair when we have both
                    if english_text and arabic_text:
                        pairs.append({
                            'source': english_text,
                            'target': arabic_text,
                            'domain': domain,
                            'language_pair': lang_pair,
                            'source_lang': source_lang,
                            'target_lang': target_lang
                        })
                        english_text = None
                        arabic_text = None
    
    if pairs:
        return pd.DataFrame(pairs)
    return None


def clean_translation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate translation data"""
    
    # Remove null values
    df = df.dropna(subset=['source', 'target'])
    
    # Convert to string
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    
    # Trim whitespace
    df['source'] = df['source'].str.strip()
    df['target'] = df['target'].str.strip()
    
    # Remove empty strings
    df = df[df['source'].str.len() > 0]
    df = df[df['target'].str.len() > 0]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['source', 'target'])
    
    # Remove very short sentences (< 3 words)
    df['source_word_count'] = df['source'].str.split().str.len()
    df['target_word_count'] = df['target'].str.split().str.len()
    
    df = df[df['source_word_count'] >= settings.MIN_SENTENCE_LENGTH]
    df = df[df['target_word_count'] >= settings.MIN_SENTENCE_LENGTH]
    
    # Calculate lengths for metadata
    df['source_length'] = df['source_word_count']
    df['target_length'] = df['target_word_count']
    
    # Drop helper columns
    df = df.drop(columns=['source_word_count', 'target_word_count'])
    
    # Assign unique IDs
    df['id'] = range(1, len(df) + 1)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def get_domain_statistics(df: pd.DataFrame) -> dict:
    """Get statistics about loaded data by domain"""
    stats = {}
    
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        stats[domain] = {
            'count': len(domain_df),
            'avg_source_length': domain_df['source_length'].mean(),
            'avg_target_length': domain_df['target_length'].mean(),
            'language_pairs': domain_df['language_pair'].unique().tolist()
        }
    
    return stats


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    df = load_translation_data()
    print(f"\nLoaded {len(df)} translation pairs")
    print(f"\nSample data:")
    print(df.head())
    
    stats = get_domain_statistics(df)
    print(f"\nDomain statistics:")
    for domain, stat in stats.items():
        print(f"  {domain}: {stat['count']} pairs")
