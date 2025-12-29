"""
Seed Database from Processed CSV Data
Reads standardized CSV files from PROCESSED_DATA directory and inserts them into SQLite
"""

import sqlite3
import pandas as pd
import os
import glob
import logging

# Configuration
DB_PATH = "../data/glossary.db"
PROCESSED_DATA_DIR = "../data/PROCESSED_DATA"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_database():
    """Read processed CSVs and seed the database"""
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        logger.error(f"Processed data directory not found: {PROCESSED_DATA_DIR}")
        return

    # Find all CSV files in the processed directory
    csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {PROCESSED_DATA_DIR}")
        return

    logger.info(f"Found {len(csv_files)} files to seed.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_inserted = 0
    
    try:
        for csv_file in csv_files:
            try:
                # Read the clean CSV
                df = pd.read_csv(csv_file)
                
                # Convert DataFrame to list of tuples for insertion
                records = df.to_records(index=False).tolist()
                
                if not records:
                    continue

                # Batch insert using INSERT OR IGNORE
                # Assumes table 'glossary_terms' already exists
                cursor.executemany("""
                    INSERT OR IGNORE INTO glossary_terms 
                    (source_term, target_term, source_lang, target_lang, domain, n_gram_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, records)
                
                logger.info(f"Processed {os.path.basename(csv_file)}")

            except Exception as e:
                logger.error(f"Failed to process file {csv_file}: {e}")

        conn.commit()
        logger.info("Seeding Complete.")

    except sqlite3.Error as e:
        logger.error(f"Database error during seeding: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_database()