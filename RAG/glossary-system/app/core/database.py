import sqlite3
import os
import logging
from contextlib import contextmanager
from app.core.config import settings

logger = logging.getLogger(__name__)

def init_database_schema(db_path: str = None):
    """
    Initialize the database schema with FTS5 support.
    
    Args:
        db_path: Path to the database file. If None, uses settings.DATABASE_URL
    """
    if db_path is None:
        db_path = settings.DATABASE_URL
        if db_path.startswith("file:"):
            db_path = db_path.split("?")[0].replace("file:", "")
    
    # Remove existing database to recreate with FTS5
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create Content Table (stores all glossary data)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS glossary_terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_term TEXT NOT NULL,
        target_term TEXT NOT NULL,
        source_lang TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        domain TEXT NOT NULL,
        n_gram_size INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create Index for domain/language filtering
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_lookup 
    ON glossary_terms (domain, source_lang, target_lang);
    """)
    
    # Create FTS5 Virtual Table for fast text search on source_term
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS glossary_fts USING fts5(
        source_term,
        content='glossary_terms',
        content_rowid='id'
    );
    """)
    
    # Create FTS5 Virtual Table for fast text search on target_term (for bidirectional lookup)
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS glossary_fts_target USING fts5(
        target_term,
        content='glossary_terms',
        content_rowid='id'
    );
    """)
    
    # Create triggers to keep FTS5 table in sync with content table
    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS glossary_ai AFTER INSERT ON glossary_terms BEGIN
        INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
        INSERT INTO glossary_fts_target(rowid, target_term) VALUES (new.id, new.target_term);
    END;
    """)
    
    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS glossary_ad AFTER DELETE ON glossary_terms BEGIN
        INSERT INTO glossary_fts(glossary_fts, rowid, source_term) VALUES('delete', old.id, old.source_term);
        INSERT INTO glossary_fts_target(glossary_fts_target, rowid, target_term) VALUES('delete', old.id, old.target_term);
    END;
    """)
    
    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS glossary_au AFTER UPDATE ON glossary_terms BEGIN
        INSERT INTO glossary_fts(glossary_fts, rowid, source_term) VALUES('delete', old.id, old.source_term);
        INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
        INSERT INTO glossary_fts_target(glossary_fts_target, rowid, target_term) VALUES('delete', old.id, old.target_term);
        INSERT INTO glossary_fts_target(rowid, target_term) VALUES (new.id, new.target_term);
    END;
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database schema initialized with FTS5 support.")

@contextmanager
def get_db_connection():
    """
    Establishes a connection to SQLite with optimization flags.
    Implements the 'SQLite Database Connection' checklist items.
    """
    conn = None
    try:
        # Extract file path from URI if present
        db_path = settings.DATABASE_URL
        if db_path.startswith("file:"):
            # Handle file URI format: file:path?mode=ro
            db_path = db_path.split("?")[0].replace("file:", "")
        
        # Check if database file exists (for production safety)
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # uri=True allows mode=ro
        conn = sqlite3.connect(settings.DATABASE_URL, uri=True, timeout=30)
        
        # Enable Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        raise
    finally:
        if conn:
            conn.close()
