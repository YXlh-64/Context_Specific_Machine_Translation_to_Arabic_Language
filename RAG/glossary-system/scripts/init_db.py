import sqlite3
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import init_database_schema

DB_PATH = "data/glossary.db"

def init_db():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    init_database_schema(DB_PATH)
    print("Database initialized with FTS5 support.")

if __name__ == "__main__":
    init_db()