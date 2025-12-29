"""
Full Retrieval Comparison Test with Markdown Export
"""

import sys
import os
import datetime
from qdrant_client import QdrantClient

# Adjust path to find your app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.optimized_retrieval_service_test import OptimizedRetriever
from app.core.config import settings

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "translation_memory"
QUERY = "What steps should be taken if I am pregnant before an x-ray?"
OUTPUT_FILE = "search_results_new4.md"

class ReportGenerator:
    def __init__(self):
        self.content = f"# 🚀 Retrieval System Report\n"
        self.content += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        self.content += f"**Query:** `{QUERY}`\n\n"
        self.content += "---\n"

    def add_section(self, title, results):
        """Formats a list of results into Markdown and adds to report."""
        print(f"\n{title}...") # Print to console to show progress
        
        self.content += f"\n## {title}\n"
        
        if not results:
            self.content += "> ❌ No results found.\n"
            return

        self.content += f"**Found {len(results)} results:**\n\n"

        for i, res in enumerate(results, 1):
            # Extract metadata
            meta = res.get('metadata', {}) or {}
            is_test = meta.get('test_category') == 'diabetes_rrf_test'
            
            # Format Score
            score = res.get('score', 0.0)
            
            # Hybrid specific info
            extra_info = ""
            if 'semantic_rank' in meta:
                sem = meta.get('semantic_rank', 'N/A')
                word = meta.get('wording_rank', 'N/A')
                extra_info = f" | **Ranks:** Sem `#{sem}` / Word `#{word}`"

            # Marker
            marker = "🌟 **[TEST DATA]**" if is_test else "📄 [EXISTING]"
            
            # Build the Markdown Block
            self.content += f"### {i}. {marker}\n"
            self.content += f"- **Score:** `{score:.4f}` {extra_info}\n"
            self.content += f"- **Source:** {res.get('source', '')}\n"
            self.content += f"- **Target:** {res.get('target', '')}\n"
            self.content += "\n---\n"

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.content)
        print(f"\n✅ Report successfully saved to: {os.path.abspath(filename)}")


def test_full_pipeline():
    print(f"🚀 Initializing Retriever for query: '{QUERY}'...")
    
    client = QdrantClient(url=QDRANT_URL)
    retriever = OptimizedRetriever(client=client, collection_name=COLLECTION_NAME)
    report = ReportGenerator()
    
    # ---------------------------------------------------------
    # 1. Semantic Search Only
    # ---------------------------------------------------------
    semantic_results = retriever.search_semantic(
        query=QUERY,
        top_k=5,
        domain="health",
        use_cache=False,
        min_score=0.0
    )
    report.add_section("🔍 1. Pure Semantic Search", semantic_results)

    # ---------------------------------------------------------
    # 2. Wording Search Only
    # ---------------------------------------------------------
    # Access internal method for testing purposes
    wording_results = retriever._search_wording(
        query=QUERY,
        top_k=5,
        domain="health",
        source_lang=None,
        target_lang=None
    )
    report.add_section("🔍 2. Pure Wording Search", wording_results)

    # ---------------------------------------------------------
    # 3. Hybrid Search (RRF)
    # ---------------------------------------------------------
    hybrid_results = retriever.search_hybrid(
        query=QUERY,
        top_k=5,
        domain="health",
        use_cache=False
    )
    report.add_section("🔍 3. Hybrid RRF Search", hybrid_results)

    # ---------------------------------------------------------
    # Save File
    # ---------------------------------------------------------
    report.save(OUTPUT_FILE)

if __name__ == "__main__":
    test_full_pipeline()