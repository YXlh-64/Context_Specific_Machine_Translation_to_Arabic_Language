"""CSV parser for RAG team output format."""

import re
import logging
from typing import List, Optional, Tuple
import pandas as pd

from .models import GlossaryMatch, FuzzyMatch, RAGOutput

logger = logging.getLogger(__name__)


class RAGParsingError(Exception):
    """Custom exception for RAG parsing errors."""
    pass


def parse_glossary_terms(terms_line: str) -> List[GlossaryMatch]:
    """
    Parse glossary terms from "Terms: X=Y - A=B" format.
    
    Args:
        terms_line: String like "Terms: intensive care=العناية المركزة - patients=المرضى"
    
    Returns:
        List of GlossaryMatch objects
    """
    glossary_matches = []
    
    # Remove "Terms: " prefix
    if not terms_line.startswith("Terms:"):
        logger.warning(f"Terms line doesn't start with 'Terms:': {terms_line[:50]}")
        return glossary_matches
    
    terms_content = terms_line[6:].strip()
    
    # Split by " - " to get individual term pairs
    term_pairs = terms_content.split(" - ")
    
    for pair in term_pairs:
        pair = pair.strip()
        if not pair:
            continue
        
        # Split by "=" to get source and target
        if "=" not in pair:
            logger.warning(f"Invalid term pair format (missing '='): {pair}")
            continue
        
        parts = pair.split("=", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid term pair format: {pair}")
            continue
        
        source_term = parts[0].strip()
        target_term = parts[1].strip()
        
        if source_term and target_term:
            ngram_size = len(source_term.split())
            glossary_matches.append(GlossaryMatch(
                source_term=source_term,
                target_term=target_term,
                ngram_size=ngram_size
            ))
    
    return glossary_matches


def parse_fuzzy_matches(text_block: str) -> List[Tuple[str, str]]:
    """
    Parse fuzzy matches from "English: ... Arabic: ..." format.
    
    Args:
        text_block: Block of text containing English-Arabic pairs
    
    Returns:
        List of (english_text, arabic_text) tuples
    """
    matches = []
    
    # Split into lines
    lines = text_block.split("\n")
    
    english_text = None
    for line in lines:
        line = line.strip()
        
        if line.startswith("English:"):
            english_text = line[8:].strip()
        elif line.startswith("Arabic:"):
            arabic_text = line[7:].strip()
            if english_text and arabic_text:
                matches.append((english_text, arabic_text))
            english_text = None
    
    return matches


def calculate_similarity_score(match_index: int, total_matches: int) -> float:
    """
    Calculate similarity score based on position (earlier = more similar).
    
    In real implementation, this would use actual similarity metrics.
    For now, we assign scores based on position.
    
    Args:
        match_index: Index of the match (0-based)
        total_matches: Total number of matches
    
    Returns:
        Similarity score between 0.7 and 0.95
    """
    if total_matches <= 1:
        return 0.90
    
    # Earlier matches get higher scores (0.95 to 0.70)
    score = 0.95 - (match_index / (total_matches - 1)) * 0.25
    return round(score, 2)


def parse_rag_csv(csv_path: str) -> List[RAGOutput]:
    """
    Parse RAG team's CSV into structured format.
    
    Expected CSV format:
    "query","field"
    "Terms: X=Y - A=B
    
    English: Example 1.
    Arabic: Translation 1.
    
    English: Source text.
    Arabic:","domain"
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        List of RAGOutput objects
    
    Raises:
        RAGParsingError: If CSV format is invalid
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RAGParsingError(f"Failed to read CSV file: {e}")
    
    if "query" not in df.columns or "field" not in df.columns:
        raise RAGParsingError("CSV must have 'query' and 'field' columns")
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            query = str(row["query"])
            domain = str(row["field"])
            
            # Parse the query block
            rag_output = parse_query_block(query, domain)
            results.append(rag_output)
            
        except Exception as e:
            logger.error(f"Error parsing row {idx}: {e}")
            continue
    
    logger.info(f"Successfully parsed {len(results)} RAG outputs from {csv_path}")
    return results


def parse_query_block(query: str, domain: str) -> RAGOutput:
    """
    Parse a single query block into RAGOutput.
    
    Args:
        query: The query string containing terms and examples
        domain: The domain/field
    
    Returns:
        RAGOutput object
    
    Raises:
        RAGParsingError: If query format is invalid
    """
    lines = query.split("\n")
    
    # Collect all glossary terms
    all_glossary_matches = []
    
    # Collect English-Arabic pairs
    fuzzy_match_pairs = []
    
    # Track source text
    source_text = None
    last_english = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if not line:
            continue
        
        # Check for glossary terms
        if line.startswith("Terms:"):
            glossary_matches = parse_glossary_terms(line)
            all_glossary_matches.extend(glossary_matches)
        
        # Check for English text
        elif line.startswith("English:"):
            last_english = line[8:].strip()
        
        # Check for Arabic text
        elif line.startswith("Arabic:"):
            arabic_text = line[7:].strip()
            
            if last_english:
                if arabic_text:
                    # This is a fuzzy match example
                    fuzzy_match_pairs.append((last_english, arabic_text))
                else:
                    # This is the source text to translate (Arabic is empty)
                    source_text = last_english
                
                last_english = None
    
    # Validate we found source text
    if not source_text:
        raise RAGParsingError("Could not find source text (English line before empty Arabic)")
    
    # Deduplicate glossary matches
    seen_sources = set()
    unique_glossary = []
    for match in all_glossary_matches:
        if match.source_term not in seen_sources:
            seen_sources.add(match.source_term)
            unique_glossary.append(match)
    
    # Create FuzzyMatch objects with similarity scores
    fuzzy_matches = []
    for idx, (eng, ara) in enumerate(fuzzy_match_pairs):
        similarity = calculate_similarity_score(idx, len(fuzzy_match_pairs))
        fuzzy_matches.append(FuzzyMatch(
            source_text=eng,
            target_text=ara,
            similarity_score=similarity
        ))
    
    return RAGOutput(
        glossary_matches=unique_glossary,
        fuzzy_matches=fuzzy_matches,
        source_text=source_text,
        domain=domain
    )


def parse_rag_string(query_string: str, domain: str) -> RAGOutput:
    """
    Parse a RAG query string directly (without CSV file).
    
    Useful for testing or when data is already loaded.
    
    Args:
        query_string: The query string
        domain: The domain/field
    
    Returns:
        RAGOutput object
    """
    return parse_query_block(query_string, domain)


def validate_csv_format(csv_path: str) -> Tuple[bool, List[str]]:
    """
    Validate CSV format without fully parsing.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]
    
    # Check columns
    if "query" not in df.columns:
        issues.append("Missing 'query' column")
    if "field" not in df.columns:
        issues.append("Missing 'field' column")
    
    if issues:
        return False, issues
    
    # Check each row
    for idx, row in df.iterrows():
        query = str(row["query"])
        
        if "English:" not in query:
            issues.append(f"Row {idx}: Missing 'English:' markers")
        if "Arabic:" not in query:
            issues.append(f"Row {idx}: Missing 'Arabic:' markers")
        if "Terms:" not in query:
            issues.append(f"Row {idx}: Missing 'Terms:' section (warning)")
    
    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    # Test parsing
    logging.basicConfig(level=logging.INFO)
    
    # Example query string
    example_query = """Terms: intensive care units=وحدات العناية المركزة - severe symptoms=الأعراض الشديدة - patients=المرضى - admitted=يتم إدخالهم

English: Individuals with critical conditions should be transferred to ICU.
Arabic: يجب نقل الأفراد ذوي الحالات الحرجة إلى العناية المركزة.

English: Severe cases require immediate hospitalization in intensive care.
Arabic: الحالات الشديدة تتطلب التنويم الفوري في العناية المركزة.

Terms: intensive care units=وحدات العناية المركزة - severe symptoms=الأعراض الشديدة - patients=المرضى - admitted=يتم إدخالهم
English: Patients with severe symptoms must be admitted to intensive care units.
Arabic:"""
    
    result = parse_rag_string(example_query, "medical")
    
    print(f"Source: {result.source_text}")
    print(f"Domain: {result.domain}")
    print(f"Glossary matches: {len(result.glossary_matches)}")
    for match in result.glossary_matches:
        print(f"  {match.source_term} → {match.target_term}")
    print(f"Fuzzy matches: {len(result.fuzzy_matches)}")
    for match in result.fuzzy_matches:
        print(f"  [{match.similarity_score}] {match.source_text[:50]}...")
