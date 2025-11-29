"""Tests for CSV parser."""

import pytest
from src.parser import (
    parse_glossary_terms,
    parse_fuzzy_matches,
    parse_query_block,
    parse_rag_string,
    validate_csv_format
)
from src.models import GlossaryMatch, FuzzyMatch, RAGOutput


def test_parse_glossary_terms():
    """Test parsing glossary terms from Terms: format."""
    terms_line = "Terms: intensive care=العناية المركزة - patients=المرضى - severe=شديد"
    
    matches = parse_glossary_terms(terms_line)
    
    assert len(matches) == 3
    assert matches[0].source_term == "intensive care"
    assert matches[0].target_term == "العناية المركزة"
    assert matches[0].ngram_size == 2
    
    assert matches[1].source_term == "patients"
    assert matches[1].target_term == "المرضى"
    assert matches[1].ngram_size == 1


def test_parse_glossary_terms_empty():
    """Test parsing empty glossary."""
    terms_line = "Terms: "
    matches = parse_glossary_terms(terms_line)
    assert len(matches) == 0


def test_parse_fuzzy_matches():
    """Test parsing English-Arabic pairs."""
    text_block = """English: Hello world.
Arabic: مرحبا بالعالم.

English: This is a test.
Arabic: هذا اختبار."""
    
    matches = parse_fuzzy_matches(text_block)
    
    assert len(matches) == 2
    assert matches[0] == ("Hello world.", "مرحبا بالعالم.")
    assert matches[1] == ("This is a test.", "هذا اختبار.")


def test_parse_query_block():
    """Test parsing complete query block."""
    query = """Terms: patients=المرضى - hospital=المستشفى

English: Patients go to hospital.
Arabic: المرضى يذهبون إلى المستشفى.

English: The hospital treats patients.
Arabic: المستشفى يعالج المرضى.

Terms: patients=المرضى - hospital=المستشفى
English: All patients must register at the hospital.
Arabic:"""
    
    rag_output = parse_query_block(query, "medical")
    
    assert rag_output.source_text == "All patients must register at the hospital."
    assert rag_output.domain == "medical"
    assert len(rag_output.glossary_matches) == 2
    assert len(rag_output.fuzzy_matches) == 2
    
    # Check glossary
    assert any(m.source_term == "patients" for m in rag_output.glossary_matches)
    assert any(m.source_term == "hospital" for m in rag_output.glossary_matches)
    
    # Check fuzzy matches have similarity scores
    assert all(0 <= m.similarity_score <= 1.0 for m in rag_output.fuzzy_matches)


def test_parse_rag_string():
    """Test parsing RAG string directly."""
    query_string = """Terms: test=اختبار

English: This is a test.
Arabic: هذا اختبار.

Terms: test=اختبار
English: Test sentence.
Arabic:"""
    
    rag_output = parse_rag_string(query_string, "technical")
    
    assert rag_output.source_text == "Test sentence."
    assert rag_output.domain == "technical"
    assert len(rag_output.glossary_matches) == 1


def test_parse_glossary_deduplication():
    """Test that duplicate glossary terms are deduplicated."""
    query = """Terms: test=اختبار - test=اختبار

English: Test.
Arabic:"""
    
    rag_output = parse_query_block(query, "test")
    
    # Should deduplicate
    assert len(rag_output.glossary_matches) == 1


def test_invalid_query_no_source():
    """Test that missing source text raises error."""
    query = """Terms: test=اختبار

English: Example.
Arabic: مثال."""
    
    with pytest.raises(Exception):
        parse_query_block(query, "test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
