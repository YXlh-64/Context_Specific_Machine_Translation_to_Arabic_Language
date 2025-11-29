"""Tests for orchestrator."""

import pytest
import asyncio
from src.orchestrator import TranslationOrchestrator
from src.models import (
    RAGOutput,
    GlossaryMatch,
    FuzzyMatch,
    OrchestratorConfig,
    AgentType
)
from src.llm_client import MockLLMClient


@pytest.fixture
def sample_rag_output():
    """Create sample RAG output for testing."""
    return RAGOutput(
        glossary_matches=[
            GlossaryMatch("intensive care units", "وحدات العناية المركزة", 3),
            GlossaryMatch("severe symptoms", "الأعراض الشديدة", 2),
            GlossaryMatch("patients", "المرضى", 1),
            GlossaryMatch("admitted", "يتم إدخالهم", 1)
        ],
        fuzzy_matches=[
            FuzzyMatch(
                "Individuals with critical conditions should be transferred to ICU.",
                "يجب نقل الأفراد ذوي الحالات الحرجة إلى العناية المركزة.",
                0.94
            ),
            FuzzyMatch(
                "Severe cases require immediate hospitalization in intensive care.",
                "الحالات الشديدة تتطلب التنويم الفوري في العناية المركزة.",
                0.89
            )
        ],
        source_text="Patients with severe symptoms must be admitted to intensive care units.",
        domain="medical"
    )


@pytest.fixture
def orchestrator():
    """Create orchestrator with mock LLM client."""
    config = OrchestratorConfig()
    llm_client = MockLLMClient()
    return TranslationOrchestrator(llm_client, config)


@pytest.mark.asyncio
async def test_orchestrate_basic(orchestrator, sample_rag_output):
    """Test basic orchestration flow."""
    result = await orchestrator.orchestrate(sample_rag_output)
    
    # Check result structure
    assert result.source_text == sample_rag_output.source_text
    assert result.domain == sample_rag_output.domain
    assert len(result.translations) == 3
    assert result.recommended_variant in result.translations
    assert result.processing_time_ms > 0
    
    # Check all agent types present
    agent_types = set(result.translations.keys())
    expected_types = {
        AgentType.CONTEXT_AWARE.value,
        AgentType.TERMINOLOGY_OPTIMIZED.value,
        AgentType.CONSERVATIVE.value
    }
    assert agent_types == expected_types


@pytest.mark.asyncio
async def test_translation_variants(orchestrator, sample_rag_output):
    """Test that translation variants have required fields."""
    result = await orchestrator.orchestrate(sample_rag_output)
    
    for agent_type, variant in result.translations.items():
        # Check required fields
        assert variant.text
        assert len(variant.text) > 0
        assert 0 <= variant.quality_score <= 10
        assert 0 <= variant.glossary_compliance <= 100
        assert isinstance(variant.strengths, list)
        assert isinstance(variant.weaknesses, list)
        assert variant.agent_type == agent_type


@pytest.mark.asyncio
async def test_glossary_compliance(orchestrator, sample_rag_output):
    """Test that glossary compliance is calculated."""
    result = await orchestrator.orchestrate(sample_rag_output)
    
    for variant in result.translations.values():
        # Should have compliance score
        assert variant.glossary_compliance >= 0
        assert variant.glossary_compliance <= 100


@pytest.mark.asyncio
async def test_quality_scores(orchestrator, sample_rag_output):
    """Test that quality scores are reasonable."""
    result = await orchestrator.orchestrate(sample_rag_output)
    
    for variant in result.translations.values():
        # Quality score should be in valid range
        assert 0 <= variant.quality_score <= 10
        # With mock client, should be > 0
        assert variant.quality_score > 0


@pytest.mark.asyncio
async def test_recommendation(orchestrator, sample_rag_output):
    """Test that a variant is recommended."""
    result = await orchestrator.orchestrate(sample_rag_output)
    
    # Should recommend one variant
    assert result.recommended_variant
    assert result.recommended_variant in result.translations
    
    # Recommended variant should exist
    recommended = result.translations[result.recommended_variant]
    assert recommended is not None


@pytest.mark.asyncio
async def test_batch_processing(orchestrator, sample_rag_output):
    """Test batch processing multiple RAG outputs."""
    # Create batch of 3 items
    rag_outputs = [sample_rag_output] * 3
    
    results = await orchestrator.orchestrate_batch(rag_outputs)
    
    assert len(results) == 3
    for result in results:
        assert len(result.translations) == 3
        assert result.recommended_variant


@pytest.mark.asyncio
async def test_empty_glossary(orchestrator):
    """Test handling of empty glossary."""
    rag_output = RAGOutput(
        glossary_matches=[],
        fuzzy_matches=[
            FuzzyMatch("Test sentence.", "جملة اختبار.", 0.90)
        ],
        source_text="This is a test sentence.",
        domain="general"
    )
    
    result = await orchestrator.orchestrate(rag_output)
    
    # Should still work with empty glossary
    assert len(result.translations) == 3
    # Glossary compliance should be 100% (no terms to check)
    for variant in result.translations.values():
        assert variant.glossary_compliance == 100.0


@pytest.mark.asyncio
async def test_processing_time(orchestrator, sample_rag_output):
    """Test that processing time is recorded."""
    result = await orchestrator.orchestrate(sample_rag_output)
    
    # Processing time should be positive
    assert result.processing_time_ms > 0
    # Should be reasonable (less than 10 seconds for mock)
    assert result.processing_time_ms < 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
