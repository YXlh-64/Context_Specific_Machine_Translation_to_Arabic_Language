"""Tests for prompt builder."""

import pytest
from src.prompt_builder import PromptBuilder
from src.models import GlossaryMatch, FuzzyMatch, AgentType


@pytest.fixture
def sample_glossary():
    """Sample glossary matches."""
    return [
        GlossaryMatch("intensive care", "العناية المركزة", 2),
        GlossaryMatch("patients", "المرضى", 1),
        GlossaryMatch("treatment", "العلاج", 1)
    ]


@pytest.fixture
def sample_fuzzy_matches():
    """Sample fuzzy matches."""
    return [
        FuzzyMatch(
            "Patients receive treatment in hospitals.",
            "يتلقى المرضى العلاج في المستشفيات.",
            0.92
        ),
        FuzzyMatch(
            "The intensive care unit provides specialized treatment.",
            "توفر وحدة العناية المركزة علاجًا متخصصًا.",
            0.88
        )
    ]


def test_build_context_aware_prompt(sample_glossary, sample_fuzzy_matches):
    """Test building context-aware prompt."""
    source_text = "Patients in intensive care require special treatment."
    domain = "medical"
    
    prompt = PromptBuilder.build_context_aware_prompt(
        source_text=source_text,
        glossary=sample_glossary,
        fuzzy_matches=sample_fuzzy_matches,
        domain=domain
    )
    
    # Check prompt structure
    assert source_text in prompt
    assert domain in prompt.lower()
    
    # Check glossary formatting
    assert "intensive care" in prompt
    assert "العناية المركزة" in prompt
    
    # Check fuzzy matches
    assert "Example" in prompt
    assert sample_fuzzy_matches[0].source_text in prompt
    
    # Check strategy emphasis
    assert "natural" in prompt.lower() or "fluent" in prompt.lower()
    assert "reference" in prompt.lower()  # Glossary as reference
    
    # Check reasonable length
    assert len(prompt) > 500
    assert len(prompt) < 3000


def test_build_terminology_prompt(sample_glossary, sample_fuzzy_matches):
    """Test building terminology-optimized prompt."""
    source_text = "Patients in intensive care require special treatment."
    domain = "medical"
    
    prompt = PromptBuilder.build_terminology_prompt(
        source_text=source_text,
        glossary=sample_glossary,
        fuzzy_matches=sample_fuzzy_matches,
        domain=domain
    )
    
    # Check prompt structure
    assert source_text in prompt
    assert domain in prompt.lower()
    
    # Check terminology emphasis
    assert "MANDATORY" in prompt or "mandatory" in prompt.lower()
    assert "EXACT" in prompt or "exact" in prompt.lower()
    
    # Check glossary formatting emphasizes compliance
    assert "intensive care" in prompt
    assert "العناية المركزة" in prompt
    
    # Should warn about deviations
    assert "WARNING" in prompt or "warning" in prompt.lower()
    
    # Check reasonable length
    assert len(prompt) > 500
    assert len(prompt) < 3000


def test_build_conservative_prompt(sample_glossary, sample_fuzzy_matches):
    """Test building conservative prompt."""
    source_text = "Patients in intensive care require special treatment."
    domain = "medical"
    
    prompt = PromptBuilder.build_conservative_prompt(
        source_text=source_text,
        glossary=sample_glossary,
        fuzzy_matches=sample_fuzzy_matches,
        domain=domain
    )
    
    # Check prompt structure
    assert source_text in prompt
    assert domain in prompt.lower()
    
    # Check conservative emphasis
    assert "literal" in prompt.lower()
    assert "structure" in prompt.lower() or "preserv" in prompt.lower()
    
    # Check glossary present
    assert "intensive care" in prompt
    
    # Check reasonable length
    assert len(prompt) > 500
    assert len(prompt) < 3000


def test_build_all_prompts(sample_glossary, sample_fuzzy_matches):
    """Test building all three prompts at once."""
    source_text = "Test sentence for translation."
    domain = "technical"
    
    prompts = PromptBuilder.build_prompts(
        source_text=source_text,
        glossary=sample_glossary,
        fuzzy_matches=sample_fuzzy_matches,
        domain=domain
    )
    
    # Should have all three agent types
    assert len(prompts) == 3
    assert AgentType.CONTEXT_AWARE in prompts
    assert AgentType.TERMINOLOGY_OPTIMIZED in prompts
    assert AgentType.CONSERVATIVE in prompts
    
    # Each should be different
    prompt_texts = [prompts[agent] for agent in prompts]
    assert len(set(prompt_texts)) == 3  # All unique
    
    # All should contain source text
    for prompt in prompts.values():
        assert source_text in prompt


def test_empty_glossary():
    """Test prompt building with empty glossary."""
    source_text = "Test sentence."
    domain = "general"
    
    prompt = PromptBuilder.build_context_aware_prompt(
        source_text=source_text,
        glossary=[],
        fuzzy_matches=[],
        domain=domain
    )
    
    # Should still generate valid prompt
    assert len(prompt) > 300
    assert source_text in prompt
    assert "No specific terminology" in prompt or "terminology" in prompt.lower()


def test_empty_fuzzy_matches(sample_glossary):
    """Test prompt building with empty fuzzy matches."""
    source_text = "Test sentence."
    domain = "general"
    
    prompt = PromptBuilder.build_terminology_prompt(
        source_text=source_text,
        glossary=sample_glossary,
        fuzzy_matches=[],
        domain=domain
    )
    
    # Should still generate valid prompt
    assert len(prompt) > 300
    assert source_text in prompt
    assert "No translation examples" in prompt or "example" in prompt.lower()


def test_prompt_differences():
    """Test that three prompts have distinct strategies."""
    source_text = "Patients need treatment."
    glossary = [GlossaryMatch("patients", "المرضى", 1)]
    fuzzy = [FuzzyMatch("Test.", "اختبار.", 0.9)]
    domain = "medical"
    
    prompts = PromptBuilder.build_prompts(source_text, glossary, fuzzy, domain)
    
    context_prompt = prompts[AgentType.CONTEXT_AWARE]
    terminology_prompt = prompts[AgentType.TERMINOLOGY_OPTIMIZED]
    conservative_prompt = prompts[AgentType.CONSERVATIVE]
    
    # Context-aware emphasizes naturalness
    assert "natural" in context_prompt.lower() or "fluent" in context_prompt.lower()
    
    # Terminology emphasizes compliance
    assert "MANDATORY" in terminology_prompt or "exact" in terminology_prompt.lower()
    
    # Conservative emphasizes literalness
    assert "literal" in conservative_prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
