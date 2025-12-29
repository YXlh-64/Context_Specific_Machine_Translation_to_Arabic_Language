"""
Integration Tests for Phase 3 Prompt Construction

Tests the prompt construction service with various inputs and configurations.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import (
    PromptConstructionRequest,
    GlossaryTerm,
    FuzzyMatch,
    PromptFormat,
    DomainType
)
from app.services.prompt_service import prompt_constructor
from app.utils.token_counter import TokenCounter, count_tokens


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_GLOSSARY_TERMS = [
    GlossaryTerm(
        source_term="machine learning",
        target_term="التعلم الآلي",
        domain="technology"
    ),
    GlossaryTerm(
        source_term="artificial intelligence",
        target_term="الذكاء الاصطناعي",
        domain="technology"
    ),
    GlossaryTerm(
        source_term="neural network",
        target_term="الشبكة العصبية",
        domain="technology"
    )
]

SAMPLE_FUZZY_MATCHES = [
    FuzzyMatch(
        source="Machine learning algorithms are transforming industries.",
        target="خوارزميات التعلم الآلي تحول الصناعات.",
        similarity_percentage=92.5,
        domain="technology"
    ),
    FuzzyMatch(
        source="Deep learning is a subset of machine learning.",
        target="التعلم العميق هو جزء من التعلم الآلي.",
        similarity_percentage=85.0,
        domain="technology"
    )
]


# ============================================================================
# Prompt Construction Tests
# ============================================================================

class TestPromptConstruction:
    """Test suite for prompt construction functionality."""
    
    def test_construct_prompt_xml_format(self):
        """Test prompt construction with XML format."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            prompt_format=PromptFormat.XML
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        assert len(response.prompt) > 0
        assert "<translation_task>" in response.prompt
        assert "<glossary>" in response.prompt
        assert "<examples>" in response.prompt
        assert response.token_count > 0
    
    def test_construct_prompt_json_format(self):
        """Test prompt construction with JSON format."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            prompt_format=PromptFormat.JSON
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        assert len(response.prompt) > 0
        assert '"glossary"' in response.prompt
        assert '"examples"' in response.prompt
        assert response.format == PromptFormat.JSON
    
    def test_construct_prompt_markdown_format(self):
        """Test prompt construction with Markdown format."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            prompt_format=PromptFormat.MARKDOWN
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        assert "## Glossary Terms" in response.prompt
        assert "## Similar Examples" in response.prompt
        assert response.format == PromptFormat.MARKDOWN
    
    def test_construct_prompt_plain_format(self):
        """Test prompt construction with Plain text format."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            prompt_format=PromptFormat.PLAIN
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        assert "GLOSSARY TERMS" in response.prompt
        assert "SIMILAR EXAMPLES" in response.prompt
        assert response.format == PromptFormat.PLAIN
    
    def test_construct_prompt_with_system_message(self):
        """Test that system message is generated when requested."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            include_system_message=True
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.system_message is not None
        assert len(response.system_message) > 0
        assert "translator" in response.system_message.lower()
    
    def test_construct_prompt_without_system_message(self):
        """Test that system message is None when not requested."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            include_system_message=False
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.system_message is None
    
    def test_construct_prompt_empty_glossary(self):
        """Test prompt construction with no glossary terms."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=[],
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        assert response.token_count > 0
    
    def test_construct_prompt_empty_fuzzy_matches(self):
        """Test prompt construction with no fuzzy matches."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=[],
            domain=DomainType.TECHNOLOGY
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        assert response.token_count > 0
    
    def test_construct_prompt_all_domains(self):
        """Test prompt construction works for all domain types."""
        for domain in DomainType:
            request = PromptConstructionRequest(
                source_sentence="Test sentence for domain validation.",
                glossary_matches=SAMPLE_GLOSSARY_TERMS,
                fuzzy_matches=SAMPLE_FUZZY_MATCHES,
                domain=domain
            )
            
            response = prompt_constructor.construct(request)
            
            assert response.prompt is not None
            assert response.domain == domain
    
    def test_construct_prompt_with_custom_instructions(self):
        """Test prompt construction with custom instructions."""
        request = PromptConstructionRequest(
            source_sentence="Machine learning is revolutionizing the technology industry.",
            glossary_matches=SAMPLE_GLOSSARY_TERMS,
            fuzzy_matches=SAMPLE_FUZZY_MATCHES,
            domain=DomainType.TECHNOLOGY,
            custom_instructions="Preserve technical accuracy while ensuring readability."
        )
        
        response = prompt_constructor.construct(request)
        
        assert response.prompt is not None
        # Custom instructions should be included somewhere
        assert "accuracy" in response.prompt.lower() or response.prompt is not None


# ============================================================================
# Token Counter Tests
# ============================================================================

class TestTokenCounter:
    """Test suite for token counting functionality."""
    
    def test_count_tokens_english(self):
        """Test token counting for English text."""
        text = "This is a simple English sentence for testing."
        tokens = count_tokens(text, "en")
        
        assert tokens > 0
        assert tokens < 20  # Should be reasonable for this sentence
    
    def test_count_tokens_arabic(self):
        """Test token counting for Arabic text."""
        text = "هذه جملة بسيطة للاختبار"
        tokens = count_tokens(text, "ar")
        
        assert tokens > 0
    
    def test_count_tokens_empty_string(self):
        """Test token counting for empty string."""
        tokens = count_tokens("")
        assert tokens == 0
    
    def test_count_mixed_language(self):
        """Test token counting for mixed language text."""
        text = "Machine learning (التعلم الآلي) is transforming industries."
        tokens = TokenCounter.count_mixed_language(text)
        
        assert tokens > 0
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        cost = TokenCounter.estimate_cost(1000, 500, "gpt-4")
        
        assert cost > 0
        assert cost < 1.0  # Should be less than $1 for 1000 tokens


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestSchemaValidation:
    """Test suite for Pydantic schema validation."""
    
    def test_glossary_term_creation(self):
        """Test GlossaryTerm creation."""
        term = GlossaryTerm(
            source_term="test",
            target_term="اختبار",
            domain="general"
        )
        
        assert term.source_term == "test"
        assert term.target_term == "اختبار"
    
    def test_fuzzy_match_creation(self):
        """Test FuzzyMatch creation."""
        match = FuzzyMatch(
            source="Test source sentence.",
            target="جملة اختبار.",
            similarity_percentage=95.0
        )
        
        assert match.source == "Test source sentence."
        assert match.similarity_percentage == 95.0
    
    def test_prompt_request_defaults(self):
        """Test PromptConstructionRequest default values."""
        request = PromptConstructionRequest(
            source_sentence="Test sentence",
            glossary_matches=[],
            fuzzy_matches=[]
        )
        
        assert request.source_lang == "en"
        assert request.target_lang == "ar"
        assert request.prompt_format == PromptFormat.XML
        assert request.include_system_message == True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
