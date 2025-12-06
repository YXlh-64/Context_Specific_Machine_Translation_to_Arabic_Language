"""Data models for the Translation Orchestrator."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class AgentType(str, Enum):
    """Translation agent types."""
    CONTEXT_AWARE = "context_aware"
    TERMINOLOGY_OPTIMIZED = "terminology_optimized"
    CONSERVATIVE = "conservative"


@dataclass
class GlossaryMatch:
    """Represents a glossary term match from RAG output."""
    source_term: str
    target_term: str
    ngram_size: int
    
    def __post_init__(self):
        """Validate and clean glossary match data."""
        self.source_term = self.source_term.strip()
        self.target_term = self.target_term.strip()
        if self.ngram_size <= 0:
            self.ngram_size = len(self.source_term.split())


@dataclass
class FuzzyMatch:
    """Represents a fuzzy translation match from RAG output."""
    source_text: str
    target_text: str
    similarity_score: float
    context: Optional[str] = None
    
    def __post_init__(self):
        """Validate fuzzy match data."""
        self.source_text = self.source_text.strip()
        self.target_text = self.target_text.strip()
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"Similarity score must be between 0 and 1, got {self.similarity_score}")


@dataclass
class RAGOutput:
    """Structured RAG team output."""
    glossary_matches: List[GlossaryMatch]
    fuzzy_matches: List[FuzzyMatch]
    source_text: str
    domain: str
    
    def __post_init__(self):
        """Validate RAG output data."""
        self.source_text = self.source_text.strip()
        self.domain = self.domain.strip().lower()
        if not self.source_text:
            raise ValueError("Source text cannot be empty")


@dataclass
class GlossaryAnalysis:
    """Analysis of glossary coverage for source text."""
    coverage_percentage: float
    mandatory_terms: List[GlossaryMatch]
    optional_terms: List[GlossaryMatch]
    uncovered_source_words: int
    total_source_words: int
    
    def __repr__(self):
        return (f"GlossaryAnalysis(coverage={self.coverage_percentage:.1f}%, "
                f"mandatory={len(self.mandatory_terms)}, "
                f"optional={len(self.optional_terms)})")


@dataclass
class ValidationResult:
    """Result of RAG output validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    filtered_fuzzy_matches: List[FuzzyMatch] = field(default_factory=list)
    
    def add_issue(self, issue: str):
        """Add a validation issue."""
        self.issues.append(issue)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a validation warning."""
        self.warnings.append(warning)


@dataclass
class ValidationReport:
    """Validation report for generated translations."""
    is_valid: bool
    issues: Dict[str, List[str]] = field(default_factory=dict)
    glossary_compliance: Dict[str, float] = field(default_factory=dict)
    length_ratios: Dict[str, float] = field(default_factory=dict)
    contains_arabic: Dict[str, bool] = field(default_factory=dict)
    
    def add_issue(self, agent_type: str, issue: str):
        """Add an issue for specific agent."""
        if agent_type not in self.issues:
            self.issues[agent_type] = []
        self.issues[agent_type].append(issue)
        self.is_valid = False


@dataclass
class TranslationVariant:
    """A single translation variant from one agent."""
    text: str
    agent_type: str
    quality_score: float
    glossary_compliance: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate translation variant."""
        if not 0.0 <= self.quality_score <= 10.0:
            raise ValueError(f"Quality score must be between 0 and 10, got {self.quality_score}")
        if not 0.0 <= self.glossary_compliance <= 100.0:
            raise ValueError(f"Glossary compliance must be between 0 and 100, got {self.glossary_compliance}")


@dataclass
class CritiqueReport:
    """Comparison and critique of translation variants."""
    quality_scores: Dict[str, float]
    strengths: Dict[str, List[str]]
    weaknesses: Dict[str, List[str]]
    key_differences: List[str]
    recommended_variant: str
    recommendation_reason: str
    
    def get_ranked_variants(self) -> List[tuple]:
        """Get variants ranked by quality score."""
        return sorted(self.quality_scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class OrchestrationResult:
    """Complete result of the orchestration process."""
    source_text: str
    domain: str
    translations: Dict[str, TranslationVariant]
    recommended_variant: str
    key_differences: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_text": self.source_text,
            "domain": self.domain,
            "translations": {
                agent_type: {
                    "text": variant.text,
                    "quality_score": variant.quality_score,
                    "glossary_compliance": variant.glossary_compliance,
                    "strengths": variant.strengths,
                    "weaknesses": variant.weaknesses
                }
                for agent_type, variant in self.translations.items()
            },
            "recommended_variant": self.recommended_variant,
            "key_differences": self.key_differences,
            "processing_time_ms": self.processing_time_ms
        }


class OrchestratorConfig(BaseModel):
    """Configuration for the Translation Orchestrator."""
    
    # Model settings
    default_model: str = Field(default="llama-3.3-70b-versatile", description="Default LLM model to use (Groq LLaMA 3.3 - FREE)")
    max_tokens: int = Field(default=2048, description="Maximum tokens for completion")
    
    # Temperature settings per agent (increased spread for differentiation)
    context_aware_temperature: float = Field(default=0.8, ge=0.0, le=2.0)  # More creative, fluent
    terminology_temperature: float = Field(default=0.4, ge=0.0, le=2.0)    # Balanced
    conservative_temperature: float = Field(default=0.1, ge=0.0, le=2.0)   # Literal, deterministic
    
    # Validation thresholds
    min_fuzzy_match_similarity: float = Field(default=0.65, ge=0.0, le=1.0)
    high_quality_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    min_length_ratio: float = Field(default=0.4, ge=0.1, le=1.0)
    max_length_ratio: float = Field(default=2.5, ge=1.0, le=10.0)
    
    # Fuzzy match filtering
    max_fuzzy_matches: int = Field(default=5, ge=1, le=10)
    length_filter_min_ratio: float = Field(default=0.3, ge=0.1, le=1.0)
    length_filter_max_ratio: float = Field(default=3.0, ge=1.0, le=10.0)
    
    # Processing settings
    enable_parallel_generation: bool = Field(default=True)
    retry_attempts: int = Field(default=3, ge=1, le=5)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Iterative refinement settings
    enable_iterative_refinement: bool = Field(default=False, description="Enable iterative refinement loop")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Maximum refinement iterations")
    min_quality_threshold: float = Field(default=7.0, ge=0.0, le=10.0, description="Minimum quality score to accept")
    min_glossary_compliance: float = Field(default=80.0, ge=0.0, le=100.0, description="Minimum glossary compliance %")
    improvement_threshold: float = Field(default=0.5, ge=0.0, le=5.0, description="Minimum improvement to continue")
    
    # Quality scoring weights
    glossary_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    fluency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    accuracy_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    completeness_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    grammar_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    @validator('glossary_weight', 'fluency_weight', 'accuracy_weight', 'completeness_weight', 'grammar_weight')
    def validate_weights(cls, v, values):
        """Ensure weights sum to approximately 1.0."""
        return v
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


@dataclass
class LLMRequest:
    """Request to LLM for translation."""
    prompt: str
    agent_type: AgentType
    temperature: float
    model: str
    max_tokens: int


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    agent_type: AgentType
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    
    def __post_init__(self):
        """Clean response text."""
        self.text = self.text.strip()


@dataclass
class ProcessingMetrics:
    """Metrics for orchestration process."""
    total_time_ms: float
    validation_time_ms: float
    prompt_building_time_ms: float
    translation_time_ms: float
    critique_time_ms: float
    llm_calls: int
    tokens_used: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_time_ms": self.total_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "prompt_building_time_ms": self.prompt_building_time_ms,
            "translation_time_ms": self.translation_time_ms,
            "critique_time_ms": self.critique_time_ms,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used
        }
