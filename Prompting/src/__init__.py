"""Initialize the src package."""

from .models import (
    GlossaryMatch,
    FuzzyMatch,
    RAGOutput,
    TranslationVariant,
    OrchestrationResult,
    OrchestratorConfig,
    AgentType,
    ValidationResult,
    ValidationReport,
    GlossaryAnalysis,
    CritiqueReport
)

__all__ = [
    "GlossaryMatch",
    "FuzzyMatch",
    "RAGOutput",
    "TranslationVariant",
    "OrchestrationResult",
    "OrchestratorConfig",
    "AgentType",
    "ValidationResult",
    "ValidationReport",
    "GlossaryAnalysis",
    "CritiqueReport"
]

__version__ = "1.0.0"
