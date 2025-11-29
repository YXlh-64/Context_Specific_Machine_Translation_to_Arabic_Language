"""Validation logic for RAG output and translations."""

import re
import logging
from typing import List, Dict, Set
from rapidfuzz import fuzz

from .models import (
    RAGOutput,
    GlossaryMatch,
    FuzzyMatch,
    ValidationResult,
    ValidationReport,
    GlossaryAnalysis,
    OrchestratorConfig
)

logger = logging.getLogger(__name__)


def is_valid_arabic(text: str) -> bool:
    """
    Check if text contains Arabic characters.
    
    Args:
        text: Text to check
    
    Returns:
        True if text contains Arabic characters
    """
    if not text:
        return False
    return any('\u0600' <= char <= '\u06FF' for char in text)


def check_length_ratio(source: str, translation: str) -> float:
    """
    Calculate length ratio between source and translation.
    
    Args:
        source: Source text
        translation: Translation text
    
    Returns:
        Length ratio (translation_words / source_words)
    """
    source_words = len(source.split())
    trans_words = len(translation.split())
    
    if source_words == 0:
        return 0.0
    
    return trans_words / source_words


def is_length_ratio_reasonable(ratio: float, min_ratio: float = 0.4, max_ratio: float = 2.5) -> bool:
    """
    Check if length ratio is within reasonable bounds.
    
    Args:
        ratio: Length ratio
        min_ratio: Minimum acceptable ratio
        max_ratio: Maximum acceptable ratio
    
    Returns:
        True if ratio is reasonable
    """
    return min_ratio <= ratio <= max_ratio


def find_glossary_violations(translation: str, glossary: List[GlossaryMatch]) -> List[str]:
    """
    Find glossary terms that should be present but aren't.
    
    Args:
        translation: Translation text
        glossary: List of glossary matches
    
    Returns:
        List of source terms that are missing their target translation
    """
    violations = []
    
    for match in glossary:
        if match.target_term not in translation:
            violations.append(match.source_term)
    
    return violations


def calculate_glossary_compliance(translation: str, glossary: List[GlossaryMatch]) -> float:
    """
    Calculate glossary compliance percentage.
    
    Args:
        translation: Translation text
        glossary: List of glossary matches
    
    Returns:
        Compliance percentage (0-100)
    """
    if not glossary:
        return 100.0
    
    found = 0
    for match in glossary:
        if match.target_term in translation:
            found += 1
    
    return (found / len(glossary)) * 100


def validate_rag_output(rag_output: RAGOutput, config: OrchestratorConfig) -> ValidationResult:
    """
    Validate RAG output quality.
    
    Checks:
    - Glossary matches are non-empty and valid
    - Fuzzy matches meet minimum similarity threshold
    - Data structure completeness
    
    Args:
        rag_output: RAG output to validate
        config: Orchestrator configuration
    
    Returns:
        ValidationResult with issues and filtered fuzzy matches
    """
    result = ValidationResult(is_valid=True)
    
    # Check source text
    if not rag_output.source_text or len(rag_output.source_text.strip()) == 0:
        result.add_issue("Source text is empty")
    
    # Check domain
    if not rag_output.domain:
        result.add_warning("Domain is empty")
    
    # Check glossary matches
    if not rag_output.glossary_matches:
        result.add_warning("No glossary matches provided")
    else:
        for idx, match in enumerate(rag_output.glossary_matches):
            if not match.source_term:
                result.add_issue(f"Glossary match {idx}: source term is empty")
            if not match.target_term:
                result.add_issue(f"Glossary match {idx}: target term is empty")
            if not is_valid_arabic(match.target_term):
                result.add_warning(f"Glossary match {idx}: target term '{match.target_term}' doesn't contain Arabic")
    
    # Filter and validate fuzzy matches
    filtered_fuzzy = []
    for idx, match in enumerate(rag_output.fuzzy_matches):
        # Check similarity threshold
        if match.similarity_score < config.min_fuzzy_match_similarity:
            result.add_warning(
                f"Fuzzy match {idx} filtered out: similarity {match.similarity_score} < {config.min_fuzzy_match_similarity}"
            )
            continue
        
        # Check for empty texts
        if not match.source_text or not match.target_text:
            result.add_warning(f"Fuzzy match {idx}: empty source or target text")
            continue
        
        # Check target contains Arabic
        if not is_valid_arabic(match.target_text):
            result.add_warning(f"Fuzzy match {idx}: target doesn't contain Arabic")
            continue
        
        filtered_fuzzy.append(match)
    
    result.filtered_fuzzy_matches = filtered_fuzzy
    
    if not filtered_fuzzy:
        result.add_warning("No fuzzy matches passed filters")
    
    logger.info(f"Validation: {len(result.issues)} issues, {len(result.warnings)} warnings, "
                f"{len(filtered_fuzzy)}/{len(rag_output.fuzzy_matches)} fuzzy matches kept")
    
    return result


def analyze_glossary_coverage(
    source_text: str,
    glossary_matches: List[GlossaryMatch]
) -> GlossaryAnalysis:
    """
    Analyze glossary coverage of source text.
    
    Calculates what percentage of source text is covered by glossary terms.
    
    Args:
        source_text: Source text to analyze
        glossary_matches: List of glossary matches
    
    Returns:
        GlossaryAnalysis with coverage statistics
    """
    source_lower = source_text.lower()
    total_words = len(source_text.split())
    covered_words = 0
    
    mandatory_terms = []
    optional_terms = []
    
    for match in glossary_matches:
        source_term_lower = match.source_term.lower()
        
        # Check if term appears in source
        if source_term_lower in source_lower:
            covered_words += len(match.source_term.split())
            mandatory_terms.append(match)
        else:
            optional_terms.append(match)
    
    coverage_percentage = (covered_words / total_words * 100) if total_words > 0 else 0
    uncovered_words = total_words - covered_words
    
    analysis = GlossaryAnalysis(
        coverage_percentage=coverage_percentage,
        mandatory_terms=mandatory_terms,
        optional_terms=optional_terms,
        uncovered_source_words=uncovered_words,
        total_source_words=total_words
    )
    
    logger.info(f"Glossary coverage: {coverage_percentage:.1f}%, "
                f"{len(mandatory_terms)} mandatory terms, {len(optional_terms)} optional")
    
    return analysis


def filter_and_rank_fuzzy_matches(
    fuzzy_matches: List[FuzzyMatch],
    source_text: str,
    config: OrchestratorConfig
) -> List[FuzzyMatch]:
    """
    Filter and rank fuzzy matches by quality.
    
    Process:
    1. Keep only high quality (>= high_quality_similarity threshold)
    2. Filter by length similarity
    3. Rank by combined score
    4. Return top N matches
    
    Args:
        fuzzy_matches: List of fuzzy matches
        source_text: Source text being translated
        config: Orchestrator configuration
    
    Returns:
        Filtered and ranked list of top fuzzy matches
    """
    if not fuzzy_matches:
        return []
    
    scored_matches = []
    source_len = len(source_text.split())
    
    for match in fuzzy_matches:
        # Calculate length ratio
        match_len = len(match.source_text.split())
        if source_len > 0:
            length_ratio = match_len / source_len
        else:
            length_ratio = 1.0
        
        # Filter by length
        if not (config.length_filter_min_ratio <= length_ratio <= config.length_filter_max_ratio):
            logger.debug(f"Filtered match by length ratio {length_ratio:.2f}")
            continue
        
        # Calculate length similarity score (closer to 1.0 is better)
        length_sim = 1.0 - abs(1.0 - length_ratio)
        
        # Calculate combined score
        # Weights: similarity (60%), length similarity (20%), position bonus (20%)
        combined_score = (
            match.similarity_score * 0.6 +
            length_sim * 0.2 +
            (1.0 if match.similarity_score >= config.high_quality_similarity else 0.8) * 0.2
        )
        
        scored_matches.append((combined_score, match))
    
    # Sort by score descending
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Return top N
    top_matches = [match for score, match in scored_matches[:config.max_fuzzy_matches]]
    
    logger.info(f"Filtered fuzzy matches: {len(fuzzy_matches)} → {len(top_matches)}")
    
    return top_matches


def validate_translations(
    translations: Dict[str, str],
    glossary: List[GlossaryMatch],
    source_text: str,
    config: OrchestratorConfig
) -> ValidationReport:
    """
    Validate generated translations.
    
    Checks:
    - Not empty and contains Arabic characters
    - Glossary compliance
    - Length reasonableness
    
    Args:
        translations: Dict mapping agent_type to translation text
        glossary: List of glossary matches
        source_text: Original source text
        config: Orchestrator configuration
    
    Returns:
        ValidationReport with all validation results
    """
    report = ValidationReport(is_valid=True)
    
    for agent_type, translation in translations.items():
        # Check not empty
        if not translation or len(translation.strip()) == 0:
            report.add_issue(agent_type, "Translation is empty")
            report.contains_arabic[agent_type] = False
            continue
        
        # Check contains Arabic
        has_arabic = is_valid_arabic(translation)
        report.contains_arabic[agent_type] = has_arabic
        if not has_arabic:
            report.add_issue(agent_type, "Translation doesn't contain Arabic characters")
        
        # Check length ratio
        length_ratio = check_length_ratio(source_text, translation)
        report.length_ratios[agent_type] = length_ratio
        
        if not is_length_ratio_reasonable(length_ratio, config.min_length_ratio, config.max_length_ratio):
            report.add_issue(
                agent_type,
                f"Length ratio {length_ratio:.2f} outside reasonable bounds "
                f"({config.min_length_ratio}-{config.max_length_ratio})"
            )
        
        # Check glossary compliance
        compliance = calculate_glossary_compliance(translation, glossary)
        report.glossary_compliance[agent_type] = compliance
        
        violations = find_glossary_violations(translation, glossary)
        if violations and agent_type == "terminology_optimized":
            # Terminology agent should have perfect compliance
            report.add_issue(
                agent_type,
                f"Missing glossary terms: {', '.join(violations[:3])}"
            )
    
    logger.info(f"Translation validation: {'PASSED' if report.is_valid else 'FAILED'}")
    for agent_type in translations.keys():
        logger.info(f"  {agent_type}: compliance={report.glossary_compliance.get(agent_type, 0):.1f}%, "
                   f"ratio={report.length_ratios.get(agent_type, 0):.2f}")
    
    return report


def detect_cross_variant_inconsistencies(translations: Dict[str, str]) -> List[str]:
    """
    Detect inconsistencies across translation variants.
    
    Args:
        translations: Dict of agent_type to translation text
    
    Returns:
        List of inconsistency descriptions
    """
    inconsistencies = []
    
    if len(translations) < 2:
        return inconsistencies
    
    # Check if translations are too similar (possible duplication)
    translation_list = list(translations.items())
    for i in range(len(translation_list)):
        for j in range(i + 1, len(translation_list)):
            agent1, text1 = translation_list[i]
            agent2, text2 = translation_list[j]
            
            similarity = fuzz.ratio(text1, text2)
            
            if similarity > 95:
                inconsistencies.append(
                    f"{agent1} and {agent2} produced nearly identical translations (similarity: {similarity}%)"
                )
    
    return inconsistencies


if __name__ == "__main__":
    # Test validation functions
    logging.basicConfig(level=logging.INFO)
    
    from .models import OrchestratorConfig
    
    config = OrchestratorConfig()
    
    # Test Arabic detection
    assert is_valid_arabic("يجب نقل المرضى")
    assert not is_valid_arabic("This is English")
    
    # Test length ratio
    ratio = check_length_ratio("Hello world", "مرحبا بالعالم")
    print(f"Length ratio: {ratio}")
    
    # Test glossary compliance
    glossary = [
        GlossaryMatch("patients", "المرضى", 1),
        GlossaryMatch("intensive care", "العناية المركزة", 2)
    ]
    
    translation = "يجب نقل المرضى إلى العناية المركزة"
    compliance = calculate_glossary_compliance(translation, glossary)
    print(f"Glossary compliance: {compliance}%")
    
    violations = find_glossary_violations("يجب نقل المرضى", glossary)
    print(f"Violations: {violations}")
