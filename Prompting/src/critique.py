"""Critique system for comparing and scoring translation variants."""

import logging
from typing import List, Dict
from rapidfuzz import fuzz

from .models import (
    GlossaryMatch,
    TranslationVariant,
    CritiqueReport,
    ValidationReport,
    AgentType,
    OrchestratorConfig
)
from .validator import calculate_glossary_compliance, is_valid_arabic, check_length_ratio

logger = logging.getLogger(__name__)


class TranslationCritic:
    """
    Critic for analyzing and comparing translation variants.
    
    Generates quality scores, identifies strengths/weaknesses,
    and recommends the best variant for HITL interface.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize critic with configuration.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
    
    def _calculate_fluency_score(
        self,
        translation: str,
        agent_type: AgentType
    ) -> float:
        """
        Estimate fluency score based on translation characteristics.
        
        This is a heuristic-based estimation. In production, you might
        use a language model-based fluency scorer.
        
        Args:
            translation: Translation text
            agent_type: Agent type that generated translation
        
        Returns:
            Fluency score (0-10)
        """
        score = 7.0  # Base score
        
        # Context-aware agent gets bonus for fluency
        if agent_type == AgentType.CONTEXT_AWARE:
            score += 1.5
        elif agent_type == AgentType.CONSERVATIVE:
            score -= 1.0  # Conservative might be less fluent
        
        # Check for basic Arabic fluency indicators
        if is_valid_arabic(translation):
            score += 0.5
        
        # Penalize very short or very long translations
        word_count = len(translation.split())
        if word_count < 3:
            score -= 2.0
        elif word_count > 50:
            score -= 0.5
        
        return max(0.0, min(10.0, score))
    
    def _calculate_accuracy_score(
        self,
        translation: str,
        source_text: str,
        agent_type: AgentType
    ) -> float:
        """
        Estimate accuracy score.
        
        Args:
            translation: Translation text
            source_text: Source text
            agent_type: Agent type
        
        Returns:
            Accuracy score (0-10)
        """
        score = 8.0  # Base score (assume generally accurate)
        
        # Conservative translation tends to be more accurate
        if agent_type == AgentType.CONSERVATIVE:
            score += 1.0
        
        # Check length ratio (extreme ratios suggest potential issues)
        ratio = check_length_ratio(source_text, translation)
        if ratio < 0.3 or ratio > 3.0:
            score -= 2.0
        elif ratio < 0.5 or ratio > 2.0:
            score -= 0.5
        
        return max(0.0, min(10.0, score))
    
    def _calculate_completeness_score(
        self,
        translation: str,
        source_text: str
    ) -> float:
        """
        Estimate completeness score.
        
        Args:
            translation: Translation text
            source_text: Source text
        
        Returns:
            Completeness score (0-10)
        """
        score = 9.0  # Base score (assume mostly complete)
        
        # Check if translation is too short compared to source
        ratio = check_length_ratio(source_text, translation)
        if ratio < 0.4:
            score -= 3.0
        elif ratio < 0.6:
            score -= 1.0
        
        # Check for empty or very short translations
        if not translation or len(translation.strip()) < 5:
            score = 0.0
        
        return max(0.0, min(10.0, score))
    
    def _calculate_grammar_score(
        self,
        translation: str,
        agent_type: AgentType
    ) -> float:
        """
        Estimate grammar score.
        
        Args:
            translation: Translation text
            agent_type: Agent type
        
        Returns:
            Grammar score (0-10)
        """
        score = 8.5  # Base score (assume generally good grammar)
        
        # All agents should produce grammatically correct output
        # This is a placeholder - in production, use grammar checker
        
        if not is_valid_arabic(translation):
            score -= 5.0
        
        return max(0.0, min(10.0, score))
    
    def calculate_quality_score(
        self,
        translation: str,
        source_text: str,
        glossary_compliance: float,
        agent_type: AgentType
    ) -> float:
        """
        Calculate overall quality score.
        
        Quality formula:
        - Glossary compliance (30%)
        - Fluency (30%)
        - Accuracy (20%)
        - Completeness (10%)
        - Grammar (10%)
        
        Args:
            translation: Translation text
            source_text: Source text
            glossary_compliance: Glossary compliance percentage (0-100)
            agent_type: Agent type
        
        Returns:
            Quality score (0-10)
        """
        # Normalize glossary compliance to 0-10 scale
        glossary_score = glossary_compliance / 10.0
        
        fluency_score = self._calculate_fluency_score(translation, agent_type)
        accuracy_score = self._calculate_accuracy_score(translation, source_text, agent_type)
        completeness_score = self._calculate_completeness_score(translation, source_text)
        grammar_score = self._calculate_grammar_score(translation, agent_type)
        
        # Weighted combination
        quality_score = (
            glossary_score * self.config.glossary_weight +
            fluency_score * self.config.fluency_weight +
            accuracy_score * self.config.accuracy_weight +
            completeness_score * self.config.completeness_weight +
            grammar_score * self.config.grammar_weight
        )
        
        logger.debug(
            f"{agent_type}: glossary={glossary_score:.1f}, fluency={fluency_score:.1f}, "
            f"accuracy={accuracy_score:.1f}, completeness={completeness_score:.1f}, "
            f"grammar={grammar_score:.1f} → quality={quality_score:.1f}"
        )
        
        return round(quality_score, 1)
    
    def identify_strengths(
        self,
        translation: str,
        agent_type: AgentType,
        glossary_compliance: float,
        quality_score: float
    ) -> List[str]:
        """
        Identify strengths of a translation.
        
        Args:
            translation: Translation text
            agent_type: Agent type
            glossary_compliance: Glossary compliance percentage
            quality_score: Quality score
        
        Returns:
            List of strength descriptions
        """
        strengths = []
        
        # Agent-specific strengths
        if agent_type == AgentType.CONTEXT_AWARE:
            strengths.append("Natural and fluent Arabic expression")
            strengths.append("Professional and readable tone")
            if quality_score >= 8.0:
                strengths.append("Excellent overall quality")
        
        elif agent_type == AgentType.TERMINOLOGY_OPTIMIZED:
            strengths.append("Strong terminology consistency")
            if glossary_compliance >= 90:
                strengths.append("Excellent glossary compliance")
            strengths.append("Professionally precise language")
        
        elif agent_type == AgentType.CONSERVATIVE:
            strengths.append("Close adherence to source structure")
            strengths.append("High fidelity to original meaning")
            strengths.append("Easy to verify against source")
        
        # Universal strengths
        if glossary_compliance == 100:
            strengths.append("Perfect glossary term usage")
        
        if is_valid_arabic(translation):
            strengths.append("Proper Arabic characters and encoding")
        
        return strengths
    
    def identify_weaknesses(
        self,
        translation: str,
        agent_type: AgentType,
        glossary_compliance: float,
        source_text: str
    ) -> List[str]:
        """
        Identify weaknesses of a translation.
        
        Args:
            translation: Translation text
            agent_type: Agent type
            glossary_compliance: Glossary compliance percentage
            source_text: Source text
        
        Returns:
            List of weakness descriptions
        """
        weaknesses = []
        
        # Agent-specific weaknesses
        if agent_type == AgentType.CONTEXT_AWARE:
            if glossary_compliance < 100:
                weaknesses.append("Minor glossary term variations")
            weaknesses.append("Slight structural deviation from source")
        
        elif agent_type == AgentType.TERMINOLOGY_OPTIMIZED:
            weaknesses.append("Somewhat formal register")
            weaknesses.append("Could be more concise")
        
        elif agent_type == AgentType.CONSERVATIVE:
            weaknesses.append("Less natural Arabic flow")
            weaknesses.append("Structure influenced by English")
        
        # Universal weaknesses
        if glossary_compliance < 80:
            weaknesses.append("Low glossary compliance")
        
        ratio = check_length_ratio(source_text, translation)
        if ratio < 0.5:
            weaknesses.append("Translation seems too concise")
        elif ratio > 2.0:
            weaknesses.append("Translation seems overly verbose")
        
        if not is_valid_arabic(translation):
            weaknesses.append("Missing or invalid Arabic text")
        
        return weaknesses
    
    def identify_key_differences(
        self,
        translations: Dict[str, str],
        glossary: List[GlossaryMatch]
    ) -> List[str]:
        """
        Identify key differences between translation variants.
        
        Args:
            translations: Dict mapping agent_type to translation text
            glossary: List of glossary matches
        
        Returns:
            List of key difference descriptions
        """
        differences = []
        
        if len(translations) < 2:
            return differences
        
        # Convert to list for comparison
        trans_list = list(translations.items())
        
        # Compare glossary term usage
        for match in glossary:
            usages = {}
            for agent_type, translation in translations.items():
                if match.target_term in translation:
                    usages[agent_type] = "uses term"
                else:
                    usages[agent_type] = "missing term"
            
            # Check if there are differences
            unique_usages = set(usages.values())
            if len(unique_usages) > 1:
                using = [k for k, v in usages.items() if v == "uses term"]
                missing = [k for k, v in usages.items() if v == "missing term"]
                if using and missing:
                    differences.append(
                        f"Term '{match.source_term}': present in {', '.join(using)} but missing in {', '.join(missing)}"
                    )
        
        # Compare lengths
        lengths = {agent: len(trans.split()) for agent, trans in translations.items()}
        min_len = min(lengths.values())
        max_len = max(lengths.values())
        if max_len > min_len * 1.3:  # 30% difference
            shortest = [k for k, v in lengths.items() if v == min_len][0]
            longest = [k for k, v in lengths.items() if v == max_len][0]
            differences.append(
                f"Length varies: {shortest} is more concise ({min_len} words) vs {longest} is more verbose ({max_len} words)"
            )
        
        # Compare similarity
        for i in range(len(trans_list)):
            for j in range(i + 1, len(trans_list)):
                agent1, text1 = trans_list[i]
                agent2, text2 = trans_list[j]
                
                similarity = fuzz.ratio(text1, text2)
                
                if similarity < 70:
                    differences.append(
                        f"{agent1} and {agent2} use significantly different wording (similarity: {similarity}%)"
                    )
        
        return differences[:5]  # Return top 5 differences
    
    def recommend_variant(
        self,
        quality_scores: Dict[str, float],
        glossary_compliance: Dict[str, float],
        domain: str
    ) -> tuple[str, str]:
        """
        Recommend the best translation variant.
        
        Args:
            quality_scores: Dict mapping agent_type to quality score
            glossary_compliance: Dict mapping agent_type to compliance percentage
            domain: Domain/field
        
        Returns:
            Tuple of (recommended_variant, reason)
        """
        # For medical/technical domains, prioritize terminology compliance
        technical_domains = ["medical", "technical", "legal", "scientific"]
        
        if domain.lower() in technical_domains:
            # Prefer terminology-optimized for technical domains
            if AgentType.TERMINOLOGY_OPTIMIZED.value in quality_scores:
                term_score = quality_scores[AgentType.TERMINOLOGY_OPTIMIZED.value]
                term_compliance = glossary_compliance.get(AgentType.TERMINOLOGY_OPTIMIZED.value, 0)
                
                if term_score >= 8.0 and term_compliance >= 90:
                    return (
                        AgentType.TERMINOLOGY_OPTIMIZED.value,
                        f"Best choice for {domain} domain: excellent terminology compliance and quality"
                    )
        
        # Otherwise, choose highest quality score
        best_agent = max(quality_scores.items(), key=lambda x: x[1])
        
        return (
            best_agent[0],
            f"Highest quality score ({best_agent[1]:.1f}/10) with good overall balance"
        )
    
    def critique_translations(
        self,
        translations: Dict[str, str],
        source_text: str,
        glossary: List[GlossaryMatch],
        validation_report: ValidationReport,
        domain: str
    ) -> CritiqueReport:
        """
        Compare and critique translation variants.
        
        Generates:
        - Quality scores for each variant
        - Strengths for each
        - Weaknesses for each
        - Key differences between variants
        - Recommendation on which to use
        
        Args:
            translations: Dict mapping agent_type to translation text
            source_text: Source text
            glossary: List of glossary matches
            validation_report: Validation report
            domain: Domain/field
        
        Returns:
            CritiqueReport with all analysis
        """
        logger.info(f"Critiquing {len(translations)} translation variants")
        
        quality_scores = {}
        strengths_map = {}
        weaknesses_map = {}
        
        for agent_type, translation in translations.items():
            # Get glossary compliance from validation report
            glossary_compliance = validation_report.glossary_compliance.get(agent_type, 0.0)
            
            # Calculate quality score
            try:
                agent_enum = AgentType(agent_type)
            except ValueError:
                agent_enum = AgentType.CONTEXT_AWARE
            
            quality_score = self.calculate_quality_score(
                translation=translation,
                source_text=source_text,
                glossary_compliance=glossary_compliance,
                agent_type=agent_enum
            )
            quality_scores[agent_type] = quality_score
            
            # Identify strengths and weaknesses
            strengths = self.identify_strengths(
                translation, agent_enum, glossary_compliance, quality_score
            )
            weaknesses = self.identify_weaknesses(
                translation, agent_enum, glossary_compliance, source_text
            )
            
            strengths_map[agent_type] = strengths
            weaknesses_map[agent_type] = weaknesses
        
        # Identify key differences
        key_differences = self.identify_key_differences(translations, glossary)
        
        # Recommend best variant
        recommended_variant, reason = self.recommend_variant(
            quality_scores,
            validation_report.glossary_compliance,
            domain
        )
        
        report = CritiqueReport(
            quality_scores=quality_scores,
            strengths=strengths_map,
            weaknesses=weaknesses_map,
            key_differences=key_differences,
            recommended_variant=recommended_variant,
            recommendation_reason=reason
        )
        
        logger.info(f"Critique complete. Recommended: {recommended_variant}")
        logger.info(f"Quality scores: {quality_scores}")
        
        return report


if __name__ == "__main__":
    # Test critique system
    logging.basicConfig(level=logging.INFO)
    
    from .models import OrchestratorConfig, ValidationReport, GlossaryMatch
    
    config = OrchestratorConfig()
    critic = TranslationCritic(config)
    
    # Mock translations
    translations = {
        "context_aware": "يجب إدخال المرضى الذين يعانون من أعراض شديدة إلى وحدات العناية المركزة",
        "terminology_optimized": "يجب أن يتم إدخالهم المرضى مع الأعراض الشديدة إلى وحدات العناية المركزة",
        "conservative": "المرضى مع الأعراض الشديدة يجب يتم إدخالهم إلى وحدات العناية المركزة"
    }
    
    source_text = "Patients with severe symptoms must be admitted to intensive care units."
    
    glossary = [
        GlossaryMatch("intensive care units", "وحدات العناية المركزة", 3),
        GlossaryMatch("severe symptoms", "الأعراض الشديدة", 2),
        GlossaryMatch("patients", "المرضى", 1)
    ]
    
    validation_report = ValidationReport(
        is_valid=True,
        glossary_compliance={
            "context_aware": 100.0,
            "terminology_optimized": 100.0,
            "conservative": 100.0
        },
        length_ratios={
            "context_aware": 1.1,
            "terminology_optimized": 1.2,
            "conservative": 1.0
        },
        contains_arabic={
            "context_aware": True,
            "terminology_optimized": True,
            "conservative": True
        }
    )
    
    report = critic.critique_translations(
        translations=translations,
        source_text=source_text,
        glossary=glossary,
        validation_report=validation_report,
        domain="medical"
    )
    
    print("\n" + "="*80)
    print("CRITIQUE REPORT")
    print("="*80)
    print(f"\nRecommended: {report.recommended_variant}")
    print(f"Reason: {report.recommendation_reason}")
    print(f"\nQuality Scores:")
    for agent, score in report.quality_scores.items():
        print(f"  {agent}: {score}/10")
