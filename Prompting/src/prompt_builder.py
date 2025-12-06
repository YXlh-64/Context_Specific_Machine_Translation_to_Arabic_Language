"""Prompt builder for 3 specialized translation agents."""

import logging
from typing import List, Dict
from .models import GlossaryMatch, FuzzyMatch, AgentType

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builder for agent-specific translation prompts."""
    
    @staticmethod
    def _format_glossary_for_context_aware(glossary: List[GlossaryMatch]) -> str:
        """Format glossary for context-aware agent (reference only)."""
        if not glossary:
            return "No specific terminology provided."
        
        lines = ["The following terms have standard translations. Use them as reference:"]
        for match in glossary:
            lines.append(f"- {match.source_term} → {match.target_term}")
        
        lines.append("\nNote: These are recommendations. Prioritize natural expression over exact glossary matching.")
        return "\n".join(lines)
    
    @staticmethod
    def _format_glossary_for_terminology(glossary: List[GlossaryMatch]) -> str:
        """Format glossary for terminology agent (mandatory)."""
        if not glossary:
            return "No specific terminology provided."
        
        lines = ["**MANDATORY TERMINOLOGY** - You MUST use these EXACT translations:"]
        for match in glossary:
            lines.append(f"✓ {match.source_term} → {match.target_term}")
        
        lines.append("\n⚠️ **WARNING**: Using different translations for these terms is NOT acceptable.")
        lines.append("⚠️ **WARNING**: Consistency and terminology accuracy are your TOP priorities.")
        return "\n".join(lines)
    
    @staticmethod
    def _format_glossary_for_conservative(glossary: List[GlossaryMatch]) -> str:
        """Format glossary for conservative agent (accuracy reference)."""
        if not glossary:
            return "No specific terminology provided."
        
        lines = ["Reference terminology for accuracy:"]
        for match in glossary:
            lines.append(f"• {match.source_term} = {match.target_term}")
        
        lines.append("\nUse these terms for accuracy while maintaining literal translation approach.")
        return "\n".join(lines)
    
    @staticmethod
    def _format_fuzzy_matches_for_context_aware(fuzzy_matches: List[FuzzyMatch]) -> str:
        """Format fuzzy matches for context-aware agent (naturalness focus)."""
        if not fuzzy_matches:
            return "No translation examples available."
        
        lines = ["## High-Quality Translation Examples:\n"]
        
        for idx, match in enumerate(fuzzy_matches, 1):
            similarity_pct = int(match.similarity_score * 100)
            lines.append(f"Example {idx} ({similarity_pct}% similar context):")
            lines.append(f'English: "{match.source_text}"')
            lines.append(f'Arabic: "{match.target_text}"')
            lines.append("")
            lines.append("→ Notice: Natural Arabic structure, professional tone, clear and idiomatic")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_fuzzy_matches_for_terminology(fuzzy_matches: List[FuzzyMatch]) -> str:
        """Format fuzzy matches for terminology agent (consistency focus)."""
        if not fuzzy_matches:
            return "No translation examples available."
        
        lines = ["## Reference Translations (Study the terminology consistency):\n"]
        
        for idx, match in enumerate(fuzzy_matches, 1):
            similarity_pct = int(match.similarity_score * 100)
            lines.append(f"Example {idx} ({similarity_pct}% similar):")
            lines.append(f'English: "{match.source_text}"')
            lines.append(f'Arabic: "{match.target_text}"')
            lines.append("")
            lines.append("✓ Notice: Consistent terminology, exact term usage, professional precision")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_fuzzy_matches_for_conservative(fuzzy_matches: List[FuzzyMatch]) -> str:
        """Format fuzzy matches for conservative agent (structure focus)."""
        if not fuzzy_matches:
            return "No translation examples available."
        
        lines = ["## Reference Translations (Note structural preservation):\n"]
        
        for idx, match in enumerate(fuzzy_matches, 1):
            similarity_pct = int(match.similarity_score * 100)
            lines.append(f"Example {idx} ({similarity_pct}% similar):")
            lines.append(f'English: "{match.source_text}"')
            lines.append(f'Arabic: "{match.target_text}"')
            lines.append("")
            lines.append("→ Structure preserved: Word-for-word correspondence, literal approach")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def build_context_aware_prompt(
        source_text: str,
        glossary: List[GlossaryMatch],
        fuzzy_matches: List[FuzzyMatch],
        domain: str
    ) -> str:
        """
        Build context-aware prompt emphasizing natural, fluent Arabic.
        
        Strategy:
        - Format fuzzy matches emphasizing FLUENCY and naturalness
        - Include glossary as REFERENCE (not mandatory)
        - Emphasize: "sound natural, use idiomatic expressions"
        - Include instructions: prioritize readability over literalness
        
        Args:
            source_text: Text to translate
            glossary: Glossary matches
            fuzzy_matches: Similar translation examples
            domain: Domain/field (medical, technical, etc.)
        
        Returns:
            Complete prompt string
        """
        domain_upper = domain.upper()
        
        glossary_section = PromptBuilder._format_glossary_for_context_aware(glossary)
        fuzzy_section = PromptBuilder._format_fuzzy_matches_for_context_aware(fuzzy_matches)
        
        prompt = f"""You are an expert Arabic {domain} translator with 20+ years of experience translating professional content for native Arabic speakers.

# STRATEGY: Natural, Context-Aware Translation

Your goal is to produce a translation that sounds completely natural in Arabic, as if it were originally written in Arabic by a native Arabic-speaking {domain} professional.

## Domain: {domain_upper}

## Key Priorities (in order):
1. **Natural, fluent Arabic expression** - Sound like a native speaker
2. **Professional {domain} tone** - Maintain appropriate register
3. **Readability and clarity** - Easy to understand
4. **Appropriate terminology** - Use domain-standard terms naturally
5. **Accuracy to source meaning** - Preserve core message

## Reference Terminology:
{glossary_section}

{fuzzy_section}

## Your Task:

Translate the following English {domain} text to Arabic.

**Important Instructions:**
- Make it sound natural and fluent in Arabic - this is your TOP priority
- Use contemporary, professional Arabic appropriate for {domain} contexts
- Feel free to adjust sentence structure for clarity and natural flow
- Prioritize readability and professional tone over literal word-for-word translation
- Avoid awkward or overly literal translations that don't sound natural
- Use appropriate pronouns, verb forms, and sentence structures for Arabic
- Think: "How would a native Arabic-speaking {domain} professional write this?"

**English Text to Translate:**
"{source_text}"

**Your Natural Arabic Translation:**
[Provide ONLY the Arabic translation, no explanations, notes, or comments]"""
        
        logger.debug(f"Built context-aware prompt: {len(prompt)} chars")
        return prompt
    
    @staticmethod
    def build_terminology_prompt(
        source_text: str,
        glossary: List[GlossaryMatch],
        fuzzy_matches: List[FuzzyMatch],
        domain: str
    ) -> str:
        """
        Build terminology-optimized prompt emphasizing glossary compliance.
        
        Strategy:
        - Format fuzzy matches highlighting CONSISTENT terminology usage
        - Include glossary as MANDATORY (must use exact translations)
        - Emphasize: "use EXACT glossary terms, never deviate"
        - Include strong warnings about terminology compliance
        
        Args:
            source_text: Text to translate
            glossary: Glossary matches
            fuzzy_matches: Similar translation examples
            domain: Domain/field
        
        Returns:
            Complete prompt string
        """
        domain_upper = domain.upper()
        
        glossary_section = PromptBuilder._format_glossary_for_terminology(glossary)
        fuzzy_section = PromptBuilder._format_fuzzy_matches_for_terminology(fuzzy_matches)
        
        prompt = f"""You are a specialized Arabic {domain} terminology expert with deep expertise in standardized {domain} translation.

# STRATEGY: Terminology-Optimized Translation

Your PRIMARY goal is PERFECT terminology compliance. You must use the EXACT glossary terms provided - no variations, no alternatives, no paraphrasing.

## Domain: {domain_upper}

## Key Priorities (in order):
1. **EXACT glossary term usage** - Use provided terms EXACTLY as given (TOP PRIORITY)
2. **Terminology consistency** - Same term always translated the same way
3. **{domain.title()} accuracy** - Technically/medically precise
4. **Professional register** - Appropriate formality level
5. **Clarity** - Clear and unambiguous meaning

{glossary_section}

{fuzzy_section}

## CRITICAL INSTRUCTIONS:

**TERMINOLOGY RULES (NON-NEGOTIABLE):**
1. ✓ Use EXACT glossary terms - character-for-character match required
2. ✓ NEVER substitute synonyms or alternative translations for glossary terms
3. ✓ NEVER paraphrase or reword glossary terms
4. ✓ Maintain consistency - same source term → same target term throughout
5. ✓ If a glossary term appears in source, its translation MUST appear in your output

**Quality Standards:**
- Terminology compliance is your #1 metric for success
- Professional {domain} register and tone
- Grammatically correct Arabic
- Clear and precise meaning
- Consistency across all terminology

**Your Task:**

Translate the following English {domain} text to Arabic, ensuring PERFECT glossary compliance.

**English Text to Translate:**
"{source_text}"

**Your Terminology-Compliant Arabic Translation:**
[Provide ONLY the Arabic translation with EXACT glossary terms, no explanations or notes]"""
        
        logger.debug(f"Built terminology prompt: {len(prompt)} chars")
        return prompt
    
    @staticmethod
    def build_conservative_prompt(
        source_text: str,
        glossary: List[GlossaryMatch],
        fuzzy_matches: List[FuzzyMatch],
        domain: str
    ) -> str:
        """
        Build conservative (literal) prompt emphasizing structure preservation.
        
        Strategy:
        - Format fuzzy matches showing STRUCTURAL preservation
        - Include glossary for accuracy
        - Emphasize: "translate literally, preserve structure"
        - Include instructions: minimize paraphrasing, stay close to source
        
        Args:
            source_text: Text to translate
            glossary: Glossary matches
            fuzzy_matches: Similar translation examples
            domain: Domain/field
        
        Returns:
            Complete prompt string
        """
        domain_upper = domain.upper()
        
        glossary_section = PromptBuilder._format_glossary_for_conservative(glossary)
        fuzzy_section = PromptBuilder._format_fuzzy_matches_for_conservative(fuzzy_matches)
        
        prompt = f"""You are a precise Arabic {domain} translator specializing in literal, structure-preserving translations.

# STRATEGY: Conservative, Literal Translation

Your goal is to produce a translation that stays as close as possible to the source text structure, providing a literal, word-for-word translation while maintaining grammatical correctness in Arabic.

## Domain: {domain_upper}

## Key Priorities (in order):
1. **Literal translation** - Translate word-for-word as much as possible
2. **Structural preservation** - Keep source sentence structure
3. **Accuracy** - Precise meaning transfer
4. **Minimal interpretation** - Avoid adding or removing meaning
5. **Grammatical correctness** - Ensure valid Arabic grammar

{glossary_section}

{fuzzy_section}

## Translation Approach:

**CONSERVATIVE PRINCIPLES:**
1. Translate as literally as possible while maintaining Arabic grammar
2. Preserve the word order of the source when grammatically feasible
3. Minimize paraphrasing - stay close to source wording
4. Avoid idiomatic expressions unless they appear in source
5. Don't add interpretive elements or explanatory phrases
6. Maintain parallel structure between source and target
7. Use the most direct, literal Arabic equivalent for each word/phrase

**When to Deviate from Literal:**
- Only when required for basic Arabic grammar rules
- Only when literal translation would be grammatically impossible
- Keep deviations minimal and necessary

**Your Task:**

Translate the following English {domain} text to Arabic using a literal, structure-preserving approach.

**English Text to Translate:**
"{source_text}"

**Your Literal Arabic Translation:**
[Provide ONLY the Arabic translation, maintaining source structure as closely as possible, no explanations]"""
        
        logger.debug(f"Built conservative prompt: {len(prompt)} chars")
        return prompt
    
    @staticmethod
    def build_prompts(
        source_text: str,
        glossary: List[GlossaryMatch],
        fuzzy_matches: List[FuzzyMatch],
        domain: str
    ) -> Dict[AgentType, str]:
        """
        Build all three agent prompts.
        
        Args:
            source_text: Text to translate
            glossary: Glossary matches
            fuzzy_matches: Similar translation examples
            domain: Domain/field
        
        Returns:
            Dict mapping AgentType to prompt string
        """
        prompts = {
            AgentType.CONTEXT_AWARE: PromptBuilder.build_context_aware_prompt(
                source_text, glossary, fuzzy_matches, domain
            ),
            AgentType.TERMINOLOGY_OPTIMIZED: PromptBuilder.build_terminology_prompt(
                source_text, glossary, fuzzy_matches, domain
            ),
            AgentType.CONSERVATIVE: PromptBuilder.build_conservative_prompt(
                source_text, glossary, fuzzy_matches, domain
            )
        }
        
        logger.info(f"Built {len(prompts)} prompts for domain: {domain}")
        return prompts


if __name__ == "__main__":
    # Test prompt building
    logging.basicConfig(level=logging.INFO)
    
    from .models import GlossaryMatch, FuzzyMatch
    
    glossary = [
        GlossaryMatch("intensive care units", "وحدات العناية المركزة", 3),
        GlossaryMatch("severe symptoms", "الأعراض الشديدة", 2),
        GlossaryMatch("patients", "المرضى", 1),
        GlossaryMatch("admitted", "يتم إدخالهم", 1)
    ]
    
    fuzzy_matches = [
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
    ]
    
    source_text = "Patients with severe symptoms must be admitted to intensive care units."
    
    # Build context-aware prompt
    prompt = PromptBuilder.build_context_aware_prompt(
        source_text, glossary, fuzzy_matches, "medical"
    )
    
    print("="*80)
    print("CONTEXT-AWARE PROMPT")
    print("="*80)
    print(prompt)
    print(f"\nPrompt length: {len(prompt)} characters")
