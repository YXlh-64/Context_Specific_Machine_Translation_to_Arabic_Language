"""
Prompt Construction Service
Core logic for assembling translation prompts from glossary and fuzzy matches
"""

import logging
from typing import List, Dict, Optional, Tuple
from jinja2 import Template

from app.core.config import settings, get_language_name
from app.models.schemas import (
    GlossaryTerm, 
    FuzzyMatch,
    PromptConstructionRequest,
    PromptConstructionResponse,
    PromptFormat
)

logger = logging.getLogger(__name__)


class PromptConstructor:
    """
    Constructs translation prompts following the Phase 3 specification.
    
    Prompt Structure:
    1. Glossary Terms (mandatory terminology)
    2. Similar Examples (fuzzy matches from translation memory)
    3. Source Sentence with repeated glossary terms
    """
    
    # Jinja2 template for the main prompt
    PROMPT_TEMPLATE = """{% if terms_str %}Terms: {{ terms_str }}

{% endif %}{% for example in examples %}{{ source_lang_name }}: {{ example.source }}
{{ target_lang_name }}: {{ example.target }}

{% endfor %}{% if terms_str %}Terms: {{ terms_str }}
{% endif %}{{ source_lang_name }}: {{ source_sentence }}
{{ target_lang_name }}:"""

    # System message template
    SYSTEM_MESSAGE_TEMPLATE = """You are a specialized {{ domain }} translator for {{ source_lang_name }}↔{{ target_lang_name }} translation.
Use the provided glossary terms EXACTLY as shown - they are mandatory.
Follow the translation style demonstrated in the examples.
{{ tone_instruction }}
Output ONLY the translation, no explanations or commentary."""

    def __init__(self):
        self.prompt_template = Template(self.PROMPT_TEMPLATE)
        self.system_template = Template(self.SYSTEM_MESSAGE_TEMPLATE)
    
    def format_glossary_terms(
        self, 
        glossary_matches: List[GlossaryTerm],
        max_terms: int = None
    ) -> str:
        """
        Format glossary terms into the required string format.
        
        Format: term1_source=term1_target - term2_source=term2_target - ...
        
        Args:
            glossary_matches: List of glossary term matches
            max_terms: Maximum number of terms to include
            
        Returns:
            Formatted terms string
        """
        if not glossary_matches:
            return ""
        
        max_terms = max_terms or settings.MAX_GLOSSARY_TERMS
        terms = glossary_matches[:max_terms]
        
        formatted = [
            f"{term.source_term}={term.target_term}"
            for term in terms
        ]
        
        return " - ".join(formatted)
    
    def format_fuzzy_examples(
        self,
        fuzzy_matches: List[FuzzyMatch],
        source_lang: str = "en",
        target_lang: str = "ar",
        max_examples: int = None
    ) -> List[Dict[str, str]]:
        """
        Format fuzzy matches into example pairs.
        
        Args:
            fuzzy_matches: List of fuzzy matches (should be sorted by similarity)
            source_lang: Source language code
            target_lang: Target language code
            max_examples: Maximum examples to include
            
        Returns:
            List of formatted example dictionaries
        """
        if not fuzzy_matches:
            return []
        
        max_examples = max_examples or settings.MAX_FUZZY_MATCHES
        matches = fuzzy_matches[:max_examples]
        
        examples = []
        for match in matches:
            examples.append({
                "source": match.source.strip(),
                "target": match.target.strip()
            })
        
        return examples
    
    def get_domain_tone(self, domain: str) -> str:
        """Get tone instruction for domain"""
        tone = settings.DOMAIN_TONES.get(
            domain.lower(), 
            settings.DOMAIN_TONES["general"]
        )
        return f"Use {tone.lower()}."
    
    def build_system_message(
        self,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Build the system message for LLM.
        
        Args:
            domain: Translation domain
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Formatted system message
        """
        return self.system_template.render(
            domain=domain,
            source_lang_name=get_language_name(source_lang),
            target_lang_name=get_language_name(target_lang),
            tone_instruction=self.get_domain_tone(domain)
        )
    
    def construct_prompt(
        self,
        source_sentence: str,
        glossary_matches: List[GlossaryTerm],
        fuzzy_matches: List[FuzzyMatch],
        source_lang: str = "en",
        target_lang: str = "ar",
        max_terms: int = None,
        max_examples: int = None
    ) -> str:
        """
        Construct the full translation prompt.
        
        Follows the structure:
        1. Terms (glossary)
        2. Examples (fuzzy matches)
        3. Terms (repeated)
        4. Source sentence
        5. Target language marker (blank for LLM to complete)
        
        Args:
            source_sentence: Sentence to translate
            glossary_matches: Glossary terms from Phase 1
            fuzzy_matches: Fuzzy matches from Phase 2
            source_lang: Source language code
            target_lang: Target language code
            max_terms: Max glossary terms
            max_examples: Max fuzzy examples
            
        Returns:
            Complete prompt string
        """
        # Format glossary terms
        terms_str = self.format_glossary_terms(glossary_matches, max_terms)
        
        # Format fuzzy examples
        examples = self.format_fuzzy_examples(
            fuzzy_matches, source_lang, target_lang, max_examples
        )
        
        # Get language names
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        
        # Render prompt
        prompt = self.prompt_template.render(
            terms_str=terms_str,
            examples=examples,
            source_lang_name=source_lang_name,
            target_lang_name=target_lang_name,
            source_sentence=source_sentence.strip()
        )
        
        return prompt
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Uses rough approximation: ~4 chars per token for English,
        ~2 chars per token for Arabic/other scripts.
        
        For production, use tiktoken for accurate counts.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough estimation
            # Mix of English (~4 chars/token) and Arabic (~2 chars/token)
            return len(text) // 3
    
    def format_as_xml(
        self,
        source_sentence: str,
        glossary_matches: List[GlossaryTerm],
        fuzzy_matches: List[FuzzyMatch],
        source_lang: str = "en",
        target_lang: str = "ar",
        max_terms: int = None,
        max_examples: int = None
    ) -> str:
        """Format prompt as XML."""
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        
        terms = glossary_matches[:max_terms or settings.MAX_GLOSSARY_TERMS]
        examples = fuzzy_matches[:max_examples or settings.MAX_FUZZY_MATCHES]
        
        lines = ["<translation_task>"]
        
        if terms:
            lines.append("  <glossary>")
            for t in terms:
                lines.append(f'    <term source="{t.source_term}" target="{t.target_term}"/>')
            lines.append("  </glossary>")
        
        if examples:
            lines.append("  <examples>")
            for ex in examples:
                lines.append("    <example>")
                lines.append(f"      <source>{ex.source}</source>")
                lines.append(f"      <target>{ex.target}</target>")
                lines.append("    </example>")
            lines.append("  </examples>")
        
        lines.append(f"  <source_sentence>{source_sentence}</source_sentence>")
        lines.append(f"  <target_language>{target_lang_name}</target_language>")
        lines.append("</translation_task>")
        
        return "\n".join(lines)
    
    def format_as_json(
        self,
        source_sentence: str,
        glossary_matches: List[GlossaryTerm],
        fuzzy_matches: List[FuzzyMatch],
        source_lang: str = "en",
        target_lang: str = "ar",
        max_terms: int = None,
        max_examples: int = None
    ) -> str:
        """Format prompt as JSON."""
        import json
        
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        
        terms = glossary_matches[:max_terms or settings.MAX_GLOSSARY_TERMS]
        examples = fuzzy_matches[:max_examples or settings.MAX_FUZZY_MATCHES]
        
        data = {
            "glossary": [
                {"source": t.source_term, "target": t.target_term}
                for t in terms
            ],
            "examples": [
                {"source": ex.source, "target": ex.target}
                for ex in examples
            ],
            "source_sentence": source_sentence,
            "target_language": target_lang_name
        }
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def format_as_markdown(
        self,
        source_sentence: str,
        glossary_matches: List[GlossaryTerm],
        fuzzy_matches: List[FuzzyMatch],
        source_lang: str = "en",
        target_lang: str = "ar",
        max_terms: int = None,
        max_examples: int = None
    ) -> str:
        """Format prompt as Markdown."""
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        
        terms = glossary_matches[:max_terms or settings.MAX_GLOSSARY_TERMS]
        examples = fuzzy_matches[:max_examples or settings.MAX_FUZZY_MATCHES]
        
        lines = ["# Translation Task", ""]
        
        if terms:
            lines.append("## Glossary Terms")
            lines.append("| Source | Target |")
            lines.append("|--------|--------|")
            for t in terms:
                lines.append(f"| {t.source_term} | {t.target_term} |")
            lines.append("")
        
        if examples:
            lines.append("## Similar Examples")
            for i, ex in enumerate(examples, 1):
                lines.append(f"### Example {i}")
                lines.append(f"**{source_lang_name}:** {ex.source}")
                lines.append(f"**{target_lang_name}:** {ex.target}")
                lines.append("")
        
        lines.append("## Source Sentence")
        lines.append(f"**{source_lang_name}:** {source_sentence}")
        lines.append("")
        lines.append(f"**Target ({target_lang_name}):** _[Translate here]_")
        
        return "\n".join(lines)
    
    def format_as_plain(
        self,
        source_sentence: str,
        glossary_matches: List[GlossaryTerm],
        fuzzy_matches: List[FuzzyMatch],
        source_lang: str = "en",
        target_lang: str = "ar",
        max_terms: int = None,
        max_examples: int = None
    ) -> str:
        """Format prompt as plain text."""
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        
        terms = glossary_matches[:max_terms or settings.MAX_GLOSSARY_TERMS]
        examples = fuzzy_matches[:max_examples or settings.MAX_FUZZY_MATCHES]
        
        lines = ["TRANSLATION TASK", "=" * 40, ""]
        
        if terms:
            lines.append("GLOSSARY TERMS:")
            for t in terms:
                lines.append(f"  - {t.source_term} -> {t.target_term}")
            lines.append("")
        
        if examples:
            lines.append("SIMILAR EXAMPLES:")
            for i, ex in enumerate(examples, 1):
                lines.append(f"  Example {i}:")
                lines.append(f"    {source_lang_name}: {ex.source}")
                lines.append(f"    {target_lang_name}: {ex.target}")
            lines.append("")
        
        lines.append("SOURCE SENTENCE:")
        lines.append(f"  {source_lang_name}: {source_sentence}")
        lines.append("")
        lines.append(f"TRANSLATION ({target_lang_name}):")
        
        return "\n".join(lines)

    def construct(self, request: PromptConstructionRequest) -> PromptConstructionResponse:
        """
        Main entry point for prompt construction.
        
        Args:
            request: PromptConstructionRequest with all inputs
            
        Returns:
            PromptConstructionResponse with constructed prompt
        """
        # Get the prompt format
        prompt_format = request.prompt_format
        
        # Build prompt based on format
        if prompt_format == PromptFormat.XML:
            prompt = self.format_as_xml(
                source_sentence=request.source_sentence,
                glossary_matches=request.glossary_matches,
                fuzzy_matches=request.fuzzy_matches,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                max_terms=request.max_terms,
                max_examples=request.max_examples
            )
        elif prompt_format == PromptFormat.JSON:
            prompt = self.format_as_json(
                source_sentence=request.source_sentence,
                glossary_matches=request.glossary_matches,
                fuzzy_matches=request.fuzzy_matches,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                max_terms=request.max_terms,
                max_examples=request.max_examples
            )
        elif prompt_format == PromptFormat.MARKDOWN:
            prompt = self.format_as_markdown(
                source_sentence=request.source_sentence,
                glossary_matches=request.glossary_matches,
                fuzzy_matches=request.fuzzy_matches,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                max_terms=request.max_terms,
                max_examples=request.max_examples
            )
        else:  # PLAIN
            prompt = self.format_as_plain(
                source_sentence=request.source_sentence,
                glossary_matches=request.glossary_matches,
                fuzzy_matches=request.fuzzy_matches,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                max_terms=request.max_terms,
                max_examples=request.max_examples
            )
        
        # Build system message if requested
        system_message = None
        if request.include_system_message:
            system_message = self.build_system_message(
                domain=request.domain,
                source_lang=request.source_lang,
                target_lang=request.target_lang
            )
        
        # Estimate tokens
        total_text = (system_message or "") + prompt
        token_count = self.estimate_tokens(total_text)
        
        # Count actual usage
        glossary_used = min(
            len(request.glossary_matches),
            request.max_terms or settings.MAX_GLOSSARY_TERMS
        )
        fuzzy_used = min(
            len(request.fuzzy_matches),
            request.max_examples or settings.MAX_FUZZY_MATCHES
        )
        
        return PromptConstructionResponse(
            prompt=prompt,
            system_message=system_message,
            format=prompt_format,
            token_count=token_count,
            glossary_terms_used=glossary_used,
            fuzzy_matches_used=fuzzy_used,
            domain=request.domain,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            metadata={
                "max_terms_limit": request.max_terms or settings.MAX_GLOSSARY_TERMS,
                "max_examples_limit": request.max_examples or settings.MAX_FUZZY_MATCHES
            }
        )


# Global instance
prompt_constructor = PromptConstructor()


def construct_translation_prompt(
    source_sentence: str,
    glossary_matches: List[Dict],
    fuzzy_matches: List[Dict],
    domain: str = "general",
    source_lang: str = "en",
    target_lang: str = "ar"
) -> Tuple[str, str]:
    """
    Convenience function to construct prompt.
    
    Args:
        source_sentence: Sentence to translate
        glossary_matches: List of glossary match dicts
        fuzzy_matches: List of fuzzy match dicts
        domain: Translation domain
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        Tuple of (prompt, system_message)
    """
    # Convert dicts to models
    glossary_terms = [GlossaryTerm(**m) for m in glossary_matches]
    fuzzy_examples = [FuzzyMatch(**m) for m in fuzzy_matches]
    
    request = PromptConstructionRequest(
        source_sentence=source_sentence,
        glossary_matches=glossary_terms,
        fuzzy_matches=fuzzy_examples,
        domain=domain,
        source_lang=source_lang,
        target_lang=target_lang,
        include_system_message=True
    )
    
    response = prompt_constructor.construct(request)
    return response.prompt, response.system_message
