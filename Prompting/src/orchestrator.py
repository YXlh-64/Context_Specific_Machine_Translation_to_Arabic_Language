"""Main Translation Orchestrator coordinating the translation pipeline."""

import asyncio
import logging
import time
from typing import Dict

from .models import (
    RAGOutput,
    OrchestrationResult,
    TranslationVariant,
    OrchestratorConfig,
    AgentType,
    LLMRequest,
    ProcessingMetrics
)
from .validator import (
    validate_rag_output,
    analyze_glossary_coverage,
    filter_and_rank_fuzzy_matches,
    validate_translations,
    calculate_glossary_compliance
)
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient
from .critique import TranslationCritic

logger = logging.getLogger(__name__)


class TranslationOrchestrator:
    """
    Main orchestrator coordinating the translation pipeline.
    
    Orchestrates the complete flow:
    1. Validate RAG output
    2. Analyze glossary coverage
    3. Filter and rank fuzzy matches
    4. Build 3 agent-specific prompts
    5. Generate 3 translations (parallel)
    6. Validate translations
    7. Critique and compare translations
    8. Return complete result with recommendations
    """
    
    def __init__(self, llm_client: LLMClient, config: OrchestratorConfig):
        """
        Initialize orchestrator.
        
        Args:
            llm_client: LLM client for API calls
            config: Configuration object
        """
        self.llm_client = llm_client
        self.config = config
        self.prompt_builder = PromptBuilder()
        self.critic = TranslationCritic(config)
        
        logger.info("Translation Orchestrator initialized")
        logger.info(f"Config: model={config.default_model}, "
                   f"parallel={config.enable_parallel_generation}")
    
    async def orchestrate(
        self,
        rag_output: RAGOutput
    ) -> OrchestrationResult:
        """
        Main orchestration method.
        
        Executes complete translation pipeline:
        1. Validate RAG output
        2. Analyze glossary
        3. Filter fuzzy matches
        4. Build 3 prompts
        5. Generate 3 translations (parallel)
        6. Validate translations
        7. Critique translations
        8. Return complete result
        
        Args:
            rag_output: Validated RAG output
        
        Returns:
            OrchestrationResult with all translations and recommendations
        
        Raises:
            Exception: If critical steps fail
        """
        start_time = time.time()
        logger.info(f"Starting orchestration for source: '{rag_output.source_text[:50]}...'")
        
        # STEP 1: Validate RAG output
        logger.info("STEP 1: Validating RAG output")
        validation_start = time.time()
        
        validation_result = validate_rag_output(rag_output, self.config)
        
        if not validation_result.is_valid:
            error_msg = f"RAG validation failed: {validation_result.issues}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            logger.warning(f"Validation warnings: {validation_result.warnings}")
        
        # Use filtered fuzzy matches
        filtered_fuzzy_matches = validation_result.filtered_fuzzy_matches
        
        validation_time = (time.time() - validation_start) * 1000
        logger.info(f"Validation complete: {len(filtered_fuzzy_matches)} fuzzy matches kept")
        
        # STEP 2: Analyze glossary coverage
        logger.info("STEP 2: Analyzing glossary coverage")
        
        glossary_analysis = analyze_glossary_coverage(
            source_text=rag_output.source_text,
            glossary_matches=rag_output.glossary_matches
        )
        
        logger.info(f"Glossary analysis: {glossary_analysis}")
        
        # STEP 3: Filter and rank fuzzy matches
        logger.info("STEP 3: Filtering and ranking fuzzy matches")
        
        ranked_fuzzy_matches = filter_and_rank_fuzzy_matches(
            fuzzy_matches=filtered_fuzzy_matches,
            source_text=rag_output.source_text,
            config=self.config
        )
        
        logger.info(f"Using top {len(ranked_fuzzy_matches)} fuzzy matches")
        
        # STEP 4: Build 3 agent-specific prompts
        logger.info("STEP 4: Building agent-specific prompts")
        prompt_start = time.time()
        
        prompts = self.prompt_builder.build_prompts(
            source_text=rag_output.source_text,
            glossary=rag_output.glossary_matches,
            fuzzy_matches=ranked_fuzzy_matches,
            domain=rag_output.domain
        )
        
        prompt_time = (time.time() - prompt_start) * 1000
        logger.info(f"Built {len(prompts)} prompts")
        
        # STEP 5: Generate 3 translations (parallel)
        logger.info("STEP 5: Generating translations")
        translation_start = time.time()
        
        translations_raw = await self._generate_translations(prompts)
        
        translation_time = (time.time() - translation_start) * 1000
        logger.info(f"Generated {len(translations_raw)} translations in {translation_time:.1f}ms")
        
        # STEP 6: Validate translations
        logger.info("STEP 6: Validating translations")
        
        translation_validation = validate_translations(
            translations=translations_raw,
            glossary=rag_output.glossary_matches,
            source_text=rag_output.source_text,
            config=self.config
        )
        
        if not translation_validation.is_valid:
            logger.warning(f"Translation validation issues: {translation_validation.issues}")
        
        # STEP 7: Critique translations
        logger.info("STEP 7: Critiquing translations")
        critique_start = time.time()
        
        critique_report = self.critic.critique_translations(
            translations=translations_raw,
            source_text=rag_output.source_text,
            glossary=rag_output.glossary_matches,
            validation_report=translation_validation,
            domain=rag_output.domain
        )
        
        critique_time = (time.time() - critique_start) * 1000
        logger.info(f"Critique complete in {critique_time:.1f}ms")
        
        # Build TranslationVariant objects
        translation_variants = {}
        for agent_type, translation_text in translations_raw.items():
            glossary_compliance = translation_validation.glossary_compliance.get(agent_type, 0.0)
            quality_score = critique_report.quality_scores.get(agent_type, 0.0)
            strengths = critique_report.strengths.get(agent_type, [])
            weaknesses = critique_report.weaknesses.get(agent_type, [])
            
            variant = TranslationVariant(
                text=translation_text,
                agent_type=agent_type,
                quality_score=quality_score,
                glossary_compliance=glossary_compliance,
                strengths=strengths,
                weaknesses=weaknesses
            )
            translation_variants[agent_type] = variant
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Build final result
        result = OrchestrationResult(
            source_text=rag_output.source_text,
            domain=rag_output.domain,
            translations=translation_variants,
            recommended_variant=critique_report.recommended_variant,
            key_differences=critique_report.key_differences,
            processing_time_ms=total_time
        )
        
        logger.info(f"Orchestration complete in {total_time:.1f}ms")
        logger.info(f"Recommended variant: {result.recommended_variant}")
        
        return result
    
    async def orchestrate_with_refinement(
        self,
        rag_output: RAGOutput
    ) -> OrchestrationResult:
        """
        Orchestration with iterative refinement loop (matches diagram).
        
        Flow:
        1. Plan: Analyze input and set quality targets
        2. Generate and Validate: Create translations, check quality
        3. Critique and Evaluate: Score translations
        4. Loop: If quality insufficient, regenerate with feedback
        5. Return: Best translations after max iterations or quality met
        
        Args:
            rag_output: Validated RAG output
        
        Returns:
            OrchestrationResult with best translations after refinement
        """
        start_time = time.time()
        iteration = 1
        best_result = None
        previous_scores = {}
        
        logger.info(f"Starting orchestration WITH iterative refinement (max {self.config.max_iterations} iterations)")
        
        while iteration <= self.config.max_iterations:
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{self.config.max_iterations}")
            logger.info(f"{'='*60}")
            
            # STEP 1: PLAN (first iteration only)
            if iteration == 1:
                logger.info("STEP 1: Planning - Analyzing input and setting targets")
                plan = self._create_plan(rag_output)
                logger.info(f"Plan: target_quality={plan['target_quality']}, focus_areas={plan['focus_areas']}")
            
            # STEP 2: GENERATE AND VALIDATE
            logger.info(f"STEP 2: Generate and Validate (iteration {iteration})")
            
            # Add feedback from previous iteration
            feedback = self._create_feedback(best_result, previous_scores) if iteration > 1 else None
            if feedback:
                logger.info(f"Feedback: {len(feedback.get('suggestions', []))} suggestions")
            
            # Generate translations
            result = await self.orchestrate(rag_output)
            
            # STEP 3: CRITIQUE AND EVALUATE
            logger.info(f"STEP 3: Critique and Evaluate")
            
            # Check if quality targets met
            quality_check = self._check_quality_targets(result)
            logger.info(f"Quality check: {quality_check['quality_pass_count']}/3 quality, {quality_check['glossary_pass_count']}/3 glossary")
            
            # Track scores for improvement detection
            current_scores = {
                agent: variant.quality_score 
                for agent, variant in result.translations.items()
            }
            
            # DECISION POINT: Continue loop or exit?
            if quality_check['targets_met']:
                logger.info(f"✅ Quality targets met on iteration {iteration}")
                best_result = result
                break
            
            # Check if we're improving
            if iteration > 1:
                improvement = self._calculate_improvement(previous_scores, current_scores)
                logger.info(f"Improvement from previous iteration: {improvement:.2f}")
                
                if improvement < self.config.improvement_threshold:
                    logger.info(f"⚠️ Insufficient improvement ({improvement:.2f} < {self.config.improvement_threshold}), stopping")
                    best_result = result if not best_result else best_result
                    break
            
            # Save best result so far
            if not best_result or self._is_better(result, best_result):
                best_result = result
                logger.info(f"💾 Saved as best result so far")
            
            previous_scores = current_scores
            iteration += 1
            
            if iteration <= self.config.max_iterations:
                logger.info(f"🔄 Loop: Regenerating with feedback...")
                await asyncio.sleep(1)  # Brief pause before next iteration
        
        # Add iteration metadata
        end_time = time.time()
        best_result.processing_time_ms = (end_time - start_time) * 1000
        best_result.metadata = {
            'iterations': iteration - 1,
            'refinement_enabled': True,
            'final_quality_met': quality_check.get('targets_met', False) if best_result else False
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Orchestration complete after {iteration-1} iterations")
        logger.info(f"Total time: {best_result.processing_time_ms:.0f}ms")
        logger.info(f"{'='*60}\n")
        
        return best_result
    
    def _create_plan(self, rag_output: RAGOutput) -> dict:
        """
        STEP 1: Create plan based on input analysis.
        """
        return {
            'target_quality': self.config.min_quality_threshold,
            'target_glossary_compliance': self.config.min_glossary_compliance,
            'focus_areas': [
                'terminology' if len(rag_output.glossary_matches) > 5 else 'fluency',
                'accuracy' if rag_output.domain in ['medical', 'legal'] else 'readability'
            ],
            'iteration_strategy': 'iterative_refinement' if self.config.enable_iterative_refinement else 'single_pass'
        }
    
    def _create_feedback(self, previous_result: OrchestrationResult, previous_scores: dict) -> dict:
        """
        Create feedback for next iteration based on previous attempt.
        """
        if not previous_result:
            return None
        
        feedback = {
            'previous_scores': previous_scores,
            'weaknesses': [],
            'suggestions': []
        }
        
        # Analyze each variant's weaknesses
        for agent_type, variant in previous_result.translations.items():
            if variant.quality_score < self.config.min_quality_threshold:
                feedback['weaknesses'].extend([
                    f"{agent_type}: {w}" for w in variant.weaknesses[:2]
                ])
                
                # Add specific suggestions
                if variant.glossary_compliance < self.config.min_glossary_compliance:
                    feedback['suggestions'].append(
                        f"{agent_type}: Focus on using exact glossary terms"
                    )
                
                if "unnatural" in ' '.join(variant.weaknesses).lower():
                    feedback['suggestions'].append(
                        f"{agent_type}: Improve fluency and naturalness"
                    )
        
        return feedback
    
    def _check_quality_targets(self, result: OrchestrationResult) -> dict:
        """
        Check if quality targets are met.
        """
        checks = {
            'quality_met': [],
            'glossary_met': [],
            'targets_met': False
        }
        
        for agent_type, variant in result.translations.items():
            quality_ok = variant.quality_score >= self.config.min_quality_threshold
            glossary_ok = variant.glossary_compliance >= self.config.min_glossary_compliance
            
            checks['quality_met'].append(quality_ok)
            checks['glossary_met'].append(glossary_ok)
        
        # Consider targets met if at least 2 out of 3 variants pass
        quality_pass_count = sum(checks['quality_met'])
        glossary_pass_count = sum(checks['glossary_met'])
        
        checks['targets_met'] = quality_pass_count >= 2 and glossary_pass_count >= 2
        checks['quality_pass_count'] = quality_pass_count
        checks['glossary_pass_count'] = glossary_pass_count
        
        return checks
    
    def _calculate_improvement(self, previous_scores: dict, current_scores: dict) -> float:
        """
        Calculate average improvement from previous iteration.
        """
        improvements = []
        for agent_type in previous_scores:
            if agent_type in current_scores:
                improvement = current_scores[agent_type] - previous_scores[agent_type]
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _is_better(self, result1: OrchestrationResult, result2: OrchestrationResult) -> bool:
        """
        Compare two results and determine if result1 is better.
        """
        # Calculate average quality scores
        avg1 = sum(v.quality_score for v in result1.translations.values()) / len(result1.translations)
        avg2 = sum(v.quality_score for v in result2.translations.values()) / len(result2.translations)
        
        # Calculate average glossary compliance
        gloss1 = sum(v.glossary_compliance for v in result1.translations.values()) / len(result1.translations)
        gloss2 = sum(v.glossary_compliance for v in result2.translations.values()) / len(result2.translations)
        
        # Weighted comparison: 60% quality, 40% glossary
        score1 = avg1 * 0.6 + gloss1 * 0.4
        score2 = avg2 * 0.6 + gloss2 * 0.4
        
        return score1 > score2
    
    async def _generate_translations(
        self,
        prompts: Dict[AgentType, str]
    ) -> Dict[str, str]:
        """
        Generate 3 translations in parallel.
        
        Uses different temperature per agent:
        - context_aware: 0.3
        - terminology_optimized: 0.2
        - conservative: 0.1
        
        Args:
            prompts: Dict mapping AgentType to prompt string
        
        Returns:
            Dict mapping agent_type (str) to translation text
        """
        # Build LLM requests
        requests = {}
        
        for agent_type, prompt in prompts.items():
            # Get temperature based on agent type
            if agent_type == AgentType.CONTEXT_AWARE:
                temperature = self.config.context_aware_temperature
            elif agent_type == AgentType.TERMINOLOGY_OPTIMIZED:
                temperature = self.config.terminology_temperature
            elif agent_type == AgentType.CONSERVATIVE:
                temperature = self.config.conservative_temperature
            else:
                temperature = 0.3
            
            requests[agent_type] = LLMRequest(
                prompt=prompt,
                agent_type=agent_type,
                temperature=temperature,
                model=self.config.default_model,
                max_tokens=self.config.max_tokens
            )
        
        # Execute parallel or sequential based on config
        if self.config.enable_parallel_generation:
            logger.info("Generating translations in parallel")
            responses = await self.llm_client.complete_multiple(requests)
        else:
            logger.info("Generating translations sequentially")
            responses = {}
            for agent_type, request in requests.items():
                response = await self.llm_client.complete(
                    prompt=request.prompt,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    agent_type=agent_type
                )
                responses[agent_type] = response
        
        # Extract text from responses
        translations = {}
        for agent_type, response in responses.items():
            translations[agent_type.value] = response.text
        
        return translations
    
    async def orchestrate_batch(
        self,
        rag_outputs: list[RAGOutput]
    ) -> list[OrchestrationResult]:
        """
        Process multiple RAG outputs in batch.
        
        Args:
            rag_outputs: List of RAG outputs to process
        
        Returns:
            List of OrchestrationResult objects
        """
        logger.info(f"Starting batch orchestration for {len(rag_outputs)} items")
        
        results = []
        for idx, rag_output in enumerate(rag_outputs):
            logger.info(f"Processing batch item {idx + 1}/{len(rag_outputs)}")
            try:
                result = await self.orchestrate(rag_output)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item {idx + 1} failed: {e}")
                # Continue with next item
        
        logger.info(f"Batch orchestration complete: {len(results)}/{len(rag_outputs)} succeeded")
        return results


async def orchestrate_from_csv(
    csv_path: str,
    llm_client: LLMClient,
    config: OrchestratorConfig
) -> list[OrchestrationResult]:
    """
    Convenience function to orchestrate directly from CSV file.
    
    Args:
        csv_path: Path to RAG CSV file
        llm_client: LLM client
        config: Configuration
    
    Returns:
        List of OrchestrationResult objects
    """
    from .parser import parse_rag_csv
    
    logger.info(f"Loading RAG outputs from {csv_path}")
    rag_outputs = parse_rag_csv(csv_path)
    
    orchestrator = TranslationOrchestrator(llm_client, config)
    results = await orchestrator.orchestrate_batch(rag_outputs)
    
    return results


if __name__ == "__main__":
    # Test orchestrator
    import asyncio
    
    async def test_orchestrator():
        logging.basicConfig(level=logging.INFO)
        
        from .models import GlossaryMatch, FuzzyMatch, RAGOutput, OrchestratorConfig
        from .llm_client import MockLLMClient
        
        # Create test data
        rag_output = RAGOutput(
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
        
        # Initialize orchestrator with mock client
        config = OrchestratorConfig()
        llm_client = MockLLMClient()
        orchestrator = TranslationOrchestrator(llm_client, config)
        
        # Run orchestration
        result = await orchestrator.orchestrate(rag_output)
        
        # Print results
        print("\n" + "="*80)
        print("ORCHESTRATION RESULT")
        print("="*80)
        print(f"\nSource: {result.source_text}")
        print(f"Domain: {result.domain}")
        print(f"\n3 Translation Variants:")
        
        for agent_type, variant in result.translations.items():
            print(f"\n{agent_type}:")
            print(f"  Translation: {variant.text}")
            print(f"  Quality: {variant.quality_score}/10")
            print(f"  Glossary Compliance: {variant.glossary_compliance}%")
            print(f"  Strengths: {', '.join(variant.strengths[:2])}")
            print(f"  Weaknesses: {', '.join(variant.weaknesses[:2])}")
        
        print(f"\n{'='*80}")
        print(f"Recommended: {result.recommended_variant}")
        print(f"Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"{'='*80}")
    
    asyncio.run(test_orchestrator())
