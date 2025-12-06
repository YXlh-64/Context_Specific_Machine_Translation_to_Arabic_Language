"""
TRANSLATION ORCHESTRATOR DEMO - With Official Evaluation
========================================================================

This demonstrates the 3-translation generation task WITH proper evaluation metrics.

OFFICIAL EVALUATION DATA:
- Evaluation/Augmented Queries Samples/{domain}_examples.csv (RAG team queries)
- Evaluation/Augmented Queries Samples/{domain}_translation_template.csv (Ground truth)

EVALUATION METRICS:
- chrF++ (Character-level F-score, 60% weight)
- BLEU (Word-level precision, 40% weight)
- COMET (Neural semantic similarity, 50% weight when available)

HOW TO USE:
1. Choose domain: medical, technology, education, or economic
2. Choose task number (1-20 for medical, varies by domain)
3. Run: python demo.py
4. See 3 translations with official evaluation scores
"""

import asyncio
import logging
import os
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Hide verbose logs
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

from src.orchestrator import TranslationOrchestrator
from src.parser import parse_rag_csv
from src.llm_client import LLMClient
from src.models import OrchestratorConfig
from Evaluation.translation_evaluator import TranslationEvaluator


# ============================================================================
# STEP 1: CHOOSE YOUR INPUT (Official evaluation data)
# ============================================================================
# AVAILABLE DOMAINS: medical, technology, education, economic
# Each domain has:
#   - {domain}_examples.csv: RAG team's augmented queries (inputs)
#   - {domain}_translation_template.csv: Ground truth translations (for evaluation)

DOMAIN = "medical"  # or "technology", "education", "economic"
TASK_NUMBER = 10   # Which task from the examples CSV (1-20 for medical)

# ITERATIVE REFINEMENT SETTINGS (matches diagram loop)
ENABLE_ITERATIVE_REFINEMENT = True  # Set to False for single-pass generation
MAX_ITERATIONS = 3  # Maximum refinement cycles
MIN_QUALITY_THRESHOLD = 7.5  # Minimum quality score (0-10) to accept
MIN_GLOSSARY_COMPLIANCE = 85.0  # Minimum glossary compliance (%) to accept

# ============================================================================

def load_ground_truth(domain):
    """Load ground truth translations from template CSV."""
    template_file = f"Evaluation/Augmented Queries Samples/{domain}_translation_template.csv"
    
    ground_truths = {}
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                ground_truths[i] = {
                    'english': row['English'],
                    'arabic': row['Arabic']
                }
        return ground_truths
    except FileNotFoundError:
        print(f"⚠️ Warning: Ground truth file not found: {template_file}")
        return {}
    except Exception as e:
        print(f"⚠️ Warning: Could not load ground truth: {e}")
        return {}


async def demonstrate_translation():
    """
    Demonstrate the 3-translation generation task with official evaluation metrics.
    """
    
    # Setup file paths
    examples_csv = f"Evaluation/Augmented Queries Samples/{DOMAIN}_examples.csv"
    
    print("\n" + "="*80)
    print("🎯 TRANSLATION ORCHESTRATOR - LEADERSHIP DEMO WITH OFFICIAL EVALUATION")
    print("="*80)
    print(f"\nTask: Generate 3 translation variants with chrF++, BLEU, COMET metrics")
    print(f"Domain: {DOMAIN.upper()}")
    print(f"Input: {examples_csv}")
    print(f"Model: Groq LLaMA 3.3 (FREE)")
    
    # Check environment
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ Setup Required:")
        print("1. Visit: https://console.groq.com/keys (FREE signup)")
        print("2. Create API key")
        print("3. Add to .env file: GROQ_API_KEY=your_key_here")
        print("4. Run demo again")
        return
    
    # Load ground truth translations
    print(f"\n📂 Loading ground truth translations...")
    ground_truths = load_ground_truth(DOMAIN)
    if not ground_truths:
        print("⚠️ Warning: No ground truth available, evaluation will be limited")
    else:
        print(f"✅ Loaded {len(ground_truths)} ground truth translations")
    
    # Initialize evaluator
    print(f"\n🔧 Initializing official evaluation system...")
    try:
        evaluator = TranslationEvaluator()
    except Exception as e:
        print(f"⚠️ Evaluator initialization issue: {e}")
        evaluator = None
    
    # Parse CSV from RAG team
    print(f"\n📂 Reading RAG team's augmented queries...")
    try:
        rag_outputs = parse_rag_csv(examples_csv)
        print(f"✅ Found {len(rag_outputs)} translation tasks in CSV")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Select the task
    if TASK_NUMBER < 1 or TASK_NUMBER > len(rag_outputs):
        print(f"❌ Invalid task number. Choose 1-{len(rag_outputs)}")
        return
    
    rag_output = rag_outputs[TASK_NUMBER - 1]
    ground_truth = ground_truths.get(TASK_NUMBER, {})
    
    # Display input from RAG team
    print("\n" + "="*80)
    print("📥 INPUT FROM RAG TEAM")
    print("="*80)
    print(f"\n📝 Sentence to translate:")
    print(f"   {rag_output.source_text}")
    print(f"\n🏷️  Domain: {rag_output.domain}")
    print(f"📚 Glossary terms: {len(rag_output.glossary_matches)}")
    for term in rag_output.glossary_matches:
        print(f"   • {term.source_term} = {term.target_term}")
    print(f"📖 Similar examples: {len(rag_output.fuzzy_matches)}")
    for i, match in enumerate(rag_output.fuzzy_matches[:2], 1):
        print(f"   {i}. {match.source_text[:50]}... (similarity: {match.similarity_score:.0%})")
    
    # Setup orchestrator with refinement settings
    config = OrchestratorConfig(
        default_model="llama-3.3-70b-versatile",
        enable_iterative_refinement=ENABLE_ITERATIVE_REFINEMENT,
        max_iterations=MAX_ITERATIONS,
        min_quality_threshold=MIN_QUALITY_THRESHOLD,
        min_glossary_compliance=MIN_GLOSSARY_COMPLIANCE
    )
    llm_client = LLMClient()
    orchestrator = TranslationOrchestrator(llm_client, config)
    
    # Generate 3 translations (with or without refinement)
    print("\n" + "="*80)
    if ENABLE_ITERATIVE_REFINEMENT:
        print("🔄 GENERATING WITH ITERATIVE REFINEMENT")
        print(f"   Max iterations: {MAX_ITERATIONS}")
        print(f"   Quality target: {MIN_QUALITY_THRESHOLD}/10")
        print(f"   Glossary target: {MIN_GLOSSARY_COMPLIANCE}%")
    else:
        print("🚀 GENERATING 3 TRANSLATION VARIANTS (SINGLE PASS)")
    print("="*80)
    print("⏳ Please wait (5-10 seconds per iteration)...")
    
    try:
        if ENABLE_ITERATIVE_REFINEMENT:
            result = await orchestrator.orchestrate_with_refinement(rag_output)
        else:
            result = await orchestrator.orchestrate(rag_output)
    except Exception as e:
        print(f"\n❌ Translation failed: {e}")
        return
    
    # Display results
    print("\n" + "="*80)
    if ENABLE_ITERATIVE_REFINEMENT and result.metadata:
        print(f"✅ SUCCESS - COMPLETED IN {result.metadata.get('iterations', 1)} ITERATION(S)")
        print(f"   Quality targets met: {'Yes ✓' if result.metadata.get('final_quality_met') else 'No (max iterations reached)'}")
    else:
        print("✅ SUCCESS - 3 TRANSLATIONS GENERATED")
    print("="*80)
    print(f"\n⚡ Processing time: {result.processing_time_ms:.0f}ms")
    
    # Evaluate translations with official metrics
    evaluation_results = {}
    if evaluator and ground_truth.get('arabic'):
        print("\n" + "="*80)
        print("📊 EVALUATING TRANSLATIONS WITH OFFICIAL METRICS")
        print("="*80)
        print(f"Reference (Ground Truth): {ground_truth['arabic']}")
        
        for agent_type, variant in result.translations.items():
            print(f"\n⏳ Evaluating {agent_type}...")
            try:
                eval_result = evaluator.evaluate(
                    source=rag_output.source_text,
                    hypothesis=variant.text,
                    reference=ground_truth['arabic']
                )
                evaluation_results[agent_type] = eval_result
                print(f"   ✅ chrF++: {eval_result['chrf_score']:.2f}/100")
                print(f"   ✅ BLEU: {eval_result['bleu_score']:.2f}/100")
                if eval_result.get('comet_score') is not None:
                    print(f"   ✅ COMET: {eval_result['comet_score']:.4f}")
                print(f"   ⭐ Combined: {eval_result['combined_percentage']:.2f}% ({eval_result['quality_level']})")
            except Exception as e:
                print(f"   ⚠️ Evaluation failed: {e}")
                evaluation_results[agent_type] = None
    
    # Show all 3 variants with evaluation scores
    print("\n" + "="*80)
    print("📋 THE 3 TRANSLATION VARIANTS WITH OFFICIAL EVALUATION")
    print("="*80)
    
    # Sort by official combined score if available, otherwise by quality score
    if evaluation_results:
        sorted_translations = sorted(
            result.translations.items(),
            key=lambda x: evaluation_results.get(x[0], {}).get('combined_score', x[1].quality_score / 10),
            reverse=True
        )
    else:
        sorted_translations = sorted(
            result.translations.items(),
            key=lambda x: x[1].quality_score,
            reverse=True
        )
    
    for rank, (agent_type, variant) in enumerate(sorted_translations, 1):
        eval_result = evaluation_results.get(agent_type)
        
        print(f"\n{'='*80}")
        print(f"VARIANT #{rank}: {agent_type.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        print(f"\n🌐 Arabic Translation:")
        print(f"   {variant.text}")
        
        # Show official metrics if available
        if eval_result:
            print(f"\n📊 OFFICIAL EVALUATION METRICS:")
            print(f"   chrF++ Score: {eval_result['chrf_score']:.2f}/100 (weight: 60%)")
            print(f"   BLEU Score: {eval_result['bleu_score']:.2f}/100 (weight: 40%)")
            if eval_result.get('comet_score') is not None:
                print(f"   COMET Score: {eval_result['comet_score']:.4f} (weight: 50%)")
            print(f"   📈 Combined Score: {eval_result['combined_percentage']:.2f}%")
            print(f"   🏆 Quality Level: {eval_result['quality_level']}")
        
        # Show orchestrator's internal scores
        print(f"\n⭐ Internal Quality Score: {variant.quality_score}/10")
        print(f"📊 Glossary Compliance: {variant.glossary_compliance}%")
        
        if variant.strengths:
            print(f"\n✓ Strengths:")
            for strength in variant.strengths[:3]:
                print(f"   • {strength}")
        
        if variant.weaknesses:
            print(f"\n⚠ Weaknesses:")
            for weakness in variant.weaknesses[:2]:
                print(f"   • {weakness}")
    
    # Show recommendation
    print("\n" + "="*80)
    print(f"🏆 RECOMMENDED VARIANT: {result.recommended_variant.upper().replace('_', ' ')}")
    print("="*80)
    
    best = result.translations[result.recommended_variant]
    print(f"\nBest Translation: {best.text}")
    print(f"Quality: {best.quality_score}/10")
    print(f"Glossary: {best.glossary_compliance}%")
    
    if result.key_differences:
        print(f"\n📊 Key Differences Between Variants:")
        for diff in result.key_differences[:3]:
            print(f"   • {diff}")
    
    # Generate HTML report
    html_content = generate_report(
        rag_output, 
        result, 
        sorted_translations, 
        evaluation_results,
        ground_truth
    )
    
    # Save report (always same filename - just refresh browser!)
    report_file = f"demo_{DOMAIN}.html"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("="*80)
    print("📄 EVALUATION REPORT GENERATED")
    print("="*80)
    print(f"✅ Saved to: {report_file}")
    print(f"🌐 Open this file in browser (or just refresh if already open!)")
    print("\n📊 OFFICIAL METRICS INCLUDED:")
    print("   ✅ chrF++ (Character-level F-score)")
    print("   ✅ BLEU (Word-level precision)")
    print("   ✅ COMET (Neural semantic similarity)")
    print("   ✅ Combined Score (Weighted average)")
    print("   ✅ Quality Level Classification")
    print("="*80 + "\n")


def generate_report(rag_output, result, sorted_translations, evaluation_results, ground_truth):
    """Generate HTML report with official evaluation metrics."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = result.translations[result.recommended_variant]
    
    # Find best by official metrics if available
    if evaluation_results:
        best_eval = max(
            [(k, v) for k, v in evaluation_results.items() if v],
            key=lambda x: x[1].get('combined_score', 0),
            default=(result.recommended_variant, None)
        )
        best_agent_type = best_eval[0]
        best_eval_result = best_eval[1]
    else:
        best_agent_type = result.recommended_variant
        best_eval_result = None
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Translation Evaluation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            font-size: 1.8em;
            margin-bottom: 5px;
            color: #2c3e50;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 25px;
            font-size: 0.9em;
        }}
        .source-text {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            border-left: 3px solid #3498db;
        }}
        .source-text strong {{
            display: block;
            margin-bottom: 8px;
            color: #2c3e50;
        }}
        .variant {{
            background: #fafafa;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
            border: 1px solid #e0e0e0;
        }}
        .variant.recommended {{
            background: #f1f8f4;
            border: 2px solid #27ae60;
        }}
        .variant-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .variant-name {{
            font-weight: 600;
            font-size: 1.1em;
            color: #2c3e50;
        }}
        .variant.recommended .variant-name {{
            color: #27ae60;
        }}
        .score-badge {{
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .variant.recommended .score-badge {{
            background: #27ae60;
        }}
        .translation-text {{
            font-size: 1.3em;
            line-height: 1.6;
            direction: rtl;
            text-align: right;
            background: white;
            padding: 15px;
            border-radius: 4px;
            margin: 12px 0;
            color: #2c3e50;
        }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 12px 0;
        }}
        .meta-item {{
            padding: 8px 10px;
            background: white;
            border-radius: 4px;
            font-size: 0.85em;
            border: 1px solid #e0e0e0;
        }}
        .meta-item strong {{
            display: block;
            margin-bottom: 3px;
            color: #7f8c8d;
            font-size: 0.85em;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #7f8c8d;
            font-size: 0.85em;
        }}
        .recommended-badge {{
            background: #27ae60;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Translation Evaluation Report</h1>
        <div class="subtitle">
            Domain: {rag_output.domain.upper()} | Generated: {timestamp}
            {f" | Iterations: {result.metadata.get('iterations', 1)}" if result.metadata else ""}
            {f" | Refinement: {'✅ Enabled' if result.metadata.get('refinement_enabled') else '❌ Disabled'}" if result.metadata else ""}
        </div>
        
        <div class="source-text">
            <strong>Source Text</strong>
            {rag_output.source_text}
        </div>
        
        {f'''<div class="source-text" style="border-left-color: #27ae60; background: #f1f8f4;">
            <strong>Ground Truth (Reference Translation)</strong>
            {ground_truth.get('arabic', 'Not available')}
        </div>''' if ground_truth.get('arabic') else ''}
        
        <h2 style="margin-top: 30px; margin-bottom: 15px; font-size: 1.3em; color: #2c3e50;">Translation Variants</h2>
"""

    # Add all 3 variants with evaluation metrics
    for rank, (agent_type, variant) in enumerate(sorted_translations, 1):
        eval_result = evaluation_results.get(agent_type)
        is_recommended = agent_type == best_agent_type
        variant_class = "variant recommended" if is_recommended else "variant"
        
        # Determine display score
        if eval_result:
            display_score = f"{eval_result['combined_percentage']:.1f}%"
            score_label = "Combined Score"
        else:
            display_score = f"{variant.quality_score}/10"
            score_label = "Quality"
        
        html += f"""
        <div class="{variant_class}">
            <div class="variant-header">
                <div>
                    <span class="variant-name">#{rank} {agent_type.replace('_', ' ').title()}</span>
                    {f'<span class="recommended-badge">Best</span>' if is_recommended else ''}
                </div>
                <span class="score-badge">{display_score}</span>
            </div>
            
            <div class="translation-text">
                {variant.text}
            </div>
            
            {f'''
            <div class="metadata">
                <div class="meta-item">
                    <strong>chrF++</strong>
                    {eval_result['chrf_score']:.1f}
                </div>
                <div class="meta-item">
                    <strong>BLEU</strong>
                    {eval_result['bleu_score']:.1f}
                </div>
                <div class="meta-item">
                    <strong>COMET</strong>
                    {f"{eval_result['comet_score']:.3f}" if eval_result.get('comet_score') is not None else 'N/A'}
                </div>
                <div class="meta-item">
                    <strong>Combined</strong>
                    {eval_result['combined_percentage']:.1f}%
                </div>
                <div class="meta-item">
                    <strong>Quality</strong>
                    {eval_result['quality_level']}
                </div>
            </div>
            ''' if eval_result else f'''
            <div class="metadata">
                <div class="meta-item">
                    <strong>Quality Score</strong>
                    {variant.quality_score}/10
                </div>
                <div class="meta-item">
                    <strong>Glossary</strong>
                    {variant.glossary_compliance:.0f}%
                </div>
            </div>
            '''}
        </div>
"""

    html += f"""
        <div class="footer">
            Model: Groq LLaMA 3.3-70b | Metrics: chrF++ (60%), BLEU (40%), COMET (50%)
        </div>
    </div>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║     TRANSLATION ORCHESTRATOR - OFFICIAL EVALUATION DEMO              ║
    ║                                                                      ║
    ║  This demonstrates the completed task WITH official metrics:         ║
    ║  ✅ Generate 3 different translation variants                        ║
    ║  ✅ Evaluate with chrF++, BLEU, COMET (official metrics)            ║
    ║  ✅ Compare against ground truth translations                        ║
    ║  ✅ Quality level classification (Excellent → Very Poor)             ║
    ║  ✅ Automatic ranking by combined evaluation score                   ║
    ║                                                                      ║
    ║  ⚠️  FIRST TIME SETUP (COMET model):                                 ║
    ║  COMET will download 1-2 GB model on first run (requires internet)  ║
    ║  Download happens once, then cached. Or skip COMET - chrF++ and     ║
    ║  BLEU will still work.                                               ║
    ║                                                                      ║
    ║  TO TEST WITH DIFFERENT DOMAIN OR TASK:                              ║
    ║  1. Edit DOMAIN at top of file: medical, technology, education, or  ║
    ║     economic                                                         ║
    ║  2. Edit TASK_NUMBER (1-20 for medical, varies by domain)           ║
    ║  3. Run: python demo.py                                              ║
    ║                                                                      ║
    ║  DATA SOURCE:                                                        ║
    ║  📂 Evaluation/Augmented Queries Samples/ (Official data)            ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(demonstrate_translation())
