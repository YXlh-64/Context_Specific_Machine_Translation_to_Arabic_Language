#!/usr/bin/env python3
"""
Process partial synthetic data into preference pairs for testing
"""
import json
from pathlib import Path
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load scoring function from notebook
from comet import download_model, load_from_checkpoint

print("Loading COMET model...")
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
print("✓ COMET loaded\n")

def score_translation_cpu(source, translation):
    """Score a translation using reference-free metrics (CPU only)"""
    scores = {}
    
    # COMET-QE (Quality Estimation - force CPU)
    try:
        comet_input = [{
            'src': source,
            'mt': translation
        }]
        # Force CPU for COMET to avoid CUDA errors
        comet_output = comet_model.predict(comet_input, batch_size=1, gpus=0)
        scores['comet'] = float(comet_output.scores[0])
    except Exception as e:
        print(f"Warning: COMET scoring failed: {e}")
        scores['comet'] = 0.0
    
    # Simple heuristic scores
    src_len = len(source.split())
    tgt_len = len(translation.split())
    length_ratio = min(tgt_len, src_len * 2) / max(src_len, 1)
    scores['length_score'] = float(min(length_ratio, 1.0))
    
    # Arabic characters
    arabic_chars = sum(1 for c in translation if '\u0600' <= c <= '\u06FF')
    scores['arabic_score'] = float(min(arabic_chars / max(tgt_len, 1), 1.0))
    
    # Combined score
    combined_score = (
        scores['comet'] * 0.7 +
        scores['length_score'] * 0.2 +
        scores['arabic_score'] * 0.1
    )
    scores['combined'] = float(combined_score)
    
    return scores

def create_preference_pairs(candidates_with_scores):
    """Create pairwise preference data from scored candidates"""
    preference_pairs = []
    
    sorted_candidates = sorted(
        candidates_with_scores, 
        key=lambda x: x['scores']['combined'], 
        reverse=True
    )
    
    for i in range(len(sorted_candidates)):
        for j in range(i + 1, len(sorted_candidates)):
            chosen = sorted_candidates[i]
            rejected = sorted_candidates[j]
            
            score_diff = chosen['scores']['combined'] - rejected['scores']['combined']
            if score_diff > 0.05:
                preference_pairs.append({
                    'chosen': chosen['translation'],
                    'rejected': rejected['translation'],
                    'chosen_score': chosen['scores']['combined'],
                    'rejected_score': rejected['scores']['combined'],
                    'score_margin': score_diff
                })
    
    return preference_pairs

# Process partial data
print("Processing partial candidates...")
candidates_file = Path("data/generated_candidates.jsonl")
output_file = Path("data/synthetic_preferences.jsonl")

synthetic_dataset = []
processed = 0

with open(candidates_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            record = json.loads(line)
            source = record['source']
            source_lang = record['source_lang']
            candidates = record['candidates']
            
            print(f"Processing sample {line_num}: {source[:50]}...")
            
            # Score each candidate (CPU only)
            scored_candidates = []
            for cand in candidates:
                scores = score_translation_cpu(source, cand['translation'])
                scored_candidates.append({
                    'translation': cand['translation'],
                    'scores': scores,
                    'config': cand['config']
                })
            
            # Create preference pairs
            preference_pairs = create_preference_pairs(scored_candidates)
            
            for pair in preference_pairs:
                synthetic_dataset.append({
                    'source': source,
                    'source_lang': source_lang,
                    'chosen': pair['chosen'],
                    'rejected': pair['rejected'],
                    'chosen_score': pair['chosen_score'],
                    'rejected_score': pair['rejected_score'],
                    'margin': pair['score_margin']
                })
            
            processed += 1
            
        except Exception as e:
            print(f"Error on line {line_num}: {e}")
            continue

print(f"\n✓ Processed {processed} samples")
print(f"✓ Generated {len(synthetic_dataset)} preference pairs")

# Save to file
print(f"\nSaving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in synthetic_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Save stats
stats = {
    'total_pairs': len(synthetic_dataset),
    'avg_margin': sum(item['margin'] for item in synthetic_dataset) / len(synthetic_dataset),
    'avg_chosen_score': sum(item['chosen_score'] for item in synthetic_dataset) / len(synthetic_dataset),
    'avg_rejected_score': sum(item['rejected_score'] for item in synthetic_dataset) / len(synthetic_dataset),
    'num_sources': processed
}

stats_file = Path("data/synthetic_data_stats.json")
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✓ Saved {len(synthetic_dataset)} preference pairs")
print("\nDataset Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value if isinstance(value, int) else f'{value:.4f}'}")

print("\n✓ Ready to test reward model training!")
print(f"   Dataset: {output_file}")
print(f"   Stats: {stats_file}")
