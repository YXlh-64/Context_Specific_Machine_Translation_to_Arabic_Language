
import os
import json
import pandas as pd
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "ModelSpace/GemmaX2-28-9B-v0.1"
INPUT_CSV = "data.csv"
OUTPUT_FILE = "gold_vs_silver.jsonl"
NUM_SAMPLES = 2000
NUM_GPUS = 2
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

def worker_process(gpu_id, sources, gold_refs, output_file):
    print(f"[GPU {gpu_id}] Launching worker on cuda:{gpu_id}...")
    
    # 1. Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    
    # --- FIX: Changed to 'sdpa' to support RTX 5090 ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": f"cuda:{gpu_id}"}, 
        attn_implementation="sdpa"  
    )
    
    results = []
    print(f"[GPU {gpu_id}] Generating translations for {len(sources)} samples...")
    
    # 2. Batch Generation Loop
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc=f"GPU {gpu_id}", position=gpu_id):
        batch_src = sources[i : i + BATCH_SIZE]
        batch_gold = gold_refs[i : i + BATCH_SIZE]
        
        prompts = []
        for text in batch_src:
            chat = [{"role": "user", "content": f"Translate the following English text to Arabic:\n{text}"}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu_id}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for src, gold, silver, full_prompt in zip(batch_src, batch_gold, decoded_preds, prompts):
            entry = {
                "prompt": full_prompt,
                "chosen": [
                    {"role": "user", "content": f"Translate the following English text to Arabic:\n{src}"},
                    {"role": "assistant", "content": gold.strip()}
                ],
                "rejected": [
                    {"role": "user", "content": f"Translate the following English text to Arabic:\n{src}"},
                    {"role": "assistant", "content": silver.strip()}
                ],
                "source": src
            }
            results.append(entry)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[GPU {gpu_id}] Finished processing.")

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return

    df = pd.read_csv(INPUT_CSV)
    n_samples = min(NUM_SAMPLES, len(df))
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    print(f"Total samples to generate: {len(df)}")
    
    sources = df['en'].tolist()
    refs = df['ar'].tolist()
    midpoint = len(df) // 2
    
    shards = [
        (sources[:midpoint], refs[:midpoint]),
        (sources[midpoint:], refs[midpoint:])
    ]
    
    processes = []
    temp_files = []
    
    for i in range(NUM_GPUS):
        temp_file = f"temp_shard_{i}.jsonl"
        temp_files.append(temp_file)
        p = mp.Process(target=worker_process, args=(i, shards[i][0], shards[i][1], temp_file))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("Merging files...")
    all_data = []
    for tf in temp_files:
        if os.path.exists(tf):
            with open(tf, 'r', encoding='utf-8') as f:
                for line in f:
                    all_data.append(json.loads(line))
            os.remove(tf)
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ SUCCESS: Saved {len(all_data)} preference pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
