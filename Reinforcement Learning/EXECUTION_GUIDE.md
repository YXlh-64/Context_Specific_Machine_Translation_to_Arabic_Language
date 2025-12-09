# RLHF Arabic Translation Pipeline - Complete Execution Guide

This guide walks you through the complete RLHF (Reinforcement Learning from Human Feedback) pipeline for training an Arabic translation model.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Pipeline Execution](#4-pipeline-execution)
5. [Troubleshooting](#5-troubleshooting)
6. [Expected Outputs](#6-expected-outputs)

---

## 1. Prerequisites

### Hardware Requirements
- **Recommended**: 2x NVIDIA RTX 5090 (or equivalent with 64GB+ total VRAM)
- **Minimum**: 1x GPU with 24GB+ VRAM (reduce batch sizes accordingly)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free disk space

### Software Requirements
- Python 3.10+
- CUDA 12.0+ with cuDNN
- Git

### Accounts Needed
- **Hugging Face Account**: For downloading models (some require access approval)
  - Request access to: `google/gemma-2-9b-it`, `google/gemma-2-2b`

---

## 2. Environment Setup

### Step 2.1: Create Virtual Environment

```bash
# Navigate to project directory
cd "/home/aya/Desktop/ENSIA 4Y/S1/NLP/Project/Context_Specific_Machine_Translation_to_Arabic_Language/Reinforcement Learning"

# Create virtual environment
python -m venv venv

# Activate (fish shell)
source venv/bin/activate.fish

# Or for bash/zsh:
# source venv/bin/activate
```

### Step 2.2: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers and training libraries
pip install transformers accelerate datasets trl peft bitsandbytes

# COMET for translation quality scoring (IMPORTANT!)
pip install unbabel-comet

# Additional evaluation metrics
pip install bert-score sacrebleu

# Utilities
pip install wandb pandas numpy tqdm

# Optional: Flash Attention 2 (for faster inference)
pip install flash-attn --no-build-isolation
```

### Step 2.3: Authenticate with Hugging Face

```bash
# Login to Hugging Face (required for Gemma models)
huggingface-cli login

# Enter your access token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

### Step 2.4: Verify GPU Setup

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## 3. Data Preparation

### Step 3.1: Verify Data Location

Your translation data should be in CSV format at:
```
/home/aya/Desktop/ENSIA 4Y/S1/NLP/Project/Context_Specific_Machine_Translation_to_Arabic_Language/Data/english-arabic/
```

Expected CSV columns:
- `source` or `english`: English text
- `target` or `arabic`: Arabic translation (optional, for reference)
- `domain`: Domain label (optional, for stratified sampling)

### Step 3.2: Check Data Files

```bash
# List available CSV files
ls -la "../Data/english-arabic/"

# Check row count
wc -l ../Data/english-arabic/*.csv

# Preview data format
head -5 ../Data/english-arabic/technology.csv
```

---

## 4. Pipeline Execution

### Overview

The pipeline consists of 6 notebooks that must be run **in order**:

| Notebook | Purpose | Estimated Time |
|----------|---------|----------------|
| 0_config_setup | Configuration & setup | 2 minutes |
| 1_synthetic_data_generation | Generate translation candidates | 4-8 hours |
| 2_reward_model_training | Train reward model | 1-2 hours |
| 3_ppo_optimization | PPO training with LoRA | 2-4 hours |
| 4_inference_user_interaction | Test & collect feedback | Manual |
| 5_human_preference_finetuning | Fine-tune with human data | 1-2 hours |

---

### Step 4.1: Run Configuration Setup (Notebook 0)

**File**: `0_config_setup.ipynb`

1. Open the notebook in VS Code or Jupyter
2. Run **all cells** in order
3. Verify outputs show:
   - ✅ GPU(s) detected
   - ✅ Hugging Face token loaded
   - ✅ Project directories created
   - ✅ Translation data found

**Expected Output**:
```
GPU Available: True
Number of GPUs: 2
  GPU 0: NVIDIA GeForce RTX 5090
  GPU 1: NVIDIA GeForce RTX 5090
Total VRAM: 64.00 GB
✅ Configuration loaded successfully!
```

---

### Step 4.2: Generate Synthetic Preference Data (Notebook 1)

**File**: `1_synthetic_data_generation.ipynb`

**What it does**:
1. Loads 100,000 samples from your translation dataset
2. Generates 4 translation candidates per source sentence
3. Scores translations using COMET (quality estimation)
4. Creates preference pairs (chosen vs rejected)

**Run all cells in order**. Key cells:

| Cell | Description |
|------|-------------|
| 1 | Load config from notebook 0 |
| 2 | Import libraries + load COMET model |
| 3 | Load SFT translation model |
| 4 | Load and sample training data (100K) |
| 5 | Test generation on single sample |
| 6 | (Optional) vLLM setup for faster generation |
| 7 | Scoring function setup |
| 8 | **Main generation loop** (takes 4-8 hours) |
| 9 | Save synthetic preferences |

**Monitor Progress**:
```
📊 Progress Report:
   Samples: 50,000/100,000 (50.0%)
   Pairs generated: 150,000
   COMET scores: avg=0.7823, range=[0.45, 0.92]
   Rate: 8.2 samples/sec
   ETA: 1.70 hours
```

**Output Files**:
- `data/generated_candidates.jsonl` - Raw translation candidates
- `data/synthetic_preferences.jsonl` - Preference pairs (~400K pairs)

---

### Step 4.3: Train Reward Model (Notebook 2)

**File**: `2_reward_model_training.ipynb`

**What it does**:
1. Loads synthetic preference pairs
2. Trains a reward model (Gemma-2 2B + MLP head)
3. Uses Bradley-Terry loss for preference learning

**Run all cells in order**.

**Key Configuration** (already set in notebook 0):
```python
RM_LEARNING_RATE = 1e-5
RM_BATCH_SIZE = 32
RM_EPOCHS = 3
```

**Expected Training Output**:
```
Epoch 1/3: Loss=0.693 → 0.412
Epoch 2/3: Loss=0.412 → 0.298
Epoch 3/3: Loss=0.298 → 0.251
✅ Reward model saved to models/reward_model_coldstart
```

**Output**:
- `models/reward_model_coldstart/` - Trained reward model

---

### Step 4.4: PPO Optimization (Notebook 3)

**File**: `3_ppo_optimization.ipynb`

**What it does**:
1. Loads the SFT translation model
2. Applies LoRA adapters (only ~0.3% params trainable)
3. Runs PPO optimization using the reward model
4. Monitors KL divergence to prevent catastrophic forgetting

**Run all cells in order**.

**Safety Monitoring**:
The notebook monitors KL divergence:
- ✅ **Healthy** (KL < 0.1): Training normally
- ⚠️ **Moderate** (0.1 ≤ KL < 0.2): Watch closely
- 🔴 **High** (KL ≥ 0.2): Auto-stops to protect model

**Expected Output**:
```
Step 100/1000: reward=0.72, kl=0.05 ✅Healthy
Step 200/1000: reward=0.78, kl=0.07 ✅Healthy
Step 300/1000: reward=0.81, kl=0.09 ✅Healthy
...
✅ PPO training complete!
   Final reward: 0.85
   Final KL: 0.11
```

**Output**:
- `models/ppo_model_coldstart/` - PPO-optimized LoRA adapters

---

### Step 4.5: Test and Collect Human Feedback (Notebook 4)

**File**: `4_inference_user_interaction.ipynb`

**What it does**:
1. Loads the PPO-optimized model
2. Generates translations for test prompts
3. Displays candidates for human ranking
4. Saves human preferences

**Run cells and interact**:
1. Enter source text to translate
2. Review generated candidates (ranked by reward)
3. Provide your ranking (best to worst)
4. Repeat for 50-100 samples

**Output**:
- `data/human_preferences.jsonl` - Human-ranked preference data

---

### Step 4.6: Human Preference Fine-tuning (Notebook 5)

**File**: `5_human_preference_finetuning.ipynb`

**What it does**:
1. Re-trains reward model on human preferences
2. Runs additional PPO steps with human-aligned reward
3. Produces final optimized model

**Run all cells in order**.

**Output**:
- `models/reward_model_human/` - Human-aligned reward model
- `models/ppo_model_final/` - Final production model

---

## 5. Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch sizes in `0_config_setup.ipynb`:
```python
MEGA_BATCH_SIZE = 32  # Reduce from 64
RM_BATCH_SIZE = 16    # Reduce from 32
PPO_BATCH_SIZE = 16   # Reduce from 32
COMET_BATCH_SIZE = 32 # Reduce from 64
```

### Issue: COMET Download Fails

**Solution**: Manually download:
```bash
python -c "from comet import download_model; download_model('Unbabel/wmt22-cometkiwi-da')"
```

### Issue: Hugging Face Model Access Denied

**Solution**: 
1. Go to model page (e.g., https://huggingface.co/google/gemma-2-9b-it)
2. Click "Access repository" and accept terms
3. Wait for approval (usually instant for Gemma models)

### Issue: Flash Attention Not Working

**Solution**: Flash Attention is optional. Disable in config:
```python
USE_FLASH_ATTENTION = False
```

### Issue: Generation Too Slow

**Solutions**:
1. Reduce `SAMPLE_SIZE` from 100000 to 50000
2. Reduce `NUM_CANDIDATES` from 4 to 3
3. Enable vLLM (set `USE_VLLM = True` in notebook 1)

### Issue: KL Divergence Too High (PPO stops early)

**Solutions**:
1. Increase `KL_PENALTY_COEF` from 0.15 to 0.2
2. Decrease `PPO_LEARNING_RATE` to 1e-5
3. This is actually a safety feature - it protects your model!

---

## 6. Expected Outputs

After completing the full pipeline, you should have:

### Data Files (`data/`)
| File | Size | Description |
|------|------|-------------|
| `synthetic_preferences.jsonl` | ~200MB | 400K+ preference pairs |
| `generated_candidates.jsonl` | ~500MB | Raw translation candidates |
| `human_preferences.jsonl` | ~1MB | 50-100 human rankings |

### Model Checkpoints (`models/`)
| Directory | Size | Description |
|-----------|------|-------------|
| `reward_model_coldstart/` | ~5GB | Reward model (synthetic) |
| `reward_model_human/` | ~5GB | Reward model (human-aligned) |
| `ppo_model_coldstart/` | ~50MB | LoRA adapters (synthetic) |
| `ppo_model_final/` | ~50MB | Final LoRA adapters |

### Using the Final Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "ModelSpace/GemmaX2-28-9B-v0.1",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "models/ppo_model_final")

# Generate translation
prompt = "Translate the following English text to Arabic:\n\nHello, how are you?\n\nArabic translation:"
output = model.generate(...)
```

---

## 📊 Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Notebook 0  │───▶│  Notebook 1  │───▶│  Notebook 2  │      │
│  │    Config    │    │  Synthetic   │    │   Reward     │      │
│  │    Setup     │    │    Data      │    │   Model      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                   │              │
│                              ▼                   ▼              │
│                      400K preference      Reward Model          │
│                          pairs            (cold start)          │
│                                                  │              │
│  ┌──────────────┐    ┌──────────────┐           │              │
│  │  Notebook 5  │◀───│  Notebook 4  │◀──────────┤              │
│  │   Human      │    │   Collect    │           │              │
│  │  Fine-tune   │    │   Feedback   │    ┌──────▼──────┐       │
│  └──────────────┘    └──────────────┘    │  Notebook 3 │       │
│         │                   ▲            │     PPO     │       │
│         ▼                   │            │ Optimization│       │
│  ┌──────────────┐    Human prefs        └─────────────┘       │
│  │ Final Model  │           │                   │              │
│  │  (LoRA)      │           └───────────────────┘              │
│  └──────────────┘              PPO Model                       │
│                              (cold start)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⏱️ Total Time Estimate

| Phase | Time |
|-------|------|
| Setup & Config | 30 minutes |
| Synthetic Data Generation | 4-8 hours |
| Reward Model Training | 1-2 hours |
| PPO Optimization | 2-4 hours |
| Human Feedback Collection | 1-2 hours (manual) |
| Human Fine-tuning | 1-2 hours |
| **Total** | **10-20 hours** |

---

## 🎉 Success Criteria

Your pipeline is successful when:

1. ✅ Synthetic preferences file has 300K+ pairs
2. ✅ Reward model loss < 0.3 after training
3. ✅ PPO reward increases during training
4. ✅ KL divergence stays below 0.2
5. ✅ Final model generates fluent Arabic translations

Good luck with your RLHF training! 🚀
