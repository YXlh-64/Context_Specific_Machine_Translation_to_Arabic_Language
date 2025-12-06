# RLHF Arabic Translation System

This project implements a complete Reinforcement Learning from Human Feedback (RLHF) pipeline for English/French to Arabic translation using Gemma-2X289B and Gemma-2 2B models.

## 📋 Project Overview

The system follows a three-phase approach:

- **Phase 1**: Supervised Fine-Tuning (SFT) - ✅ Already completed with Gemma-2X289B
- **Phase 2**: Cold-Start Training with Synthetic Data
  - Generate translation candidates with varying parameters
  - Score using automatic metrics (COMET, BERTScore, CHRF)
  - Train reward model on synthetic preferences
  - Optimize with PPO
- **Phase 3**: Human Alignment
  - Collect user feedback on translations
  - Fine-tune reward model with human preferences
  - Re-run PPO for final alignment

## 📁 Project Structure

```
Reinforcement Learning/
├── 0_config_setup.ipynb                    # Configuration and setup
├── 1_synthetic_data_generation.ipynb       # Generate synthetic preference data
├── 2_reward_model_training.ipynb           # Train reward model (cold-start)
├── 3_ppo_optimization.ipynb                # PPO training (cold-start)
├── 4_inference_user_interaction.ipynb      # User interface & feedback collection
├── 5_human_preference_finetuning.ipynb     # Human alignment & final PPO
├── data/                                   # Data directory
│   ├── test_prompts.jsonl                  # Source texts for generation
│   ├── synthetic_preferences.jsonl         # Generated synthetic data
│   └── human_preferences.jsonl             # Collected human feedback
├── models/                                 # Model checkpoints
│   ├── reward_model_coldstart/
│   ├── reward_model_human/
│   ├── ppo_model_coldstart/
│   └── ppo_model_final/
├── outputs/                                # Generated outputs
└── logs/                                   # Training logs
```

## 🚀 Getting Started

### Prerequisites

The school computer should have:
- CUDA-capable GPU
- Gemma-2X289B model downloaded locally
- Python 3.8+

### Step 1: Initial Setup

1. Open `0_config_setup.ipynb`
2. **IMPORTANT**: Update the `SFT_MODEL_PATH` variable to point to your Gemma-2X289B model location:
   ```python
   SFT_MODEL_PATH = "/path/to/your/Gemma-2X289B"  # UPDATE THIS!
   ```
3. Run all cells to:
   - Install required packages
   - Check GPU availability
   - Create directory structure
   - Save configuration

### Step 2: Prepare Data (OPTIONAL)

The provided `data/test_prompts.jsonl` file contains sample source texts.

If you want to use your own prompts, create a JSONL file with this format:
```json
{"text": "Hello, how are you?", "lang": "en"}
{"text": "Bonjour, comment allez-vous?", "lang": "fr"}
```

## 📓 Notebook Execution Order

### Phase 2: Cold-Start Training

#### Notebook 1: Synthetic Data Generation
**File**: `1_synthetic_data_generation.ipynb`

Generates synthetic preference pairs using automatic metrics.

**What it does**:
- Loads the SFT model (Gemma-2X289B)
- Loads test prompts (source texts only - no references needed!)
- Generates 3-8 translation variants per source text
- Scores each with reference-free metrics (COMET-QE, length, Arabic validation)
- Creates pairwise preference data
- Saves to `data/synthetic_preferences.jsonl`

**Expected time**: 2-6 hours (depending on corpus size and GPU)

**Output**: ~10,000-50,000 preference pairs

---

#### Notebook 2: Reward Model Training
**File**: `2_reward_model_training.ipynb`

Trains a reward model to evaluate translation quality.

**What it does**:
- Loads Gemma-2 2B as base
- Adds reward head (MLP or linear)
- Trains with Bradley-Terry loss
- Validates on held-out data
- Saves to `models/reward_model_coldstart/`

**Expected time**: 1-3 hours

**Success metric**: Validation accuracy > 0.70

---

#### Notebook 3: PPO Optimization
**File**: `3_ppo_optimization.ipynb`

Optimizes the translation model using PPO with the reward model.

**What it does**:
- Loads SFT model as policy
- Loads trained reward model
- Runs PPO training loop
- Applies KL penalty to preserve faithfulness
- Saves to `models/ppo_model_coldstart/`

**Expected time**: 4-8 hours

**Monitor**: Mean reward should increase, KL should stay bounded

---

### Phase 3: Human Alignment

#### Notebook 4: Inference & User Interaction
**File**: `4_inference_user_interaction.ipynb`

User-facing translation interface with feedback collection.

**What it does**:
- Generates 5-8 candidate translations
- Scores and ranks them with reward model
- Shows top 3 to user
- Collects rankings or custom translations
- Saves feedback to `data/human_preferences.jsonl`

**Usage**:
```python
# In the interface widget:
1. Enter source text
2. Select language (English/French)
3. Click "Translate"
4. Review translations

# Provide feedback:
submit_ranking([2, 1, 3])  # Translation 2 is best
# OR
submit_custom_translation("مرحبا، كيف حالك؟")
```

**Goal**: Collect 500-1000+ feedback entries

---

#### Notebook 5: Human Preference Fine-Tuning
**File**: `5_human_preference_finetuning.ipynb`

Final alignment with human preferences.

**What it does**:
- Loads collected human feedback
- Converts to preference pairs
- Fine-tunes reward model
- Re-runs PPO with human-aligned rewards
- Saves final model to `models/ppo_model_final/`

**Expected time**: 2-5 hours

**Output**: Production-ready translation model!

---

## 🔧 Configuration Parameters

Key parameters in `0_config_setup.ipynb`:

```python
# Synthetic data generation
NUM_CANDIDATES = 8              # Translation variants per input
TEMPERATURES = [0.6, 0.8, 1.0, 1.2]
METRIC_WEIGHTS = {
    'comet': 0.5,
    'bertscore': 0.3,
    'chrf': 0.2
}

# Reward model training
RM_LEARNING_RATE = 1e-5
RM_BATCH_SIZE = 8
RM_EPOCHS = 3

# PPO training
PPO_LEARNING_RATE = 1.41e-5
PPO_STEPS = 1000
KL_PENALTY_COEF = 0.1           # Faithfulness preservation

# Inference
INFERENCE_NUM_CANDIDATES = 8    # Generate 5-8 candidates
INFERENCE_TOP_K = 3             # Show top 3 to user
```

## 📊 Monitoring Training

### Using Weights & Biases (Optional)

Enable tracking in `0_config_setup.ipynb`:
```python
USE_WANDB = True
WANDB_PROJECT = "rlhf-arabic-translation"
```

Then log in:
```python
import wandb
wandb.login()
```

### Key Metrics to Monitor

**Reward Model Training**:
- Validation accuracy > 0.70 (cold-start) or > 0.80 (human-aligned)
- Loss should decrease steadily

**PPO Training**:
- Mean reward should increase
- KL divergence should stay < 0.2 (maintains faithfulness)
- Policy loss should stabilize

## 🔄 Continuous Improvement Loop

After deployment:

1. **Collect Feedback** (Notebook 4)
   - Users interact with translation system
   - Feedback automatically saved

2. **Periodic Retraining** (Notebook 5)
   - Weekly/monthly schedule
   - When you have 200+ new feedback entries

3. **A/B Testing**
   - Compare old vs new model versions
   - Monitor quality metrics

## 💡 Tips & Best Practices

### Memory Management
- If GPU memory is limited, reduce batch sizes
- Use gradient accumulation to maintain effective batch size
- Consider LoRA/QLoRA for PPO if needed
### Data Quality
- Use diverse source prompts (various domains, lengths, complexities)
- Since Phase 1 is complete, you don't need parallel data
- The provided test prompts are sufficient to startality pairs
- Balance EN-AR and FR-AR data

### Hyperparameter Tuning
- Start with default values
- Adjust KL penalty if translations drift too much
- Increase PPO steps if reward hasn't plateaued

### Human Feedback
- Quality > Quantity for feedback
- Provide clear instructions to annotators
- Validate feedback consistency

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: 
```python
# In config notebook, reduce batch sizes:
RM_BATCH_SIZE = 4
PPO_BATCH_SIZE = 4
PPO_MINI_BATCH_SIZE = 1
```

### Issue: Low Reward Model Accuracy
**Solution**:
- Check synthetic data quality
- Increase training epochs
- Try larger base model for reward model

### Issue: PPO Not Improving
**Solution**:
- Verify reward model is working correctly
- Reduce KL penalty coefficient
- Increase number of PPO steps
- Check generation parameters

### Issue: Models Generate Gibberish
**Solution**:
- Increase KL penalty (preserve SFT behavior)
- Check tokenizer configuration
- Verify data preprocessing

## 📝 Citation

If using this code for research, please cite:

```bibtex
@misc{rlhf-arabic-translation,
  title={RLHF Arabic Translation System},
  year={2025},
  note={Based on Gemma-2 models and PPO optimization}
}
```

## 📄 License

This project uses models from Google (Gemma-2), which are subject to their respective licenses.

## 🤝 Contributing

To improve the system:
1. Collect more diverse training data
2. Experiment with different reward model architectures
3. Try alternative RL algorithms (DPO, RRHF)
4. Add domain-specific glossaries (RAG integration)

## 📞 Support

For questions or issues:
1. Check notebook markdown cells for detailed explanations
2. Review error messages and stack traces
3. Verify all paths and configurations are correct
4. Check GPU memory usage and availability

---

**Good luck with your RLHF Arabic translation project!** 🚀

Remember to start by updating the model path in `0_config_setup.ipynb`!
