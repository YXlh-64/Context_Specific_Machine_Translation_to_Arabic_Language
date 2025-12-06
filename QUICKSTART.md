# Quick Start Guide

## First Time Setup (5 minutes)

### 1. Update Model Path
Open `0_config_setup.ipynb` and find this line:
```python
SFT_MODEL_PATH = "/path/to/Gemma-2X289B"  # UPDATE THIS PATH
```

Replace with your actual model location, for example:
```python
SFT_MODEL_PATH = "/mnt/models/Gemma-2X289B"
```

### 2. Prepare Your Data
The `data/test_prompts.jsonl` file is already provided with sample prompts.

You can optionally add more source texts in JSONL format:
```json
{"text": "Hello, how are you?", "lang": "en"}
{"text": "Bonjour, comment allez-vous?", "lang": "fr"}
```

### 3. Run Setup Notebook
```
Open: 0_config_setup.ipynb
Click: Run All Cells
Wait: ~5 minutes for package installation
```

## Execution Timeline

### Phase 2: Cold-Start (1-2 days)

**Day 1 Morning**: 
- Run `1_synthetic_data_generation.ipynb` (4-6 hours)
  - Generates synthetic preference data
  - You can start this and check back later

**Day 1 Afternoon**:
- Run `2_reward_model_training.ipynb` (2-3 hours)
  - Trains the reward model
  - Monitor validation accuracy

**Day 1 Evening**:
- Start `3_ppo_optimization.ipynb` (6-8 hours)
  - Can run overnight
  - Check progress in the morning

**Day 2**: 
- Model is ready for testing!
- Run `4_inference_user_interaction.ipynb`

### Phase 3: Human Alignment (Ongoing)

**Week 1-2**: Collect feedback
- Use notebook 4 daily
- Goal: 500+ feedback entries

**Week 3**: Final training
- Run `5_human_preference_finetuning.ipynb`
- Get production model!

## Quick Command Reference

### Check GPU
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Monitor Training
```python
# In any training notebook:
# Look for these indicators:

# Reward Model:
# ✓ Validation accuracy > 0.70 is good
# ✓ Loss decreasing = training working

# PPO:
# ✓ Mean reward increasing = model improving
# ✓ KL divergence < 0.2 = staying faithful to SFT
```

### Test Translation
```python
# Quick test after any training stage:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/ppo_model_coldstart")
tokenizer = AutoTokenizer.from_pretrained("models/ppo_model_coldstart")

prompt = "Translate the following English text to Arabic:\n\nHello, how are you?\n\nArabic translation:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Common Questions

### Q: How long will this take?
**A**: 
- Phase 2 (Cold-start): 1-2 days
- Feedback collection: 1-2 weeks  
- Phase 3 (Final): 1 day
- **Total**: 2-3 weeks

### Q: How much GPU memory needed?
**A**: 
- Minimum: 24GB (RTX 3090, A5000)
- Recommended: 40GB+ (A100, A6000)
- If limited: Reduce batch sizes in config

### Q: Can I pause and resume?
**A**: 
- ✅ Yes! All notebooks save checkpoints
- Models saved after each epoch
- Can stop/restart training anytime

### Q: How much data needed?
**A**:
- ⚠️ NO parallel corpora needed! (Phase 1 already done)
- Test prompts: 40+ provided (can add more source texts)
- Human feedback: 500+ entries (collected during Phase 3)

### Q: What if training fails?
**A**:
1. Check GPU memory (reduce batch size)
2. Verify model path is correct
3. Check data format (tab-separated)
4. Look at error messages in notebook

## Workflow Checklist

### Before Starting
- [ ] GPU available and working
- [ ] Gemma-2X289B model downloaded
- [ ] Updated `SFT_MODEL_PATH` in config
- [ ] ✅ Test prompts already provided (no parallel corpus needed!)
- [ ] Run `0_config_setup.ipynb` successfully

### Phase 2 Checklist
- [ ] Generated synthetic data (notebook 1)
- [ ] Trained reward model (notebook 2)
  - [ ] Validation accuracy > 0.70
- [ ] Ran PPO optimization (notebook 3)
  - [ ] Mean reward increased
  - [ ] KL stayed bounded
- [ ] Tested translations (notebook 4)

### Phase 3 Checklist
- [ ] Collected 500+ feedback entries
- [ ] Fine-tuned reward model (notebook 5)
  - [ ] Validation accuracy > 0.80
- [ ] Re-ran PPO with human rewards
- [ ] Final model saved
- [ ] Tested production model

## Emergency Fixes

### GPU Memory Error
```python
# In 0_config_setup.ipynb, change:
RM_BATCH_SIZE = 2  # was 8
PPO_BATCH_SIZE = 2  # was 8
RM_GRADIENT_ACCUMULATION_STEPS = 16  # was 4
```

### Model Not Improving
```python
# In 0_config_setup.ipynb, try:
KL_PENALTY_COEF = 0.05  # was 0.1 (more exploration)
PPO_STEPS = 2000  # was 1000 (more training)
PPO_LEARNING_RATE = 2e-5  # was 1.41e-5 (faster learning)
```

### Poor Translations
```python
# In generation, adjust:
temperature = 0.7  # was 0.9 (more focused)
top_p = 0.9  # was 0.95 (less randomness)
```

## File Organization

```
Keep organized:
✓ Raw data in data/
✓ Models in models/
✓ Logs in logs/
✓ Backups of important checkpoints
✓ Notes on hyperparameters that worked
```

## Next Steps After Completion

1. **Deploy model** in production environment
2. **Setup feedback pipeline** for continuous collection
3. **Schedule retraining** (monthly recommended)
4. **Monitor quality** with A/B testing
5. **Expand** to other language pairs

## Contact & Resources

- **Gemma Models**: https://ai.google.dev/gemma
- **TRL Library**: https://github.com/huggingface/trl
- **COMET Metric**: https://github.com/Unbabel/COMET
- **Transformers**: https://huggingface.co/docs/transformers

---

**Ready to start?** Open `0_config_setup.ipynb` and update the model path! 🚀
