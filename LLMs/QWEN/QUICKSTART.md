# QWEN Translation - Quick Start Guide

## Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 50GB+ free disk space

### Step 1: Install Dependencies (2 minutes)

```bash
cd LLMs/QWEN
pip install -r requirements.txt
```

### Step 2: Prepare Your Data (5-10 minutes)

```bash
python prepare_data.py
```

This will process your English-Arabic parallel corpora from `../../Data/english-arabic/` and create training datasets.

### Step 3: Choose Your Training Path

#### Option A: Quick Test (1 hour per domain)
Train on limited samples to test the pipeline:

```bash
# Edit prepare_data.py line 163:
# max_samples=1000  # Instead of None

python train_qwen.py --domain technology --num_train_epochs 1
```

#### Option B: Full Training (8-24 hours per domain)
For production-quality models:

```bash
python train_all.py
```

Or train individual domains:

```bash
python train_qwen.py --domain technology --num_train_epochs 3
python train_qwen.py --domain economic --num_train_epochs 3
python train_qwen.py --domain education --num_train_epochs 3
```

### Step 4: Test Your Model

#### Quick Translation Test
```bash
python evaluate.py \
    --model_path models/technology_model \
    --domain technology \
    --translate_text "The application provides a user-friendly interface."
```

#### Full Evaluation
```bash
python evaluate.py \
    --model_path models/technology_model \
    --domain technology
```

### Step 5: Launch the Web UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📊 What to Expect

### Data After Preparation
```
data/prepared/
├── technology_train.jsonl  (~70-80% of data)
├── technology_val.jsonl    (~10% of data)
├── technology_test.jsonl   (~10% of data)
├── economic_train.jsonl
├── economic_val.jsonl
├── economic_test.jsonl
├── education_train.jsonl
├── education_val.jsonl
└── education_test.jsonl
```

### Models After Training
```
models/
├── technology_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── economic_model/
└── education_model/
```

### Training Time Estimates

**With NVIDIA RTX 3090 (24GB VRAM):**
- Technology domain: ~8 hours
- Economic domain: ~12 hours (larger dataset)
- Education domain: ~10 hours

**With NVIDIA T4 (16GB VRAM):**
- Technology domain: ~12 hours
- Economic domain: ~18 hours
- Education domain: ~15 hours

### Expected Performance

After 3 epochs of training:
- BLEU scores: 30-45
- chrF scores: 55-70
- Translation quality: Good for domain-specific terminology

---

## 🔧 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train_qwen.py --domain technology --per_device_train_batch_size 2
```

### "Too slow on CPU"
Training on CPU is not recommended. Consider:
- Using Google Colab with GPU
- Renting a GPU instance (AWS, Lambda Labs, RunPod)
- Using smaller model: Already using Qwen2-1.5B (smallest reasonable size)

### "Import errors"
```bash
pip install -r requirements.txt --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 Files Explained

| File | Purpose |
|------|---------|
| `prepare_data.py` | Loads and formats your CSV data for training |
| `train_qwen.py` | Trains a single domain model |
| `train_all.py` | Trains all three domains sequentially |
| `evaluate.py` | Tests model performance and generates translations |
| `app.py` | Streamlit web UI for interactive testing |
| `requirements.txt` | Python dependencies |
| `config.ini` | Configuration settings |
| `quick_start.ipynb` | Jupyter notebook for experimentation |

---

## 🎯 Next Steps After Training

1. **Evaluate all models**
   ```bash
   for domain in technology economic education; do
       python evaluate.py --model_path models/${domain}_model --domain $domain
   done
   ```

2. **Launch the UI**
   ```bash
   streamlit run app.py
   ```

3. **Export for production** (optional)
   - Merge LoRA weights with base model
   - Quantize for faster inference
   - Deploy as API endpoint

---

## 💡 Tips for Best Results

 **DO:**
- Use clean, high-quality parallel corpora
- Train for at least 3 epochs
- Monitor training loss in TensorBoard
- Test on diverse examples from each domain

 **DON'T:**
- Mix domains during training (train separately)
- Use data with incorrect alignments
- Stop training too early (wait for convergence)
- Ignore evaluation metrics

---

## Need Help?

1. Check the [full README](README.md) for detailed documentation
2. Review training logs in `models/{domain}_model/`
3. Use TensorBoard: `tensorboard --logdir models/`
4. Test with the Jupyter notebook: `jupyter notebook quick_start.ipynb`

---

**Ready to start? Run:**
```bash
chmod +x setup.fish  # Make setup script executable
./setup.fish         # Run automated setup
```

Or manually:
```bash
pip install -r requirements.txt
python prepare_data.py
python train_qwen.py --domain technology
```

Good luck! 
