# QWEN Fine-tuning for Domain-Specific English-Arabic Translation

This project implements domain-specific fine-tuning of QWEN LLMs for English-Arabic translation using LoRA/QLoRA. Each domain (Technology, Economic, Education/Research) is trained independently to ensure optimal performance for context-specific translations.

## 🌟 Features

- **Domain-Specific Training**: Independent fine-tuning for Technology, Economic, and Education domains
- **Memory Efficient**: Uses QLoRA (4-bit quantization) for training on consumer GPUs
- **Interactive UI**: Streamlit-based web interface for testing and comparing models
- **Comprehensive Evaluation**: BLEU and chrF score metrics for translation quality
- **Batch Processing**: Scripts for training all domains sequentially

## 📁 Project Structure

```
QWEN/
├── prepare_data.py          # Data preparation and formatting
├── train_qwen.py           # Single domain training script
├── train_all.py            # Batch training for all domains
├── evaluate.py             # Model evaluation and inference
├── app.py                  # Streamlit UI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   └── prepared/          # Processed datasets (generated)
├── models/                # Fine-tuned models (generated)
│   ├── technology_model/
│   ├── economic_model/
│   └── education_model/
└── results/               # Evaluation results (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Make sure you have CUDA installed for GPU support. The scripts require at least 16GB of GPU memory for training (or 8GB with aggressive optimization).

### 2. Prepare Data

The data preparation script will load your parallel corpora and format them for QWEN training:

```bash
python prepare_data.py
```

This will:
- Load data from `../../Data/english-arabic/` directory
- Create train/validation/test splits (80/10/10)
- Format data in QWEN chat template
- Save prepared datasets to `data/prepared/`

**Expected output:**
- `technology_train.jsonl`, `technology_val.jsonl`, `technology_test.jsonl`
- `economic_train.jsonl`, `economic_val.jsonl`, `economic_test.jsonl`
- `education_train.jsonl`, `education_val.jsonl`, `education_test.jsonl`
- Statistics files for each domain

### 3. Train Models

#### Option A: Train All Domains (Recommended for production)

```bash
python train_all.py
```

This will train models for all three domains sequentially. **Warning**: This can take several hours to days depending on your GPU.

#### Option B: Train a Single Domain (Quick testing)

```bash
python train_qwen.py \
    --domain technology \
    --data_dir data/prepared \
    --output_dir models/technology_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```

**Training Parameters Explained:**
- `--domain`: Choose from `technology`, `economic`, or `education`
- `--num_train_epochs`: Number of training epochs (3-5 recommended)
- `--per_device_train_batch_size`: Batch size per GPU (adjust based on memory)
- `--gradient_accumulation_steps`: Accumulate gradients for larger effective batch size
- `--learning_rate`: Learning rate (2e-4 is a good default for LoRA)

**Model Selection:**
By default, the script uses `Qwen/Qwen2-1.5B-Instruct`. For better quality (but slower training), use:
```bash
--model_name Qwen/Qwen2-7B-Instruct
```

### 4. Evaluate Models

After training, evaluate model performance:

```bash
python evaluate.py \
    --model_path models/technology_model \
    --domain technology \
    --test_data data/prepared/technology_test.jsonl
```

This will:
- Load the fine-tuned model
- Translate all test examples
- Calculate BLEU and chrF scores
- Save results to `results/technology_results.json`

**Quick Translation Test:**
```bash
python evaluate.py \
    --model_path models/technology_model \
    --domain technology \
    --translate_text "The application provides a user-friendly interface."
```

### 5. Launch Streamlit UI

Start the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 🖥️ Streamlit UI Features

The web interface provides four main tabs:

### 1. **Translate Tab**
- Single text translation with selected domain model
- Adjustable generation parameters (temperature, max tokens)
- Sample texts for quick testing
- Translation time and length metrics

### 2. **Compare Domains Tab**
- Translate the same text using all domain models
- Side-by-side comparison of outputs
- Identify which domain model works best for ambiguous text

### 3. **Evaluation Tab**
- View BLEU and chrF scores for all models
- Interactive charts and visualizations
- Example translations from test set
- Performance comparison across domains

### 4. **History Tab**
- View recent translation history
- Statistics on usage patterns
- Quick access to previous translations

## 📊 Model Architecture

### Base Model
- **QWEN 2 (1.5B parameters)**: Fast and efficient
- **QWEN 2 (7B parameters)**: Higher quality (optional)

### Fine-tuning Approach
- **Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (QLoRA) for memory efficiency
- **Target Modules**: Query, Key, Value, and Feed-forward layers
- **LoRA Rank**: 16 (configurable)
- **LoRA Alpha**: 32

### Training Configuration
- **Optimizer**: Paged AdamW (8-bit)
- **Learning Rate**: 2e-4 with warmup
- **Batch Size**: 4 per device × 4 gradient accumulation = effective 16
- **Precision**: BFloat16 mixed precision
- **Gradient Checkpointing**: Enabled for memory efficiency

## 📈 Expected Performance

Based on typical fine-tuning results for translation tasks:

| Domain | Expected BLEU | Expected chrF |
|--------|--------------|---------------|
| Technology | 35-45 | 60-70 |
| Economic | 30-40 | 55-65 |
| Education | 32-42 | 58-68 |

**Note**: Actual scores depend on data quality, training time, and model size.

## 🔧 Customization

### Adjust Training Parameters

Edit hyperparameters in `train_qwen.py` or pass them as arguments:

```bash
python train_qwen.py \
    --domain technology \
    --num_train_epochs 5 \
    --learning_rate 3e-4 \
    --lora_r 32 \
    --lora_alpha 64
```

### Use Different Base Model

```bash
python train_qwen.py \
    --domain technology \
    --model_name Qwen/Qwen2-7B-Instruct
```

### Limit Training Data (for testing)

Modify `prepare_data.py` line 163:
```python
preparator.prepare_all_domains(test_size=0.1, val_size=0.1, max_samples=1000)
```

## 🐛 Troubleshooting

### Out of Memory Error

1. Reduce batch size:
   ```bash
   --per_device_train_batch_size 2
   ```

2. Reduce sequence length:
   ```bash
   --max_seq_length 256
   ```

3. Ensure 4-bit quantization is enabled:
   ```bash
   --use_4bit
   ```

### Slow Training

- Use smaller model (1.5B instead of 7B)
- Reduce number of training samples for testing
- Ensure CUDA is properly installed
- Check GPU utilization: `nvidia-smi`

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

For specific packages:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes
```

### Model Loading Error

If using Hugging Face models, you may need to login:
```bash
huggingface-cli login
```

## 📝 Data Format

### Input CSV Format
Your data should be in CSV format with columns:
- `id`: Unique identifier (optional)
- `english`: English text
- `arabic`: Arabic translation

Example:
```csv
id,english,arabic
1,Application,تطبيق
2,The user interface is intuitive,واجهة المستخدم بديهية
```

### Prepared Dataset Format (JSONL)
After preparation, data is converted to JSONL with QWEN chat template:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert translator..."},
    {"role": "user", "content": "Translate the following English text to Arabic..."},
    {"role": "assistant", "content": "الترجمة العربية"}
  ]
}
```

## 🚀 Advanced Usage

### Resume Training from Checkpoint

```bash
python train_qwen.py \
    --domain technology \
    --output_dir models/technology_model \
    --resume_from_checkpoint models/technology_model/checkpoint-500
```

### Multi-GPU Training

The scripts automatically use all available GPUs with `device_map="auto"`. For explicit control:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_qwen.py --domain technology
```

### Export for Production

After training, merge LoRA weights with base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "models/technology_model")
merged = model.merge_and_unload()
merged.save_pretrained("models/technology_merged")
```

## 📚 References

- [QWEN 2 Model](https://github.com/QwenLM/Qwen2)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT](https://github.com/huggingface/peft)

## 📄 License

This project uses the following components:
- QWEN models are subject to their respective licenses
- Training code is provided as-is for research and educational purposes
- Your data remains your property

## 🤝 Contributing

To improve this pipeline:
1. Test with different domains
2. Experiment with hyperparameters
3. Add new evaluation metrics
4. Enhance the Streamlit UI

## 💡 Tips for Best Results

1. **Data Quality**: Clean and preprocess your data before training
2. **Domain Specificity**: Ensure training data truly represents the domain
3. **Epochs**: Start with 3 epochs, increase if underfitting
4. **Learning Rate**: 2e-4 works well for LoRA, adjust if loss plateaus
5. **Evaluation**: Always evaluate on held-out test set
6. **Monitoring**: Use TensorBoard to track training progress:
   ```bash
   tensorboard --logdir models/technology_model/runs
   ```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are correctly installed
4. Verify data format matches expected structure

---

**Happy Training! 🚀**
