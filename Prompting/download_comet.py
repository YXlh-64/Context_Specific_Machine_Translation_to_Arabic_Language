"""
COMET Model Downloader
=====================

Downloads the COMET neural evaluation model (1-2 GB).
Run this script once with internet connection.

The model will be cached locally and reused for all future evaluations.
"""

from comet import download_model, load_from_checkpoint

print("="*80)
print("COMET MODEL DOWNLOADER")
print("="*80)
print()
print("⚠️  This will download ~1-2 GB of data")
print("📡 Requires internet connection")
print("⏱️  May take 5-10 minutes depending on your connection")
print()
print("The model will be cached at:")
print("  Windows: C:\\Users\\{username}\\.cache\\huggingface\\hub\\")
print("  Linux/Mac: ~/.cache/huggingface/hub/")
print()

input("Press ENTER to start download, or Ctrl+C to cancel...")

print("\n" + "="*80)
print("🔽 DOWNLOADING COMET MODEL...")
print("="*80)

try:
    # Download the model
    model_path = download_model("Unbabel/wmt22-comet-da")
    
    print(f"\n✅ Model downloaded successfully!")
    print(f"📂 Location: {model_path}")
    
    # Test loading the model
    print("\n" + "="*80)
    print("🧪 TESTING MODEL LOAD...")
    print("="*80)
    
    model = load_from_checkpoint(model_path)
    
    print("\n✅ COMET model is ready to use!")
    print("\n" + "="*80)
    print("SUCCESS - YOU CAN NOW RUN demo.py")
    print("="*80)
    print("\nThe model is cached and will be reused for all future evaluations.")
    print("No need to download again unless you clear the cache.")
    
except Exception as e:
    print(f"\n❌ Error downloading COMET model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Make sure you have enough disk space (~2 GB)")
    print("3. Try running: pip install --upgrade unbabel-comet torch")
    print("\nAlternatively, you can skip COMET:")
    print("- chrF++ and BLEU metrics will still work")
    print("- COMET will show as 'N/A' in reports")
    print("- Combined score uses: 0.6×chrF++ + 0.4×BLEU")
