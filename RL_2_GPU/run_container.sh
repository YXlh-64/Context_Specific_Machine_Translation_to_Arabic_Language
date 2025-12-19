#!/bin/bash
# Launch NVIDIA PyTorch Docker Container for RTX 5090
# Usage: ./run_container.sh

CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.01-py3"
WORKSPACE_DIR="$PWD"

echo "Launching NVIDIA PyTorch Container..."
echo "Image: $CONTAINER_IMAGE"
echo "Workspace: $WORKSPACE_DIR"
echo ""

docker run --gpus all -it --rm \
  -v "$WORKSPACE_DIR:/workspace" \
  -w /workspace \
  --shm-size=32g \
  -p 8888:8888 \
  -p 6006:6006 \
  --name pytorch_rtx5090 \
  $CONTAINER_IMAGE \
  bash -c "
    echo '========================================';
    echo 'NVIDIA PyTorch Container for RTX 5090';
    echo '========================================';
    echo '';
    echo 'Checking PyTorch and GPU...';
    python -c \"
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Number of GPUs:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Compute capability: {torch.cuda.get_device_capability(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Total memory: {props.total_memory / 1e9:.2f} GB')
    # Test CUDA operations
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x * 2
    print('')
    print('✓ RTX 5090 CUDA test successful!')
    print('  Test tensor:', x.cpu().numpy())
    print('  Result:', y.cpu().numpy())
else:
    print('⚠ CUDA not available!')
\";
    echo '';
    echo '========================================';
    echo 'Installing project dependencies...';
    echo '========================================';
    pip install -q transformers accelerate bitsandbytes tqdm jupyter ipykernel 2>/dev/null;
    echo '✓ Dependencies installed';
    echo '';
    echo '========================================';
    echo 'Ready to work!';
    echo '========================================';
    echo '';
    echo 'Quick commands:';
    echo '  - Run notebook: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root';
    echo '  - Python shell: python';
    echo '  - List files: ls -lh';
    echo '  - Exit container: exit';
    echo '';
    bash
  "
