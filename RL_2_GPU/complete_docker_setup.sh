#!/bin/bash
# Complete Docker Setup - Run after logout/login
# Usage: bash complete_docker_setup.sh

echo "=========================================="
echo "Completing NVIDIA Docker Setup"
echo "=========================================="
echo ""

echo "Step 1: Verifying Docker group membership..."
groups | grep docker
if [ $? -eq 0 ]; then
    echo "✓ You are in the docker group"
else
    echo "✗ You are NOT in the docker group yet"
    echo "  Please logout and login again, then re-run this script"
    exit 1
fi

echo ""
echo "Step 2: Testing Docker access..."
docker ps > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Docker access works"
else
    echo "✗ Docker access failed"
    echo "  Try: newgrp docker"
    echo "  Or logout/login"
    exit 1
fi

echo ""
echo "Step 3: Testing GPU access in Docker..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
if [ $? -eq 0 ]; then
    echo "✓ GPU access in Docker works!"
else
    echo "✗ GPU access failed"
    exit 1
fi

echo ""
echo "Step 4: Pulling NVIDIA PyTorch container (this may take 10-20 minutes)..."
docker pull nvcr.io/nvidia/pytorch:25.01-py3

echo ""
echo "Step 5: Testing PyTorch with RTX 5090..."
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.01-py3 python -c "
import torch
print('========================================')
print('PyTorch Version:', torch.__version__)
print('CUDA Version:', torch.version.cuda)
print('CUDA Available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
print('')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    cap = torch.cuda.get_device_capability(i)
    print(f'  Compute Capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})')
    props = torch.cuda.get_device_properties(i)
    print(f'  Memory: {props.total_memory / 1e9:.2f} GB')
print('')
print('Testing CUDA operations...')
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
y = x * 2
print('Input:', x.cpu().numpy())
print('Output:', y.cpu().numpy())
print('')
print('✓✓✓ RTX 5090 FULLY WORKING WITH PYTORCH! ✓✓✓')
print('========================================')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS! Everything is working!"
    echo "=========================================="
    echo ""
    echo "Your RTX 5090 GPUs are fully compatible with PyTorch in Docker!"
    echo ""
    echo "To launch the container for your work:"
    echo "  ./run_container.sh"
    echo ""
    echo "Inside the container, your workspace will be at: /workspace"
    echo "All files are persistent (saved on host)"
    echo ""
else
    echo ""
    echo "PyTorch test failed. Check errors above."
    exit 1
fi
