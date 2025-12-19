# NVIDIA Docker Setup for RTX 5090 Support

This guide helps you set up NVIDIA PyTorch Docker containers that support RTX 5090 GPUs (compute capability 12.0).

## Why Docker for RTX 5090?

The RTX 5090 GPU (launched January 2025) has compute capability 12.0 (sm_120), which is **not supported** by:
- PyTorch 2.5.x, 2.7.x stable releases
- PyTorch nightlies before March 2025 (most don't have sm_120 compiled in)

**Solution**: NVIDIA's official PyTorch containers include the latest CUDA and PyTorch builds optimized for newest GPUs.

## Prerequisites

1. **NVIDIA Driver**: You already have driver 570.86.16 (supports RTX 5090) ✓
2. **Ubuntu 24.04**: Already installed ✓
3. **Internet connection**: Required for installation

## Quick Setup (3 Steps)

### Step 1: Install Docker & NVIDIA Container Toolkit

**Wait for your DNS to stabilize**, then run:

```bash
cd "/home/aya/Desktop/Context_Specific_Machine_Translation_to_Arabic_Language/Reinforcement Learning"
bash setup_nvidia_docker.sh
```

This script will:
- Install Docker
- Install NVIDIA Container Toolkit
- Add your user to docker group
- Pull the NVIDIA PyTorch container
- Test GPU access

**Time**: ~10-15 minutes

### Step 2: Logout/Login (Important!)

After installation, you need to logout and login again for docker group membership to take effect:

```bash
# Logout from your desktop session, then login again
# Or restart your computer
```

### Step 3: Launch Container

```bash
cd "/home/aya/Desktop/Context_Specific_Machine_Translation_to_Arabic_Language/Reinforcement Learning"
./run_container.sh
```

## What You Get

The NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:25.01-py3`) includes:
- **PyTorch**: Latest version with sm_120 support
- **CUDA**: 12.4+ (supports RTX 5090)
- **cuDNN**: Latest optimized version
- **Python**: 3.10+
- **Pre-installed**: NumPy, Pandas, Pillow, OpenCV

## Working in the Container

### Your Workspace

Your project directory is mounted at `/workspace` inside the container:
- All your notebooks and code are accessible
- Changes you make are persistent (saved on your host machine)
- Both GPUs are accessible

### Run Jupyter Notebook

Inside the container:

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open in your browser: `http://localhost:8888`

### Verify RTX 5090 Works

```python
import torch

print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('GPU 0:', torch.cuda.get_device_name(0))
print('GPU 1:', torch.cuda.get_device_name(1))
print('Compute capability:', torch.cuda.get_device_capability(0))

# Test CUDA operations
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
y = x * 2
print('CUDA test:', y.cpu().numpy())  # Should print: [2. 4. 6.]
```

**Expected output**:
```
PyTorch version: 2.7.0 (or newer)
CUDA version: 12.4
GPU 0: NVIDIA GeForce RTX 5090
GPU 1: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
CUDA test: [2. 4. 6.]
```

### Exit Container

```bash
exit
```

Your work is saved! Re-launch anytime with `./run_container.sh`

## Troubleshooting

### DNS Issues During Installation

If you see "Could not resolve host" errors:

1. **Wait**: Your network connection has intermittent DNS issues
2. **Check**: `ping 8.8.8.8` (should work) and `ping google.com` (may fail)
3. **Fix DNS temporarily**:
   ```bash
   echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
   ```
4. **Retry**: Run `bash setup_nvidia_docker.sh` again

### Docker Permission Denied

If you see "permission denied" when running docker:

```bash
# You forgot to logout/login after installation
# Either:
newgrp docker  # temporary fix for current session
# Or logout and login again (permanent fix)
```

### GPU Not Available in Container

```bash
# Check NVIDIA Container Toolkit is installed
nvidia-ctk --version

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Alternative: Different Container Versions

If the recommended container has issues, try these alternatives:

```bash
# Latest nightly (most up-to-date)
docker pull nvcr.io/nvidia/pytorch:nightly

# December 2024 version
docker pull nvcr.io/nvidia/pytorch:24.12-py3

# January 2025 version
docker pull nvcr.io/nvidia/pytorch:25.01-py3
```

Then update `run_container.sh` to use your chosen image.

## Container Management

### List Running Containers
```bash
docker ps
```

### Stop Container
```bash
docker stop pytorch_rtx5090
```

### Remove Old Images (free space)
```bash
docker image prune -a
```

### Check Container Logs
```bash
docker logs pytorch_rtx5090
```

## Performance Tips

### 1. Shared Memory Size

The container uses `--shm-size=32g` for efficient data loading. Adjust based on your RAM:

```bash
# If you have 128GB+ RAM
--shm-size=64g

# If you have 64GB RAM
--shm-size=16g
```

### 2. Multi-GPU Utilization

Inside container, verify both GPUs are visible:

```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
```

For distributed training:
```python
# Model parallelism (layers split across GPUs)
model = AutoModelForCausalLM.from_pretrained(
    "ModelSpace/GemmaX2-28-9B-v0.1",
    device_map="auto",  # Automatically uses both GPUs
    load_in_8bit=True
)
```

### 3. Persistent Pip Packages

Create a requirements file in your project:

```bash
# Inside container
pip freeze > requirements.txt

# Next time you launch container
pip install -r requirements.txt
```

## Next Steps

Once container is running:

1. **Verify GPU works** with test script above
2. **Open Jupyter** and run your notebook
3. **The CUDA errors should be gone** - RTX 5090 will work!

## Resources

- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NVIDIA Container Toolkit Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)

## Summary

✓ **Installation**: Run `setup_nvidia_docker.sh`  
✓ **Logout/Login**: Required for docker group  
✓ **Launch**: Run `./run_container.sh`  
✓ **Work**: Full RTX 5090 support with latest PyTorch  
✓ **Benefits**: No compilation, guaranteed compatibility, easy updates
