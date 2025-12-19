#!/bin/bash
# Setup NVIDIA Docker for RTX 5090 Support
# Run this script when DNS is working: bash setup_nvidia_docker.sh

set -e

echo "=================================="
echo "NVIDIA Docker Setup for RTX 5090"
echo "=================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run as root. The script will use sudo when needed."
   exit 1
fi

# 1. Install Docker
echo ""
echo "Step 1: Installing Docker..."
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

echo "Docker installed successfully!"

# 2. Add user to docker group
echo ""
echo "Step 2: Adding user to docker group..."
sudo usermod -aG docker $USER
echo "User added to docker group. You may need to logout/login for this to take effect."

# 3. Install NVIDIA Container Toolkit
echo ""
echo "Step 3: Installing NVIDIA Container Toolkit..."

# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Use the generic DEB repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "NVIDIA Container Toolkit installed successfully!"

# 4. Test Docker with GPU
echo ""
echo "Step 4: Testing Docker with GPU access..."
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 5. Pull NVIDIA PyTorch container
echo ""
echo "Step 5: Pulling NVIDIA PyTorch container (this may take a while)..."
docker pull nvcr.io/nvidia/pytorch:25.01-py3

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To run the container, use:"
echo "  cd \"$PWD\""
echo "  docker run --gpus all -it --rm -v \"\$PWD:/workspace\" -w /workspace --shm-size=32g nvcr.io/nvidia/pytorch:25.01-py3 bash"
echo ""
echo "Inside the container, verify GPU:"
echo "  python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\""
echo ""
echo "Note: You may need to logout/login for docker group membership to take effect."
