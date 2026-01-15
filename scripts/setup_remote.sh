#!/bin/bash
set -e  # Exit on error

echo "======================================"
echo "Searvids AWS Server Setup Script"
echo "======================================"

# Wait for AWS cloud-init to complete (This finishes auto-updates and releases apt locks)
echo "[0/6] Waiting for cloud-init to complete..."
cloud-init status --wait

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update -y

# Install basic tools
echo "[2/6] Installing basic tools..."
sudo apt-get install -y git build-essential curl wget

# Install NVIDIA drivers
echo "[3/6] Installing NVIDIA drivers..."
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Install Docker
echo "[4/6] Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
rm get-docker.sh

# Install NVIDIA Container Toolkit
# Enable verbose execution for debugging
set -x

# Install NVIDIA Container Toolkit
echo "[5/6] Installing NVIDIA Container Toolkit..."
# Hardcode distribution to avoid detection errors
distribution="ubuntu22.04"

# Robust GPG Key Download
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Robust Repository List Download (Try stable/ubuntu22.04 explicit path first)
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Debug: Check file content
echo "DEBUG: Content of sources list:"
cat /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Verify file content (check for HTML error)
if grep -q "<!doctype" /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
    echo "❌ Error: Downloaded file is HTML (e.g., 404 page). Trying fallback..."
    # Fallback to direct distribution path
    curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi

sudo apt-get update
# sudo -E apt-get upgrade -y -o Dpkg::Options::="--force-confnew"
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU
echo "[6/6] Verifying GPU setup..."
nvidia-smi

echo ""
echo "✅ Server setup complete!"
echo "Next: Reboot required for driver activation"
