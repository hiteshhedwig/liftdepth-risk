#!/usr/bin/env bash
set -euo pipefail

# Run inside your conda/venv environment.
# Example:
#   conda create -n liftdepth python=3.10 -y
#   conda activate liftdepth
#   bash scripts/setup_env.sh

echo "Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch manually first if you need a specific CUDA version.
# Example for CUDA 11.8:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#
# If torch is already installed, this will leave it alone in most environments.
pip install numpy opencv-python pillow tqdm matplotlib
pip install ultralytics
pip install transformers accelerate safetensors

echo
echo "Cloning Depth-Anything-V2 if missing..."
mkdir -p external checkpoints

if [ ! -d external/Depth-Anything-V2 ]; then
  git clone https://github.com/DepthAnything/Depth-Anything-V2 external/Depth-Anything-V2
else
  echo "external/Depth-Anything-V2 already exists."
fi

echo
echo "Installing DA2 metric-depth requirements..."
if [ -f external/Depth-Anything-V2/metric_depth/requirements.txt ]; then
  pip install -r external/Depth-Anything-V2/metric_depth/requirements.txt || true
fi

echo
echo "Downloading DA2 metric VKITTI ViT-S checkpoint if missing..."
CKPT="checkpoints/depth_anything_v2_metric_vkitti_vits.pth"

if [ ! -f "$CKPT" ]; then
  wget -c "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true" -O "$CKPT"
else
  echo "$CKPT already exists."
fi

echo
echo "Setup complete."
