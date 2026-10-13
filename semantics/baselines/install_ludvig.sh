#!/usr/bin/env bash
# Builds the `ludvig` env for naver/ludvig (cloned at ~/Radiance/third_party/ludvig).
# Upstream: python 3.11, torch 2.4.0 + cu118. SAM / SAM2 are left out (only needed for
# LUDVIG's SAM-based segmentation evaluation, not for the lift or its CLIP/DINOv2 diffusion).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HOME/Radiance/figs/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="$HOME/Radiance/figs/miniconda3/pkgs" TMPDIR="$HOME/Radiance/figs/.tmp"
conda env list | grep -q '^ludvig ' || conda create -y -n ludvig python=3.11
conda install -y -n ludvig -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
source "$HERE/env.sh" ludvig
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" open-clip-torch jaxtyping matplotlib omegaconf opencv-python-headless Pillow plyfile \
    PyYAML scikit-learn scikit-image scipy tqdm
cd "$THIRD_PARTY/ludvig/gaussiansplatting/submodules"
pip install --no-build-isolation ./diff-gaussian-rasterization ./simple-knn
python - <<'PY'
import torch, diff_gaussian_rasterization, simple_knn
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
PY
