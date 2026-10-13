#!/usr/bin/env bash
# Builds the `occamlgs` env for insait-institute/OccamLGS (cloned at ~/Radiance/third_party/OccamLGS,
# submodule gsplat = JoannaCCJH/gsplat, a gsplat 1.4.0 fork that returns per-Gaussian centre weights).
# Upstream: python 3.10, torch 2.2.2 + cu121; we use the cu118 wheel to match the conda nvcc 11.8.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HOME/Radiance/figs/miniconda3/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="$HOME/Radiance/figs/miniconda3/pkgs" TMPDIR="$HOME/Radiance/figs/.tmp"
conda env list | grep -q '^occamlgs ' || conda create -y -n occamlgs python=3.10
conda install -y -n occamlgs -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
source "$HERE/env.sh" occamlgs
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" open-clip-torch plyfile tqdm opencv-python-headless tensorboard jaxtyping matplotlib \
    mediapy scikit-learn ninja
cd "$THIRD_PARTY/OccamLGS/submodules"
BUILD_NO_CUDA=0 MAX_JOBS=4 pip install --no-build-isolation -e ./gsplat
pip install --no-build-isolation ./simple-knn
python - <<'PY'
import torch, gsplat, simple_knn
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(), "gsplat", gsplat.__file__)
PY

# Occam's full pipeline: LangSplat's preprocessing (SAM ViT-H at levels default / s / m / l + OpenCLIP per mask)
#   ~/Radiance/third_party/LangSplat                     (git clone https://github.com/minghanqin/LangSplat)
#   ~/Radiance/third_party/segment-anything-langsplat    (git clone https://github.com/minghanqin/segment-anything-langsplat)
#   ~/Radiance/third_party/checkpoints/sam_vit_h_4b8939.pth
pip install --no-deps -e "$THIRD_PARTY/segment-anything-langsplat"
