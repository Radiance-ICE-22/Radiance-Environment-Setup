#!/usr/bin/env bash
# Source to work in a baseline env:   source semantics/baselines/env.sh ludvig|occamlgs
# Separate conda envs next to kitchen (same miniconda under ~/Radiance), never kitchen itself.
_name="${1:?usage: source env.sh ludvig|occamlgs}"
case $- in *u*) _had_u=1;; *) _had_u=0;; esac
set +u
source "$HOME/Radiance/figs/miniconda3/etc/profile.d/conda.sh"
conda activate "$_name"
[ "$_had_u" = 1 ] && set -u
export PYTHONNOUSERSITE=1 PIP_USER=0
export PIP_CACHE_DIR="$HOME/Radiance/figs/.cache/pip"
export CONDA_PKGS_DIRS="$HOME/Radiance/figs/miniconda3/pkgs"
export TMPDIR="$HOME/Radiance/figs/.tmp"
export XDG_CACHE_HOME="$HOME/Radiance/third_party/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/huggingface"
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.5" CC=/usr/bin/gcc CXX=/usr/bin/g++      # RTX 2080 (sm_75), gcc 11.4
export THIRD_PARTY="$HOME/Radiance/third_party"
unset _name _had_u
