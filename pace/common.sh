#!/usr/bin/env bash
# Shared PACE settings for the GeoRC study; sourced by every pace/ script and sbatch file.
#
# Everything this project creates on PACE lives under $GEORC_ROOT: code, uv itself, Python,
# the venv, model weights, the dataset, every cache, outputs and logs. pace/cleanup.sh removes
# exactly that directory, so nothing may be written to $HOME (20 GB quota, 14.4 GB used).
export GEORC_ROOT="${GEORC_ROOT:-$HOME/scratch/georc}"
export GEORC_CODE="$GEORC_ROOT/code"
export GEORC_VENV="$GEORC_ROOT/venv"
export GEORC_DATA="$GEORC_ROOT/data/georc"
export GEORC_OUT="$GEORC_ROOT/out"
export GEORC_LOGS="$GEORC_ROOT/logs"
export PY="$GEORC_VENV/bin/python"

export PATH="$GEORC_ROOT/bin:$PATH"  # uv is installed here, not in ~/.local/bin
export UV_CACHE_DIR="$GEORC_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$GEORC_ROOT/python"
export UV_PYTHON_BIN_DIR="$GEORC_ROOT/bin"  # else uv links python3.12 into ~/.local/bin
export UV_TOOL_DIR="$GEORC_ROOT/uv-tools" UV_TOOL_BIN_DIR="$GEORC_ROOT/bin"
export PIP_CACHE_DIR="$GEORC_ROOT/cache/pip"
export HF_HOME="$GEORC_ROOT/hf"
export XDG_CACHE_HOME="$GEORC_ROOT/cache"
export VLLM_CACHE_ROOT="$GEORC_ROOT/cache/vllm"
export TRITON_CACHE_DIR="$GEORC_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$GEORC_ROOT/cache/inductor"
export CUDA_CACHE_PATH="$GEORC_ROOT/cache/nv"
export TMPDIR="$GEORC_ROOT/tmp"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# Free tier only. gts-awu36-paid and the inferno QoS cost money and are never used.
export PACE_ACCOUNT="gts-awu36"
export PACE_QOS="embers"
export PACE_PARTITIONS="gpu-a100,gpu-h100,gpu-h200,gpu-l40s"  # bf16-capable, >=40 GB
export JOB_PREFIX="georc-"

if [ -f "$GEORC_ROOT/.hf_token" ]; then
  HF_TOKEN="$(cat "$GEORC_ROOT/.hf_token")"
  export HF_TOKEN
fi
mkdir -p "$GEORC_ROOT"/{bin,cache,tmp,logs,out} 2>/dev/null || true
