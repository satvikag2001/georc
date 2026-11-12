#!/usr/bin/env bash
# Shared PACE settings, sourced by every pace/ script. Everything this project writes lives under
# $GEORC_ROOT (cleanup removes exactly that), never in $HOME (20 GB quota).
export GEORC_ROOT="${GEORC_ROOT:-$HOME/scratch/georc}"
export GEORC_CODE="$GEORC_ROOT/code"
export GEORC_VENV="$GEORC_ROOT/venv"
export GEORC_DATA="$GEORC_ROOT/data/georc"
export GEORC_OUT="$GEORC_ROOT/out"
export GEORC_LOGS="$GEORC_ROOT/logs"
export PY="$GEORC_VENV/bin/python"

# venv tools first (e.g. ninja), then uv, which is installed here and not in ~/.local/bin
export PATH="$GEORC_VENV/bin:$GEORC_ROOT/bin:$PATH"
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
export FLASHINFER_WORKSPACE_BASE="$GEORC_ROOT/cache"  # else ~/.cache/flashinfer
export TILELANG_CACHE_DIR="$GEORC_ROOT/cache/tilelang" TILELANG_TMP_DIR="$GEORC_ROOT/cache/tilelang/tmp"
# Inside a job, HOME on scratch too, for libraries that default to ~.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export HOME="$GEORC_ROOT/home"
  mkdir -p "$HOME"
fi
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# Job 15325038: the FlashInfer top-k/top-p sampler JIT-compiles on first use and compute nodes
# have no compiler toolchain; the PyTorch sampler needs none.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Job 15325037: "CUDA device busy or unavailable" on an H100. One process per GPU is safe even
# when the GPU only admits one process (exclusive compute mode).
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Free tier only. gts-awu36-paid and the inferno QoS cost money and are never used.
export PACE_ACCOUNT="gts-awu36"
export PACE_QOS="embers"
export PACE_PARTITIONS="gpu-a100,gpu-h100,gpu-h200,gpu-l40s,gpu-rtxpro-blackwell"  # bf16-capable, >=40 GB
# Job 15325036 still landed on a V100 (no bf16) despite the partition list; node features
# are a hard filter.
export PACE_CONSTRAINT="A100|H100|H200|L40S|RTX-Pro-Blackwell"
# Jobs 15449667-15459955: one Blackwell node kept failing with "CUDA device busy" on most of
# its GPUs (24 of 28 jobs lost). Comma-separated nodes to avoid; empty for none.
export PACE_EXCLUDE="${PACE_EXCLUDE-atl1-1-03-020-6-0}"
export JOB_PREFIX="georc-"

if [ -f "$GEORC_ROOT/.hf_token" ]; then
  HF_TOKEN="$(cat "$GEORC_ROOT/.hf_token")"
  export HF_TOKEN
fi
mkdir -p "$GEORC_ROOT"/{bin,cache,tmp,logs,out} 2>/dev/null || true
