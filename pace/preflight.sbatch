#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=8:00:00
# GPU smoke test: driver and CUDA, offline model loading, one generation per model and mode,
# one judge call. Submit with pace/submit.sh, which sets account, QoS, partitions and name.
#   submit.sh preflight.sbatch [model ...]   # models: judge qwen-vl gemma (default: all three)
# Each model runs in its own process so GPU memory is fully released between them.
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(judge qwen-vl gemma)
for model in "${models[@]}"; do
  echo "===== $model"
  "$PY" "$GEORC_CODE/scripts/preflight.py" --data "$GEORC_DATA" --model "$model"
done
