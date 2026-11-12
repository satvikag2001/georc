#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=1:00:00
# GPU smoke test: offline model loading and one generation per model and mode.
#   submit.sh preflight.sbatch [model ...]   # models: judge qwen-vl gemma (default: all three)
# One process per model, so GPU memory is fully released between them.
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(judge qwen-vl gemma)
for model in "${models[@]}"; do
  echo "===== $model"
  "$PY" "$GEORC_CODE/scripts/preflight.py" --data "$GEORC_DATA" --model "$model"
done
