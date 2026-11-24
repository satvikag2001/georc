#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=1:00:00
# Pool test; arguments go to scripts/pool_test.py, e.g. submit.sh pool.sbatch --sets qwen hallucinated
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
"$PY" "$GEORC_CODE/scripts/pool_test.py" --data "$GEORC_DATA" --out "$GEORC_OUT" "$@"
