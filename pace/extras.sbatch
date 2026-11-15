#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=2:00:00
# Extra judge passes for alternative metrics; arguments go to scripts/judge_extras.py, e.g.
#   submit.sh extras.sbatch --pass pairwise --sets qwen
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
"$PY" "$GEORC_CODE/scripts/judge_extras.py" --data "$GEORC_DATA" --out "$GEORC_OUT" "$@"
