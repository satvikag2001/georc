#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --time=0:45:00
# Location-versus-truth check for a pool test's sets; arguments go to scripts/pool_truth.py.
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
"$PY" "$GEORC_CODE/scripts/pool_truth.py" --data "$GEORC_DATA" --out "$GEORC_OUT" "$@"
