#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=2:00:00
# Generate chains; arguments go to scripts/generate_chains.py, e.g.
#   submit.sh generate.sbatch --condition G0 --seeds 0 --pilot
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
"$PY" "$GEORC_CODE/scripts/generate_chains.py" --data "$GEORC_DATA" --out "$GEORC_OUT" "$@"
