#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=2:00:00
# Judge-score chain sets; arguments go to scripts/score_chains.py, e.g.
#   submit.sh score.sbatch --released qwen gemma3_12b llama3_11b qwen3 hallucinated --seed 0
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
"$PY" "$GEORC_CODE/scripts/score_chains.py" --data "$GEORC_DATA" --out "$GEORC_OUT" "$@"
