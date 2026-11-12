#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=2:00:00
# The authors' own scorer on the released chains; arguments go to scripts/authors_score.py, e.g.
#   submit.sh authors.sbatch --tags qwen3 --shard 1/2
# Exits 0 at its time budget, so `submit.sh --after JOBID authors.sbatch ...` continues the work.
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
AUTHORS="$GEORC_ROOT/authors/GeoRC"
if [ ! -f "$AUTHORS/src/vllm/vllm_completion_util.py" ]; then
  echo "authors.sbatch: no patched clone at $AUTHORS; run pace/fetch_authors.sh on the login node" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv,noheader
"$PY" "$GEORC_CODE/scripts/authors_score.py" --authors "$AUTHORS" --data "$GEORC_DATA" \
  --out "$GEORC_OUT/authors" --cache-dir "$HF_HOME/hub" "$@"
