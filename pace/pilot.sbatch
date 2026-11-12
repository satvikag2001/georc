#!/usr/bin/env bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=2:00:00
# Pilot: seed-0 pilot chains for each condition (default G0 G0R A1), then one scoring run.
# An optional leading --variant paper|repo picks the judge rubric (default paper), e.g.
#   submit.sh pilot.sbatch --variant repo G0 A1
set -euo pipefail
source "$HOME/scratch/georc/code/pace/common.sh"
export HF_HUB_OFFLINE=1
VARIANT=paper
if [ "${1:-}" = "--variant" ]; then
  VARIANT="${2:-}"
  shift 2 || shift
fi
case "$VARIANT" in
  paper | repo) ;;
  *) echo "pilot.sbatch: --variant must be paper or repo, got '$VARIANT'" >&2; exit 2 ;;
esac
[ "$#" -gt 0 ] || set -- G0 G0R A1
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
DIRS=()
for ID in "$@"; do
  "$PY" "$GEORC_CODE/scripts/generate_chains.py" --data "$GEORC_DATA" --out "$GEORC_OUT" \
    --condition "$ID" --seeds 0 --pilot
  DIRS+=("$ID=$GEORC_OUT/chains/$ID/seed0")
done
"$PY" "$GEORC_CODE/scripts/score_chains.py" --data "$GEORC_DATA" --out "$GEORC_OUT" \
  --variant "$VARIANT" --pilot --seed 0 --dirs "${DIRS[@]}"
