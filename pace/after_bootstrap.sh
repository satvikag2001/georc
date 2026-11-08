#!/usr/bin/env bash
# Runs detached on the PACE login node: waits for setup and downloads to finish, then submits
# the preflight job and the G1 scorer-validation jobs (which start only if preflight passes).
# No local polling is needed. Gives up after 4 hours or if the bootstrap process dies.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG="$GEORC_LOGS/bootstrap.log"
deadline=$(( $(date +%s) + 4 * 3600 ))
until grep -q BOOTSTRAP_DONE "$LOG"; do
  if ! ps -u "$USER" -o cmd= | grep -q "[s]etup_env.sh && bash"; then
    echo "bootstrap is not running and did not finish; see $LOG" >&2
    exit 1
  fi
  [ "$(date +%s)" -lt "$deadline" ] || { echo "timed out waiting for bootstrap" >&2; exit 1; }
  sleep 60
done

PRE=$(bash "$GEORC_CODE/pace/submit.sh" preflight.sbatch)
echo "preflight job $PRE"
export SBATCH_DEPENDENCY="afterok:$PRE"
G1=$(bash "$GEORC_CODE/pace/submit.sh" score.sbatch \
  --released qwen gemma3_12b llama3_11b qwen3 hallucinated --seed 0)
echo "G1 scoring job $G1 (seed 0)"
NOISE=$(bash "$GEORC_CODE/pace/submit.sh" score.sbatch --released gemma3_12b --seed 1)
echo "judge-noise job $NOISE (gemma3_12b, seed 1)"
