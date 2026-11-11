#!/usr/bin/env bash
# Submit a GeoRC job on PACE (run on the login node, or from the laptop via `ssh pace`).
#   submit.sh preflight.sbatch
#   submit.sh --after 123 score.sbatch --released qwen gemma3_12b
# Always free tier: account gts-awu36 with QoS embers; jobs are named georc-<script>.
# --after JOBID starts the job only if JOBID succeeds, and cancels it if JOBID fails.
# Prints the job id.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DEPENDENCY=()
if [ "${1:-}" = "--after" ]; then
  DEPENDENCY=(--dependency="afterok:$2" --kill-on-invalid-dep=yes)
  shift 2
fi
SCRIPT="$1"
shift
NAME="${JOB_PREFIX}${SCRIPT%.sbatch}"
sbatch --parsable \
  --account="$PACE_ACCOUNT" --qos="$PACE_QOS" \
  --partition="$PACE_PARTITIONS" --gres=gpu:1 --constraint="$PACE_CONSTRAINT" \
  ${PACE_EXCLUDE:+--exclude="$PACE_EXCLUDE"} "${DEPENDENCY[@]}" \
  --job-name="$NAME" --output="$GEORC_LOGS/%x-%j.out" \
  "$GEORC_CODE/pace/$SCRIPT" "$@"
