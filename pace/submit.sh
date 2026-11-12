#!/usr/bin/env bash
# Submit a job on PACE and print its id; --after JOBID runs it only if JOBID succeeds.
#   submit.sh --after 123 score.sbatch --released qwen gemma3_12b
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
