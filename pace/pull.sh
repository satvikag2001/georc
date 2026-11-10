#!/usr/bin/env bash
# Copy scores, logs, generated chains and cached judge replies from PACE into runs/georc/ on the laptop (gitignored).
# Read-only on PACE.
set -euo pipefail
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-pace}"
# A fresh connection each time: a stale shared SSH connection has hung this before.
export RSYNC_RSH="ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15"
mkdir -p "$LOCAL/runs/georc"
rsync -az "$REMOTE_HOST:scratch/georc/out/scores" "$REMOTE_HOST:scratch/georc/logs" \
  "$LOCAL/runs/georc/"
# No chains exist before the first generation job; out/ itself exists once scores do.
rsync -az --ignore-missing-args "$REMOTE_HOST:scratch/georc/out/chains" \
  "$REMOTE_HOST:scratch/georc/out/judge_cache" "$LOCAL/runs/georc/"
echo "pulled -> $LOCAL/runs/georc"
