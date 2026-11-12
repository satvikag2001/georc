#!/usr/bin/env bash
# Copy scores, logs, chains, judge cache and authors' scorer runs from PACE into runs/georc/.
set -euo pipefail
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-pace}"
# A fresh connection each time: a stale shared SSH connection has hung this before.
export RSYNC_RSH="ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15"
mkdir -p "$LOCAL/runs/georc"
rsync -az "$REMOTE_HOST:scratch/georc/out/scores" "$REMOTE_HOST:scratch/georc/logs" \
  "$LOCAL/runs/georc/"
# These may not exist yet; views/ holds only symlinks.
rsync -az --ignore-missing-args --exclude views/ "$REMOTE_HOST:scratch/georc/out/chains" \
  "$REMOTE_HOST:scratch/georc/out/judge_cache" "$REMOTE_HOST:scratch/georc/out/authors" \
  "$LOCAL/runs/georc/"
echo "pulled -> $LOCAL/runs/georc"
