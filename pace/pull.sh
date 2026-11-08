#!/usr/bin/env bash
# Copy scores and logs from PACE into runs/georc/ on the laptop (gitignored). Read-only on PACE.
set -euo pipefail
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-pace}"
mkdir -p "$LOCAL/runs/georc"
rsync -az "$REMOTE_HOST:scratch/georc/out/scores" "$REMOTE_HOST:scratch/georc/logs" \
  "$LOCAL/runs/georc/"
echo "pulled -> $LOCAL/runs/georc"
