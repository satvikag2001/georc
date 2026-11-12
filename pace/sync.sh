#!/usr/bin/env bash
# Push local code to PACE. Outputs live outside $GEORC_ROOT/code, so --delete cannot touch them.
set -euo pipefail
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-pace}"
REMOTE_DIR="scratch/georc/code"
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
rsync -az --delete \
  --exclude /.git/ --exclude /.venv/ --exclude /data/ --exclude /runs/ \
  --exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ \
  --exclude /CLAUDE.md --exclude /.claude/ \
  "$LOCAL/" "$REMOTE_HOST:$REMOTE_DIR/"
echo "synced -> $REMOTE_HOST:$REMOTE_DIR"
