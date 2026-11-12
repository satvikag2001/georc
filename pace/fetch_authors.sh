#!/usr/bin/env bash
# Login node: clone the authors' repository at a7a1679 and apply patches/authors_a7a1679.patch.
# Re-runnable: the clone is reset to the commit first, so an updated patch replaces the old one.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
URL="${GEORC_AUTHORS_URL:-https://github.com/talrejamohit03/GeoRC.git}"
COMMIT="a7a167979b3b355c29d59d36e3bff66c79f6a530"
DEST="$GEORC_ROOT/authors/GeoRC"
PATCH="$GEORC_CODE/patches/authors_a7a1679.patch"

[ -d "$DEST/.git" ] || git clone --quiet "$URL" "$DEST"
git -C "$DEST" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet --force --detach "$COMMIT"
git -C "$DEST" clean --quiet -fdx
git -C "$DEST" apply "$PATCH"
echo "$DEST: $(git -C "$DEST" rev-parse --short HEAD) + $(basename "$PATCH")"
git -C "$DEST" status --short

# Check the venv imports their scorer (sentence-transformers is optional after the patch).
"$PY" - "$DEST" <<'EOF'
import sys

sys.path.insert(0, sys.argv[1])
import score  # noqa: E402,F401  (pulls in every module the scorer uses)
import torch  # noqa: E402
import transformers  # noqa: E402

print("authors' score.py imports; torch", torch.__version__, "transformers", transformers.__version__)
EOF
