#!/usr/bin/env bash
# Download the dataset and pinned model weights on the PACE login node; compute nodes have no
# internet, so jobs run with HF_HUB_OFFLINE=1 against this cache ($HF_HOME on scratch).
#   fetch_assets.sh                 # everything
#   fetch_assets.sh dataset judge   # a subset: dataset | qwen-vl | judge | gemma
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(dataset qwen-vl judge gemma)

fetch_model() {
  "$PY" - "$1" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from georc.models import GENERATORS, JUDGE
spec = {"qwen-vl": GENERATORS["qwen2.5-vl-7b"], "gemma": GENERATORS["gemma3-12b"], "judge": JUDGE}[sys.argv[1]]
print(snapshot_download(spec.repo, revision=spec.revision))
EOF
}

for target in "${targets[@]}"; do
  echo "== $target"
  case "$target" in
    dataset) "$PY" "$GEORC_CODE/scripts/fetch_dataset.py" --out "$GEORC_DATA" ;;
    qwen-vl | judge) fetch_model "$target" ;;
    gemma)
      [ -n "${HF_TOKEN:-}" ] || { echo "gemma is gated: put a read token in $GEORC_ROOT/.hf_token" >&2; exit 1; }
      fetch_model gemma ;;
    *) echo "unknown target: $target" >&2; exit 1 ;;
  esac
done
du -sh "$HF_HOME" "$GEORC_ROOT/data" 2>/dev/null || true
