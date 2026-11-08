#!/usr/bin/env bash
# One-time environment build on the PACE login node (compute nodes have no internet).
# Safe to re-run. Installs uv, Python 3.12, vLLM and this package, all under $GEORC_ROOT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
VLLM_VERSION="${VLLM_VERSION:?set VLLM_VERSION, e.g. VLLM_VERSION=0.11.0}"

if [ ! -x "$GEORC_ROOT/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh |
    env UV_INSTALL_DIR="$GEORC_ROOT/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv python install 3.12
[ -x "$PY" ] || uv venv --python 3.12 "$GEORC_VENV"
uv pip install --python "$PY" "vllm==$VLLM_VERSION" pillow
uv pip install --python "$PY" -e "$GEORC_CODE"
"$PY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
