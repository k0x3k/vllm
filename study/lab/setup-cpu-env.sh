#!/usr/bin/env bash
# Build a CPU-only vLLM environment for reading and stepping through the engine.
#
# No GPU, no CUDA, no kernel compilation: ~1.5 GB installed. Enough to run the
# real Scheduler, KVCacheManager and BlockPool against real tokenized prompts,
# and to run the pure-Python test suites under tests/v1/core.
#
# Two traps this avoids:
#   1. pyproject's build-system pins plain `torch == 2.13.0`, so under build
#      isolation pip/uv resolves the ~3 GB CUDA wheel. Hence --no-build-isolation
#      with the build deps installed by hand against the CPU index.
#   2. VLLM_TARGET_DEVICE=empty makes setup.py read requirements/common.txt,
#      which lists no torch at all — so the CPU torch installed first survives.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO"
CPU_INDEX=https://download.pytorch.org/whl/cpu
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
uv pip install --no-cache "torch==2.13.0+cpu" --index-url "$CPU_INDEX"
uv pip install --no-cache -r requirements/build/cpu.txt \
  --extra-index-url "$CPU_INDEX" --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=empty uv pip install --no-cache -e . --no-build-isolation \
  --extra-index-url "$CPU_INDEX" --index-strategy unsafe-best-match
uv pip install --no-cache pytest pytest-asyncio tblib

echo
echo "installed: $(du -sh "$REPO/.venv" | cut -f1)"
.venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"
