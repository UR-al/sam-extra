#!/bin/bash
# Provision the Python test dependencies so `pytest` and the JS `node --check`
# work in Claude Code on the web sessions. This repo is a Forge (Stable
# Diffusion WebUI) extension whose regression tests import torch / gradio /
# pydantic, none of which ship with the base image.
set -euo pipefail

# Only run in remote (Claude Code on the web) sessions; local checkouts already
# have the Forge environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Idempotent: if every test dependency already imports, there is nothing to do
# (the container image is cached after the first run).
if python -c "import torch, gradio, pydantic, numpy, PIL, pytest, fastapi" >/dev/null 2>&1; then
  exit 0
fi

python -m pip install --upgrade pip >/dev/null 2>&1 || true

# CPU-only torch keeps the install small; fall back to the default index if the
# CPU wheel host is unreachable behind a given network policy.
pip install torch --index-url https://download.pytorch.org/whl/cpu \
  || pip install torch

pip install pytest "gradio>=4.0,<6.0" pydantic numpy "Pillow>=11.1.0" fastapi
