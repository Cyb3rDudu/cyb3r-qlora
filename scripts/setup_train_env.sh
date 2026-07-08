#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. On carrier, rebuild the NixOS config or run: nix shell nixpkgs#uv nixpkgs#python312" >&2
  exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --upgrade pip setuptools wheel
uv pip install --python .venv/bin/python \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python \
  'unsloth[cu121-torch251]' \
  transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf

# Current torchao releases expect newer Torch dtypes than torch 2.5.1 provides.
# Unsloth imports and trains without torchao on this cu121/torch251 stack.
uv pip uninstall --python .venv/bin/python torchao || true

cat <<MSG
Training venv is ready.
Run:
  bash scripts/run_train.sh
MSG
