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

# --- NixOS patches ----------------------------------------------------------
# Triton's bundled ptxas cannot run on NixOS (generic-Linux ELF), its
# ptx_get_version() emits PTX 8.9 that the system CUDA 12.x ptxas rejects,
# and its libcuda discovery hardcodes /sbin/ldconfig. Patch all three so the
# installed wheel survives a venv rebuild. Re-runnable / idempotent.
#
# The patch script imports triton, which needs libz/libstdc++ on the linker
# path at import time (same reason run_train.sh sets LD_LIBRARY_PATH).
if command -v gcc >/dev/null 2>&1; then
  LIBSTDCXX_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
fi
LIBSTDCXX_DIR="${LIBSTDCXX_DIR:-/run/current-system/sw/lib}"
export LD_LIBRARY_PATH="${LIBSTDCXX_DIR}:/run/current-system/sw/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python scripts/patch_triton_nixos.py

cat <<MSG
Training venv is ready.
Run:
  bash scripts/run_train.sh
MSG
